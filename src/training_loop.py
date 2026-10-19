"""
Resumable training loop, used by the control runs of experiment E4.

It reproduces the training procedure of the archived notebooks
(notebooks/archive/slakh-baseline-training-8kb.ipynb and
finetune-of-slakh-baseline-training-8kb.ipynb): AdamW with weight decay on
weights only, per-step warmup + cosine learning rate (training_utils.lr_at_step),
gradient clipping at 1.0, the weighted multi-codebook loss
(training_utils.compute_batch_metrics), fp32, validation after every epoch.

Two things are new, both needed to split a run over several Kaggle
sessions (12-hour limit) without changing the result:
  * the training order is fixed by (seed, epoch) through
    ResumableEpochSampler, and a resumed run starts exactly at the next
    unseen batch instead of re-drawing a random order;
  * checkpoints store the random-number-generator states, so dropout
    after a resume is the same as in an uninterrupted run.
tests/test_training_loop.py checks that a run interrupted and resumed
ends with exactly the same weights as an uninterrupted run.

Runs stop at a fixed number of optimizer steps (max_steps), not at a
number of epochs, so runs on datasets of different size get the same
amount of training.
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from training_utils import compute_batch_metrics, evaluate_on_loader, lr_at_step


def build_optimizer(model, peak_lr, weight_decay):
    """AdamW with weight decay on weights only (biases and norms excluded), as in the archived notebooks."""
    decay_parameters = []
    no_decay_parameters = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_bias = parameter_name.endswith("bias")
        is_norm = "norm" in parameter_name.lower() or "layernorm" in parameter_name.lower()
        if is_bias or is_norm:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=peak_lr,
    )


class ResumableEpochSampler(Sampler):
    """
    Random order of the dataset indices, fixed by (seed, epoch).
    set_epoch(epoch, start_index) skips the first start_index indices of
    that order, so a resumed run continues with the next unseen samples.
    """

    def __init__(self, num_samples, seed):
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 1
        self.start_index = 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch = epoch
        self.start_index = start_index

    def order(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(self.num_samples, generator=generator).tolist()

    def __iter__(self):
        return iter(self.order()[self.start_index:])

    def __len__(self):
        return self.num_samples - self.start_index


def filter_rows(dataset, keep):
    """Keep only the manifest rows for which keep(row) is True (e.g. a subset of tracks)."""
    dataset.rows = [row for row in dataset.rows if keep(row)]
    if not dataset.rows:
        raise ValueError("No rows left after filtering.")
    return dataset


def new_history():
    return {
        "epoch": [], "global_step": [], "learning_rate": [],
        "train_loss": [], "val_loss": [],
        "train_per_codebook_loss": [], "val_per_codebook_loss": [],
        "val_per_codebook_accuracy": [],
    }


def save_training_checkpoint(path, model, optimizer, state, config):
    checkpoint = {
        "epoch": state["epoch"],
        "batch_index": state["batch_index"],
        "global_step": state["global_step"],
        "best_val_loss": state["best_val_loss"],
        "history": state["history"],
        "finished": state["finished"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary_path = Path(str(path) + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)   # never leave a half-written checkpoint


def load_training_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state["torch"].cpu())
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])

    return {
        "epoch": checkpoint["epoch"],
        "batch_index": checkpoint["batch_index"],
        "global_step": checkpoint["global_step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "history": checkpoint["history"],
        "finished": checkpoint.get("finished", False),
    }


def train(
    model,
    optimizer,
    train_dataset,
    val_loader,
    serializer,
    device,
    config,
    output_dir,
    resume_path=None,
    time_budget_seconds=None,
    stop_at_step=None,
    log=print,
):
    """
    Train until config["max_steps"] optimizer steps.

    config keys used here: batch_size, num_workers, peak_lr, min_lr,
    warmup_steps, max_steps, codebook_loss_weights, grad_clip_norm, seed,
    save_every_steps. The whole config is stored in the checkpoints (it
    must also contain the model hyperparameters, so that
    training_utils.load_checkpoint_model can rebuild the model).

    The run stops early, after saving `last.pt`, when time_budget_seconds
    have passed (Kaggle session limit) or at stop_at_step (tests); calling
    train again with resume_path=output_dir / "last.pt" continues it.
    Returns the training state.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    final_path = output_dir / "final.pt"

    if resume_path is not None:
        state = load_training_checkpoint(resume_path, model, optimizer, device)
        log(f"Resumed from {resume_path}: epoch {state['epoch']}, batch {state['batch_index']}, "
            f"step {state['global_step']}")
    else:
        state = {"epoch": 1, "batch_index": 0, "global_step": 0,
                 "best_val_loss": float("inf"), "history": new_history(), "finished": False}

    if state["finished"]:
        log("This run is already finished.")
        return state

    batch_size = config["batch_size"]
    weights = config["codebook_loss_weights"]
    sampler = ResumableEpochSampler(len(train_dataset), config["seed"])
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    started = time.time()
    session_steps = 0

    while state["global_step"] < config["max_steps"]:
        epoch = state["epoch"]
        train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch, start_index=state["batch_index"] * batch_size)
        # A dedicated generator keeps the DataLoader from drawing its worker
        # seed from the global RNG, which also drives dropout: otherwise a
        # resumed run would get different dropout masks.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(config["seed"] * 1_000_003 + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
            generator=loader_generator,
        )

        model.train()
        epoch_loss_sum = 0.0
        epoch_codebook_loss_sum = None
        epoch_batches = 0
        stop_reason = None

        for batch in loader:
            state["global_step"] += 1
            learning_rate = lr_at_step(
                state["global_step"], config["peak_lr"], config["min_lr"],
                config["warmup_steps"], config["max_steps"],
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            loss, per_codebook_loss, _ = compute_batch_metrics(
                batch=batch, model=model, serializer=serializer,
                device=device, codebook_loss_weights=weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {state['global_step']}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
            optimizer.step()

            state["batch_index"] += 1
            session_steps += 1
            epoch_loss_sum += loss.item()
            detached = per_codebook_loss.detach()
            epoch_codebook_loss_sum = detached if epoch_codebook_loss_sum is None else epoch_codebook_loss_sum + detached
            epoch_batches += 1

            if state["global_step"] % config.get("log_every_steps", 50) == 0:
                elapsed = time.time() - started
                seconds_per_step = elapsed / session_steps
                remaining = (config["max_steps"] - state["global_step"]) * seconds_per_step
                log(f"step {state['global_step']}/{config['max_steps']} | epoch {epoch} "
                    f"batch {state['batch_index']}/{steps_per_epoch} | loss {loss.item():.4f} | "
                    f"lr {learning_rate:.2e} | {seconds_per_step:.2f} s/step | "
                    f"remaining {remaining / 3600:.1f} h")

            if state["global_step"] % config["save_every_steps"] == 0:
                save_training_checkpoint(last_path, model, optimizer, state, config)

            if state["global_step"] >= config["max_steps"]:
                break
            if stop_at_step is not None and state["global_step"] >= stop_at_step:
                stop_reason = f"stop_at_step={stop_at_step}"
                break
            if time_budget_seconds is not None and time.time() - started > time_budget_seconds:
                stop_reason = "time budget reached"
                break

        if stop_reason is not None:
            save_training_checkpoint(last_path, model, optimizer, state, config)
            log(f"Stopped ({stop_reason}) at step {state['global_step']}; resume from {last_path}.")
            return state

        # End of an epoch, or max_steps reached: validate.
        val_metrics = evaluate_on_loader(model, val_loader, serializer, device, weights)
        history = state["history"]
        history["epoch"].append(epoch)
        history["global_step"].append(state["global_step"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(epoch_loss_sum / max(1, epoch_batches))
        history["train_per_codebook_loss"].append(
            (epoch_codebook_loss_sum / max(1, epoch_batches)).cpu().tolist()
            if epoch_codebook_loss_sum is not None else None
        )
        history["val_loss"].append(val_metrics["loss"])
        history["val_per_codebook_loss"].append(val_metrics["per_codebook_loss"])
        history["val_per_codebook_accuracy"].append(val_metrics["per_codebook_accuracy"])
        log(f"epoch {epoch} done at step {state['global_step']} | "
            f"train {history['train_loss'][-1]:.4f} | val {val_metrics['loss']:.4f}")

        state["epoch"], state["batch_index"] = epoch + 1, 0
        if state["global_step"] >= config["max_steps"]:
            state["finished"] = True

        if val_metrics["loss"] < state["best_val_loss"]:
            state["best_val_loss"] = val_metrics["loss"]
            save_training_checkpoint(best_path, model, optimizer, state, config)
        if state["finished"]:
            save_training_checkpoint(final_path, model, optimizer, state, config)
        save_training_checkpoint(last_path, model, optimizer, state, config)

        with open(output_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(state["history"], file, indent=2)

        if time_budget_seconds is not None and time.time() - started > time_budget_seconds and not state["finished"]:
            log(f"Time budget reached after epoch {epoch}; resume from {last_path}.")
            return state

    state["finished"] = True
    return state
