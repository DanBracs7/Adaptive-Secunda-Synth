"""
Generation speed with and without the key/value cache.

Prints the real-time factor (generation time / duration of the generated
audio; below 1 = faster than real time) for one 10-second clip.

Run from the repository root:
    python src/benchmark_generation.py                      # random weights, same size as the trained model
    python src/benchmark_generation.py --checkpoint checkpoints/secunda_e4_R2_final_model.pt
    python src/benchmark_generation.py --device cuda --no-compare
"""

import argparse
import sys

import torch

from generation import Timer, generate_codes, real_time_factor
from model import SecundaTransformer


def build_model(checkpoint, device):
    if checkpoint is None:
        torch.manual_seed(0)
        return SecundaTransformer().to(device).eval()
    from training_utils import load_checkpoint_model
    model, _ = load_checkpoint_model(checkpoint, device)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=750, help="total frames (75 = 1 s)")
    parser.add_argument("--prompt-frames", type=int, default=75)
    parser.add_argument("--no-compare", action="store_true", help="skip the slow run without cache")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model = build_model(args.checkpoint, device)
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 1024, (1, 8, args.prompt_frames), generator=generator).to(device)
    controls = torch.full((1, args.frames), 0.5, device=device)
    generated = args.frames - args.prompt_frames

    print(f"Device: {device}, threads: {torch.get_num_threads()}, "
          f"{generated} generated frames ({generated / 75:.1f} s of audio)")
    for use_cache in ([True] if args.no_compare else [True, False]):
        with Timer() as timer:
            generate_codes(model, prompt, controls, controls, seed=0, use_cache=use_cache)
        print(f"cache={use_cache!s:5s}: {timer.seconds:6.1f} s, "
              f"real-time factor {real_time_factor(timer.seconds, generated):.2f}")


if __name__ == "__main__":
    sys.exit(main())
