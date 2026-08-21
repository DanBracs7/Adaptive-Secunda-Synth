import torch

# Simulate a real Slakh .pt file's token shape
n_q, T = 32, 9000  # 24 kbps, 3-minute track, 50 frames/sec
tokens = torch.randint(0, 1024, (n_q, T))
print("Raw tokens shape:", tokens.shape)  # [32, 9000]

# The WRONG way (what you must avoid)
flat_wrong = tokens.T.reshape(-1)
print("Flattened (avoid this):", flat_wrong.shape)  # [288000]

# The RIGHT way (what your Dataset class must produce)
correct = tokens.T
print("Correct per-timestep shape:", correct.shape)  # [9000, 32]