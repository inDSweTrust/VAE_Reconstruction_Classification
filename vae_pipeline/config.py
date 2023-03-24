import torch

CFG = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
