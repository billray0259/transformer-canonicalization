# %%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from transformers import AutoModelForMaskedLM

from lib.canonicalizer import Canonicalizer, DimensionAligner
from lib.serial_model import serialize_model
from tqdm import tqdm
import matplotlib.pyplot as plt

def col_diffs(matrix):
    col_min = matrix.min(dim=-1).values
    col_max = matrix.max(dim=-1).values
    col_diff = col_max - col_min
    return col_diff

def row_diffs(matrix):
    row_min = matrix.min(dim=-2).values
    row_max = matrix.max(dim=-2).values
    row_diff = row_max - row_min
    return row_diff

def all_diffs(matrix):
    return torch.cat([col_diffs(matrix), row_diffs(matrix)])


torch.manual_seed(0)
device = torch.device("cuda")

model = AutoModelForMaskedLM.from_pretrained(
    "google/multiberts-seed_0",
    local_files_only=True,
).eval()
symmeters = serialize_model(model)

x = Canonicalizer._evidence_tensor(symmeters, "model").detach().to(device)

perm_generator = torch.Generator(device=device).manual_seed(1)

aligner = DimensionAligner(x.shape[0], x.shape[1]).to(device)

optimizer = torch.optim.Adam(
    aligner.parameters(),
    lr=1e-3,
)

n_steps = 100
batch_size = 8
for step in tqdm(range(n_steps)):
    tau = 1.0
    perms = torch.stack([
        torch.eye(x.shape[-1], device=device)[torch.randperm(x.shape[-1], generator=perm_generator, device=device)]
        for _ in range(batch_size)
    ])
    x_batch = x.unsqueeze(0) @ perms  # (batch, known, unknown)
    targets = perms.transpose(-1, -2).argmax(dim=-1)  # (batch, unknown)
    P, logits = aligner(x_batch, tau=tau)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    # reconstruction = P.unsqueeze(0) @ x_batch
    # loss = F.mse_loss(reconstruction, x_batch)
    accuracy = (P.argmax(dim=-1) == targets).float().mean().item()
    if step % 25 == 0 or step == n_steps - 1:
        diffs = all_diffs(P)
        plt.figure(figsize=(6, 6))
        plt.hist(diffs.detach().cpu().numpy().flatten(), bins=20, color="blue", alpha=0.7, range=(0, 1))
        plt.title(f"Step {step}: Average Diff={torch.mean(diffs).item():.6f}, Accuracy={accuracy:.6f}")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# %%