# %%
import itertools
import numpy as np
import torch
from transformers import AutoModelForMaskedLM
from lib.canonicalizer import Canonicalizer
from lib.serial_model import serialize_model
from tqdm import tqdm

torch.manual_seed(0)
device = torch.device("cuda")

NUM_TRAIN = 20
NUM_TEST = 5
NUM_TOTAL = NUM_TRAIN + NUM_TEST

all_evidence = []
for seed in tqdm(range(NUM_TOTAL)):
    model = AutoModelForMaskedLM.from_pretrained(f"google/multiberts-seed_{seed}", local_files_only=True).eval()
    all_evidence.append(Canonicalizer._evidence_tensor(serialize_model(model), "model").detach().float().to(device))

train_evidence = torch.stack(all_evidence[:NUM_TRAIN])  # (20, K, 768)
test_evidence = torch.stack(all_evidence[NUM_TRAIN:])   # (5, K, 768)

# %%
# Generalized Procrustes: iteratively align all train models to their mean

def procrustes_align(target, source):
    """Find orthogonal R such that source @ R ≈ target."""
    M = source.T @ target  # (768, 768)
    U, S, Vh = torch.linalg.svd(M)
    return U @ Vh

# Initialize: aligned = copies of originals, Rs = identity
Rs = [torch.eye(768, device=device) for _ in range(NUM_TRAIN)]
aligned = train_evidence.clone()

for iteration in range(50):
    # Compute mean of aligned representations
    mean = aligned.mean(dim=0)  # (K, 768)

    # Re-align each model to the mean
    max_delta = 0.0
    for i in range(NUM_TRAIN):
        R_new = procrustes_align(mean, train_evidence[i])
        aligned[i] = train_evidence[i] @ R_new
        delta = (R_new - Rs[i]).norm().item()
        max_delta = max(max_delta, delta)
        Rs[i] = R_new

    # Evaluate convergence
    pairwise = []
    for i, j in itertools.combinations(range(NUM_TRAIN), 2):
        pairwise.append((aligned[i] - aligned[j]).pow(2).mean().item())
    mse = np.mean(pairwise)

    print(f"Iter {iteration:3d} | train MSE: {mse:.6f} | max R delta: {max_delta:.6f}")
    if max_delta < 1e-6:
        print("Converged.")
        break

# %%
# The canonical basis is defined by the final mean
canonical_mean = aligned.mean(dim=0)  # (K, 768)

# Align test models to the canonical mean
test_aligned = []
for i in range(NUM_TEST):
    R = procrustes_align(canonical_mean, test_evidence[i])
    test_aligned.append(test_evidence[i] @ R)
test_aligned = torch.stack(test_aligned)

# %%
# Evaluate: pairwise MSE among test models after alignment
test_pairs = list(itertools.combinations(range(NUM_TEST), 2))

mse_test_aligned = []
mse_test_identity = []
mse_test_procrustes = []
for i, j in test_pairs:
    A, B = test_aligned[i], test_aligned[j]
    mse_test_aligned.append((A - B).pow(2).mean().item())

    A_raw, B_raw = test_evidence[i], test_evidence[j]
    mse_test_identity.append((A_raw - B_raw).pow(2).mean().item())

    R_direct = procrustes_align(A_raw, B_raw)
    mse_test_procrustes.append((A_raw - B_raw @ R_direct).pow(2).mean().item())

mse_test_aligned = np.array(mse_test_aligned)
mse_test_identity = np.array(mse_test_identity)
mse_test_procrustes = np.array(mse_test_procrustes)

print(f"\n--- Test set (held-out seeds {NUM_TRAIN}-{NUM_TOTAL-1}) ---")
print(f"MSE (identity):              {mse_test_identity.mean():.6f} ± {mse_test_identity.std():.6f}")
print(f"MSE (pairwise procrustes):   {mse_test_procrustes.mean():.6f} ± {mse_test_procrustes.std():.6f}")
print(f"MSE (canonical alignment):   {mse_test_aligned.mean():.6f} ± {mse_test_aligned.std():.6f}")
frac = (mse_test_identity.mean() - mse_test_aligned.mean()) / mse_test_identity.mean() * 100
print(f"Canonical closes {frac:.1f}% of identity MSE")

# %%