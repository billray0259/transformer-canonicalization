# %%
import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForMaskedLM
from lib.canonicalizer import Canonicalizer
from lib.serial_model import serialize_model
from tqdm import tqdm

torch.manual_seed(0)
device = torch.device("cuda")

NUM_SEEDS = 25

# Load evidence tensors for all models
all_evidence = []
for seed in tqdm(range(NUM_SEEDS)):
    model = AutoModelForMaskedLM.from_pretrained(f"google/multiberts-seed_{seed}", local_files_only=True).eval()
    all_evidence.append(Canonicalizer._evidence_tensor(serialize_model(model), "model").detach().float().to(device))

# Evaluate all ordered pairs (i, j) with i != j
pairs = list(itertools.combinations(range(NUM_SEEDS), 2))

# %%
# Baseline 1: identity (no alignment)
mse_identity_vals = []
for i, j in tqdm(pairs, desc="identity"):
    mse_identity_vals.append((all_evidence[i] - all_evidence[j]).pow(2).mean().item())

mse_identity_arr = np.array(mse_identity_vals)
print(f"MSE (identity):   {mse_identity_arr.mean():.6f} ± {mse_identity_arr.std():.6f}  "
      f"[n={len(mse_identity_arr)} pairs]")

# %%
# Baseline 2: Hungarian algorithm on weight matching objective
# maximize <A, B @ P^T> = maximize trace(A^T @ B @ P^T)
# cost matrix C[i,j] = sum over rows of A[:,i] * B[:,j]
# i.e. C = A^T @ B, then solve LAP to maximize

mse_hungarian_vals = []
for i, j in tqdm(pairs, desc="hungarian"):
    A, B = all_evidence[i], all_evidence[j]
    C = (A.T @ B).cpu().numpy()  # (768, 768)
    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    P = torch.eye(768, device=device)[col_ind]  # permutation matrix
    B_aligned = B @ P.T
    mse_hungarian_vals.append((A - B_aligned).pow(2).mean().item())

mse_hungarian_arr = np.array(mse_hungarian_vals)
print(f"MSE (hungarian):  {mse_hungarian_arr.mean():.6f} ± {mse_hungarian_arr.std():.6f}  "
      f"[n={len(mse_hungarian_arr)} pairs]")

# %%
# Baseline 3: random permutation (sanity check, average over a few per pair)
RAND_TRIALS = 10
mse_rand_vals = []
for i, j in tqdm(pairs, desc="random"):
    A, B = all_evidence[i], all_evidence[j]
    pair_vals = []
    for _ in range(RAND_TRIALS):
        perm = torch.randperm(768, device=device)
        B_rand = B @ torch.eye(768, device=device)[perm].T
        pair_vals.append((A - B_rand).pow(2).mean().item())
    mse_rand_vals.append(np.mean(pair_vals))

mse_rand_arr = np.array(mse_rand_vals)
print(f"MSE (random avg): {mse_rand_arr.mean():.6f} ± {mse_rand_arr.std():.6f}  "
      f"[n={len(mse_rand_arr)} pairs, {RAND_TRIALS} trials each]")

# %%
# Check: how much of the gap does hungarian close?
frac_vals = (mse_identity_arr - mse_hungarian_arr) / mse_identity_arr * 100
print(f"\nHungarian closes {frac_vals.mean():.1f}% ± {frac_vals.std():.1f}% of identity MSE")
print(f"Random is {(mse_rand_arr / mse_identity_arr).mean():.2f}x ± {(mse_rand_arr / mse_identity_arr).std():.2f}x identity MSE")
# %% 
# %%
# Baseline 4: Orthogonal Procrustes alignment
# Given A, B, find orthogonal R minimizing ||A - B @ R^T||^2
# Solution: SVD of A^T @ B = U S V^T, then R = V @ U^T

mse_procrustes_vals = []
for i, j in tqdm(pairs, desc="procrustes"):
    A, B = all_evidence[i], all_evidence[j]
    M = A.T @ B  # (768, 768)
    U, S, Vh = torch.linalg.svd(M)
    R = Vh.T @ U.T  # orthogonal matrix
    B_aligned = B @ R
    mse_procrustes_vals.append((A - B_aligned).pow(2).mean().item())

mse_procrustes_arr = np.array(mse_procrustes_vals)
print(f"MSE (procrustes): {mse_procrustes_arr.mean():.6f} ± {mse_procrustes_arr.std():.6f}  "
      f"[n={len(mse_procrustes_arr)} pairs]")

frac_orth = (mse_identity_arr - mse_procrustes_arr) / mse_identity_arr * 100
print(f"Procrustes closes {frac_orth.mean():.1f}% ± {frac_orth.std():.1f}% of identity MSE")

# %%
# Baseline 5: Arbitrary linear alignment (least squares)
# Find T minimizing ||A - B @ T||^2, solution: T = (B^T B)^{-1} B^T A

mse_linear_vals = []
for i, j in tqdm(pairs, desc="linear"):
    A, B = all_evidence[i], all_evidence[j]
    T = torch.linalg.lstsq(B, A).solution  # (768, 768)
    B_aligned = B @ T
    mse_linear_vals.append((A - B_aligned).pow(2).mean().item())

mse_linear_arr = np.array(mse_linear_vals)
print(f"MSE (linear):     {mse_linear_arr.mean():.6f} ± {mse_linear_arr.std():.6f}  "
      f"[n={len(mse_linear_arr)} pairs]")

frac_lin = (mse_identity_arr - mse_linear_arr) / mse_identity_arr * 100
print(f"Linear closes {frac_lin.mean():.1f}% ± {frac_lin.std():.1f}% of identity MSE")
print(f"Compare to Procrustes: {frac_orth.mean():.1f}%")
# %% 

# %%
# Baseline 6: Orthogonal + diagonal scaling
# Decompose via polar decomposition: T = R @ S where R orthogonal, S symmetric positive definite
# Then approximate S with just its diagonal

mse_orthdiag_vals = []
for i, j in tqdm(pairs, desc="orth+diag"):
    A, B = all_evidence[i], all_evidence[j]
    T = torch.linalg.lstsq(B, A).solution
    # Polar decomposition: T = R @ S
    U, S, Vh = torch.linalg.svd(T)
    R = U @ Vh
    S_mat = Vh.T @ torch.diag(S) @ Vh  # symmetric positive definite part
    D = torch.diag(S_mat).sqrt()        # diagonal approximation
    T_approx = R @ torch.diag(D)
    B_aligned = B @ T_approx
    mse_orthdiag_vals.append((A - B_aligned).pow(2).mean().item())

mse_orthdiag_arr = np.array(mse_orthdiag_vals)
print(f"MSE (orth+diag):  {mse_orthdiag_arr.mean():.6f} ± {mse_orthdiag_arr.std():.6f}")
frac_od = (mse_identity_arr - mse_orthdiag_arr) / mse_identity_arr * 100
print(f"Orth+diag closes {frac_od.mean():.1f}% ± {frac_od.std():.1f}% of identity MSE")
print(f"Compare to: Procrustes {frac_orth.mean():.1f}% | Linear {frac_lin.mean():.1f}%")

# %%
cond_vals = []
for i, j in tqdm(pairs, desc="cond"):
    A, B = all_evidence[i], all_evidence[j]
    T = torch.linalg.lstsq(B, A).solution
    cond_vals.append(torch.linalg.cond(T).item())
cond_arr = np.array(cond_vals)
print(f"Condition number: {cond_arr.mean():.1f} ± {cond_arr.std():.1f}")