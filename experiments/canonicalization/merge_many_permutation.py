# %%
import itertools
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForMaskedLM
from lib.canonicalizer import Canonicalizer, PermutationAligner
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

aligner = PermutationAligner(all_evidence[0].shape[0], all_evidence[0].shape[1]).to(device)
aligner.load_state_dict(torch.load("data/alignment_models/permutation_aligner_00177.pth"))
aligner = aligner.to(device).eval()

train_evidence = torch.stack(all_evidence[:NUM_TRAIN])
test_evidence = torch.stack(all_evidence[NUM_TRAIN:])

# %%
def permutation_align(target, source):
    """Find permutation P such that source @ P ≈ target."""
    C = (source.T @ target).cpu().numpy()  # (768, 768)
    _, col_ind = linear_sum_assignment(C, maximize=True)
    return torch.eye(768, device=source.device)[col_ind]

Ps = [torch.eye(768, device=device) for _ in range(NUM_TRAIN)]
aligned = train_evidence.clone()

for iteration in range(50):
    mean = aligned.mean(dim=0)

    max_delta = 0.0
    for i in range(NUM_TRAIN):
        P_new = permutation_align(mean, train_evidence[i])
        aligned[i] = train_evidence[i] @ P_new
        delta = (P_new - Ps[i]).norm().item()
        max_delta = max(max_delta, delta)
        Ps[i] = P_new

    pairwise = []
    for i, j in itertools.combinations(range(NUM_TRAIN), 2):
        pairwise.append((aligned[i] - aligned[j]).pow(2).mean().item())
    mse = np.mean(pairwise)

    print(f"Iter {iteration:3d} | train MSE: {mse:.6f} | max P delta: {max_delta:.6f}")
    if max_delta < 1e-6:
        print("Converged.")
        break

# %%
canonical_mean = aligned.mean(dim=0)

test_aligned = []
for i in range(NUM_TEST):
    P = permutation_align(canonical_mean, test_evidence[i])
    test_aligned.append(test_evidence[i] @ P)
test_aligned = torch.stack(test_aligned)

# %%
test_pairs = list(itertools.combinations(range(NUM_TEST), 2))

mse_test_aligned = []
mse_test_identity = []
mse_test_hungarian = []
for i, j in test_pairs:
    A, B = test_aligned[i], test_aligned[j]
    mse_test_aligned.append((A - B).pow(2).mean().item())

    A_raw, B_raw = test_evidence[i], test_evidence[j]
    mse_test_identity.append((A_raw - B_raw).pow(2).mean().item())

    P_direct = permutation_align(A_raw, B_raw)
    mse_test_hungarian.append((A_raw - B_raw @ P_direct).pow(2).mean().item())

mse_test_aligned = np.array(mse_test_aligned)
mse_test_identity = np.array(mse_test_identity)
mse_test_hungarian = np.array(mse_test_hungarian)

print(f"\n--- Test set (held-out seeds {NUM_TRAIN}-{NUM_TOTAL-1}) ---")
print(f"MSE (identity):              {mse_test_identity.mean():.6f} ± {mse_test_identity.std():.6f}")
print(f"MSE (pairwise hungarian):    {mse_test_hungarian.mean():.6f} ± {mse_test_hungarian.std():.6f}")
print(f"MSE (canonical permutation): {mse_test_aligned.mean():.6f} ± {mse_test_aligned.std():.6f}")
frac = (mse_test_identity.mean() - mse_test_aligned.mean()) / mse_test_identity.mean() * 100
print(f"Canonical closes {frac:.1f}% of identity MSE")

# %%

def harden_permutation(soft_P):
    """Convert soft doubly-stochastic matrix to nearest hard permutation.
    
    Args:
        soft_P: (D, D) or (batch, D, D) soft permutation matrix
    Returns:
        Hard permutation matrix of same shape
    """
    squeezed = soft_P.dim() == 2
    if squeezed:
        soft_P = soft_P.unsqueeze(0)

    hard = []
    for P in soft_P:
        _, col_ind = linear_sum_assignment(P.cpu().numpy(), maximize=True)
        hard.append(torch.eye(P.shape[0], device=soft_P.device)[col_ind])

    result = torch.stack(hard)
    return result.squeeze(0) if squeezed else result

with torch.no_grad():
    test_aligned = []
    for i in range(NUM_TEST):
        evidence = test_evidence[i].unsqueeze(0)
        P, _ = aligner(evidence, tau=0.05)  # use low tau for sharp permutation
        P = P.squeeze(0)
        P = harden_permutation(P)
        test_aligned.append(test_evidence[i] @ P)
    test_aligned = torch.stack(test_aligned)

    test_pairs = list(itertools.combinations(range(NUM_TEST), 2))
    mse_vals = []
    for i, j in test_pairs:
        mse_vals.append((test_aligned[i] - test_aligned[j]).pow(2).mean().item())
    mse_arr = np.array(mse_vals)

    print(f"MSE (learned aligner):       {mse_arr.mean():.6f} ± {mse_arr.std():.6f}")
    print(f"Compare to:")
    print(f"  Canonical rotation:        0.001757 ± 0.000017")
    print(f"  Canonical permutation:     0.004034 ± 0.000022")
    print(f"  Identity:                  0.004716 ± 0.000029")