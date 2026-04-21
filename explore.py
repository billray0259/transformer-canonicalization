# %%
import contextlib
import io
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as hf_logging
from lib.canonicalizer import CascadingTemplateCanonicalizer
from lib.serial_model import _build_overrides, serialize_model
from experiments.canonicalization.interpolation_eval import canonicalize_subset, uncanonicalize, symmeters_to_model

hf_logging.set_verbosity_error()

CKPT_PATH = "data/alignment_models/cascading_rotation_merge_many.ckpt.pt"
SHELL_SEED = "google/multiberts-seed_0"
NUM_TRAIN = 20
NUM_TEST = 5
TEST_SEEDS = list(range(NUM_TRAIN, NUM_TRAIN + NUM_TEST))

ckpt = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")
completed = ckpt["completed"]
canonicalizer = CascadingTemplateCanonicalizer(completed, {k: ckpt["templates"][k] for k in completed})

# %%
print("Checking determinants of 'model' transforms for each test seed:\n")
for seed in TEST_SEEDS:
    with contextlib.redirect_stdout(io.StringIO()):
        model = AutoModelForMaskedLM.from_pretrained(
            f"google/multiberts-seed_{seed}", local_files_only=True
        ).eval()
    sym = serialize_model(model).clone()
    del model

    _, transforms = canonicalize_subset(canonicalizer, sym, {"model"})
    R = transforms["model"]
    sign, logabsdet = torch.linalg.slogdet(R.double())
    print(f"  Seed {seed}: sign={sign.item():+.0f}  logabsdet={logabsdet.item():.6f}  shape={tuple(R.shape)}")

# %%
# Single-model round-trip test: canonicalize → uncanonicalize should be identity.
# Uses seed 20 as a representative test case.
ROUNDTRIP_SEED = TEST_SEEDS[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nRound-trip test for seed {ROUNDTRIP_SEED}:")

with contextlib.redirect_stdout(io.StringIO()):
    rt_model = AutoModelForMaskedLM.from_pretrained(
        f"google/multiberts-seed_{ROUNDTRIP_SEED}", local_files_only=True
    ).eval()

sym = serialize_model(rt_model).clone()
del rt_model

# Build a small eval corpus for quick PPL checks
tokenizer = AutoTokenizer.from_pretrained(SHELL_SEED, local_files_only=True)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = " ".join(t for t in dataset["text"] if t.strip())
token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
SEQ_LEN, N_SEQS, BATCH = 128, 50, 8
seqs = torch.stack([token_ids[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(N_SEQS)])

@torch.no_grad()
def quick_ppl(model):
    model.eval()
    total, n = 0.0, 0
    gen = torch.Generator(device=device).manual_seed(42)
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i:i + BATCH].to(device)
        prob = torch.full(batch.shape, 0.15, device=device)
        mask = torch.bernoulli(prob, generator=gen).bool()
        labels = batch.clone()
        labels[~mask] = -100
        masked = batch.clone()
        masked[mask] = tokenizer.mask_token_id
        total += model(input_ids=masked, labels=labels).loss.item()
        n += 1
    return float(torch.exp(torch.tensor(total / n)))

model_before = symmeters_to_model(sym)
ppl_before = quick_ppl(model_before)
del model_before

canon, transforms = canonicalize_subset(canonicalizer, sym, {"model"})
recovered = uncanonicalize(canonicalizer, canon, transforms)

model_after = symmeters_to_model(recovered)
ppl_after = quick_ppl(model_after)
del model_after

print(f"  PPL before:  {ppl_before:.4f}")
print(f"  PPL after:   {ppl_after:.4f}")

# Weight-level deviation
overrides_before = _build_overrides(sym)
overrides_after = _build_overrides(recovered)
large_diffs = {}
for key in overrides_before:
    diff = (overrides_before[key].float() - overrides_after[key].float()).abs().max().item()
    if diff > 1e-4:
        large_diffs[key] = diff

if large_diffs:
    print(f"\n  Parameters with max_diff > 1e-4:")
    for key, diff in sorted(large_diffs.items(), key=lambda x: -x[1])[:10]:
        print(f"    {key}: {diff:.6f}")
else:
    print("\n  All weights within 1e-4 — round-trip is clean.")

# %%
# Weight-space similarity before and after canonicalization.
# Check whether canonicalization brings weights closer together.
# Pairs: (22,24) was a good pair (C/N≈0.373), (21,23) was a bad pair (C/N≈12.6).

def weight_mse(sym_a, sym_b):
    ov_a = _build_overrides(sym_a)
    ov_b = _build_overrides(sym_b)
    total, count = 0.0, 0
    for key in ov_a:
        total += (ov_a[key].float() - ov_b[key].float()).pow(2).mean().item()
        count += 1
    return total / count

print("\nWeight-space MSE before/after model canonicalization:\n")
for seed_a, seed_b in [(22, 24), (21, 23)]:
    with contextlib.redirect_stdout(io.StringIO()):
        hf_a = AutoModelForMaskedLM.from_pretrained(f"google/multiberts-seed_{seed_a}", local_files_only=True).eval()
        hf_b = AutoModelForMaskedLM.from_pretrained(f"google/multiberts-seed_{seed_b}", local_files_only=True).eval()
    sa = serialize_model(hf_a).clone()
    sb = serialize_model(hf_b).clone()
    del hf_a, hf_b

    mse_raw = weight_mse(sa, sb)
    ca, _ = canonicalize_subset(canonicalizer, sa, {"model"})
    cb, _ = canonicalize_subset(canonicalizer, sb, {"model"})
    mse_canon = weight_mse(ca, cb)

    delta = mse_canon - mse_raw
    direction = "↓ better" if delta < 0 else "↑ WORSE"
    print(f"  ({seed_a},{seed_b})  raw={mse_raw:.6f}  canon={mse_canon:.6f}  Δ={delta:+.6f}  {direction}")

# %%
