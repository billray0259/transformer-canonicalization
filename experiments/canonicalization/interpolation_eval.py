# %%
"""
Evaluate the canonicalization model by comparing three merging strategies
over all C(5,2)=10 pairs of held-out test seeds (20-24):

  1. Canonical merge   – canonicalize both → average weights → uncanonicalize → evaluate
  2. Naive merge       – average weights directly → evaluate
  3. Activation ensemble (oracle) – average output logits from both originals

"Uncanonicalize" applies the inverse of model A's transform to the merged
result.  Mathematically this is equivalent to aligning B into A's frame via
R_B @ R_A^T and then averaging — a pairwise Procrustes merge.

SYMMETRY_SUBSET controls which symmetries participate in the alignment.
Set to None for all, or e.g. {"model"} to test just the global rotation.
"""
import gc
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

from lib.canonicalizer import CascadingTemplateCanonicalizer
from lib.serial_model import _build_overrides, serialize_model

# ── Config ────────────────────────────────────────────────────────────────────
CANON_PATH = Path("data/alignment_models/cascading_rotation_merge_many_reflections.pt")
NUM_TRAIN = 20
NUM_TEST = 5
TEST_SEEDS = list(range(NUM_TRAIN, NUM_TRAIN + NUM_TEST))

SYMMETRY_SUBSET = {"model"}  # Set to None for all symmetries.

EVAL_SEQ_LEN = 128
EVAL_NUM_SEQS = 500
EVAL_BATCH_SIZE = 8
MASK_RATIO = 0.15
MASK_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHELL_SEED = "google/multiberts-seed_0"  # Used only for architecture; weights are replaced.

# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_tokens(input_ids: torch.Tensor, generator: torch.Generator):
    """Return (masked_ids, labels) with MASK_RATIO of tokens replaced."""
    labels = input_ids.clone()
    prob = torch.full(input_ids.shape, MASK_RATIO, device=input_ids.device)
    # Never mask [CLS]/[SEP]/[PAD]
    special = (
        (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
        | (input_ids == tokenizer.pad_token_id)
    )
    prob[special] = 0.0
    mask = torch.bernoulli(prob, generator=generator).bool()
    labels[~mask] = -100
    masked = input_ids.clone()
    masked[mask] = tokenizer.mask_token_id
    return masked, labels


@torch.no_grad()
def eval_perplexity(model: torch.nn.Module) -> float:
    """Pseudo-PPL: average CE on masked tokens across all eval sequences."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    gen = torch.Generator(device=device).manual_seed(MASK_SEED)
    for i in range(0, len(seqs), EVAL_BATCH_SIZE):
        batch = seqs[i : i + EVAL_BATCH_SIZE].to(device)
        masked, labels = mask_tokens(batch, gen)
        loss = model(input_ids=masked, labels=labels).loss
        total_loss += loss.item()
        n_batches += 1
    return float(torch.exp(torch.tensor(total_loss / n_batches)))


@torch.no_grad()
def eval_ensemble_perplexity(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    """Activation-ensemble PPL: average logits from both models, then CE."""
    model_a.eval()
    model_b.eval()
    total_loss = 0.0
    n_batches = 0
    gen = torch.Generator(device=device).manual_seed(MASK_SEED)
    for i in range(0, len(seqs), EVAL_BATCH_SIZE):
        batch = seqs[i : i + EVAL_BATCH_SIZE].to(device)
        masked, labels = mask_tokens(batch, gen)
        logits = (model_a(input_ids=masked).logits + model_b(input_ids=masked).logits) / 2
        active = labels != -100
        loss = F.cross_entropy(logits[active], labels[active])
        total_loss += loss.item()
        n_batches += 1
    return float(torch.exp(torch.tensor(total_loss / n_batches)))


def average_symmeters(sym_a, sym_b):
    """Return a new Symmeters whose component tensors are (A + B) / 2."""
    averaged = sym_a.clone()
    for symmetry_name, components in averaged.items():
        for component_name, component in list(components.items()):
            other_tensor = sym_b[symmetry_name][component_name].tensor
            components[component_name] = component.with_tensor(
                (component.tensor + other_tensor) / 2
            )
    return averaged


def symmeters_to_model(symmeters) -> torch.nn.Module:
    """Load Symmeters weights into a shell HF model and move to device."""
    overrides = _build_overrides(symmeters)
    model = AutoModelForMaskedLM.from_pretrained(SHELL_SEED, local_files_only=True)
    state = model.state_dict()
    for key, tensor in overrides.items():
        state[key] = tensor.to(dtype=state[key].dtype)
    model.load_state_dict(state)
    return model.to(device).eval()


def canonicalize_subset(canonicalizer, symmeters, subset=None):
    """Canonicalize only the given symmetries (or all if subset is None).

    Returns (canonicalized_symmeters, transforms_dict).
    """
    canonicalized = symmeters.clone()
    transforms = {}
    for symmetry_name in canonicalizer.order:
        if subset is not None and symmetry_name not in subset:
            continue
        matrix = canonicalizer.infer_transform(canonicalized, symmetry_name)
        canonicalizer.apply_symmetry_transform(canonicalized, symmetry_name, matrix)
        transforms[symmetry_name] = matrix
    return canonicalized, transforms


def uncanonicalize(canonicalizer, symmeters, transforms):
    """Apply inverse transforms (reverse order) to undo canonicalization.

    For orthogonal R the inverse is R^T.  For permutation P it's also P^T.
    """
    result = symmeters.clone()
    for symmetry_name in reversed(canonicalizer.order):
        if symmetry_name not in transforms:
            continue
        inverse = transforms[symmetry_name].mT
        canonicalizer.apply_symmetry_transform(result, symmetry_name, inverse)
    return result


if __name__ == "__main__":
    # ── Load canonicalizer ────────────────────────────────────────────────────
    print("Loading canonicalizer...")
    canonicalizer = CascadingTemplateCanonicalizer.load(str(CANON_PATH), map_location="cpu")

    subset_label = ", ".join(sorted(SYMMETRY_SUBSET)) if SYMMETRY_SUBSET else "all"
    print(f"Symmetry subset: {subset_label}")

    # ── Tokenizer and eval corpus ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(SHELL_SEED, local_files_only=True)

    print("Loading eval corpus...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(t for t in dataset["text"] if t.strip())
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    # Build fixed non-overlapping chunks.
    n_seqs = min(EVAL_NUM_SEQS, (len(token_ids) - 1) // EVAL_SEQ_LEN)
    seqs = torch.stack([token_ids[i * EVAL_SEQ_LEN:(i + 1) * EVAL_SEQ_LEN] for i in range(n_seqs)])
    print(f"Eval corpus: {n_seqs} sequences × {EVAL_SEQ_LEN} tokens")

    # ── Main evaluation loop ──────────────────────────────────────────────────
    pairs = list(itertools.combinations(TEST_SEEDS, 2))
    print(f"\nEvaluating {len(pairs)} pairs...\n")

    results = []

    for seed_a, seed_b in tqdm(pairs, desc="pairs"):
        tqdm.write(f"\n── Seeds ({seed_a}, {seed_b}) ──")

        # Load original models
        hf_a = AutoModelForMaskedLM.from_pretrained(
            f"google/multiberts-seed_{seed_a}", local_files_only=True
        ).eval()
        hf_b = AutoModelForMaskedLM.from_pretrained(
            f"google/multiberts-seed_{seed_b}", local_files_only=True
        ).eval()

        # Clone immediately to detach from model parameters before .to(device) moves them.
        sym_a = serialize_model(hf_a).clone()
        sym_b = serialize_model(hf_b).clone()

        # ── Activation ensemble (oracle) ──
        model_a_gpu = hf_a.to(device)
        model_b_gpu = hf_b.to(device)
        ppl_ensemble = eval_ensemble_perplexity(model_a_gpu, model_b_gpu)
        tqdm.write(f"  Ensemble (oracle):  PPL = {ppl_ensemble:.2f}")
        del model_a_gpu, model_b_gpu, hf_a, hf_b
        torch.cuda.empty_cache()
        gc.collect()

        # ── Naive weight merge ──
        merged_naive = average_symmeters(sym_a, sym_b)
        model_naive = symmeters_to_model(merged_naive)
        ppl_naive = eval_perplexity(model_naive)
        tqdm.write(f"  Naive merge:        PPL = {ppl_naive:.2f}")
        del model_naive, merged_naive
        torch.cuda.empty_cache()
        gc.collect()

        # ── Canonical weight merge (canonicalize → average → uncanonicalize) ──
        canon_a, transforms_a = canonicalize_subset(canonicalizer, sym_a, SYMMETRY_SUBSET)
        canon_b, _ = canonicalize_subset(canonicalizer, sym_b, SYMMETRY_SUBSET)
        merged_canonical = average_symmeters(canon_a, canon_b)
        merged_uncanon = uncanonicalize(canonicalizer, merged_canonical, transforms_a)
        model_canonical = symmeters_to_model(merged_uncanon)
        ppl_canonical = eval_perplexity(model_canonical)
        tqdm.write(f"  Canonical merge:    PPL = {ppl_canonical:.2f}")
        tqdm.write(f"  Ratio (canon/naive):     {ppl_canonical / ppl_naive:.4f}")
        tqdm.write(f"  Ratio (canon/ensemble):  {ppl_canonical / ppl_ensemble:.4f}")
        del model_canonical, merged_canonical, merged_uncanon, canon_a, canon_b, sym_a, sym_b
        torch.cuda.empty_cache()
        gc.collect()

        results.append({
            "seed_a": seed_a,
            "seed_b": seed_b,
            "ppl_ensemble": ppl_ensemble,
            "ppl_naive": ppl_naive,
            "ppl_canonical": ppl_canonical,
        })


    # ── Summary ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ppl_e = [r["ppl_ensemble"] for r in results]
    ppl_n = [r["ppl_naive"] for r in results]
    ppl_c = [r["ppl_canonical"] for r in results]

    # Absolute PPL
    print(f"\nEnsemble (oracle):  {np.mean(ppl_e):.2f} ± {np.std(ppl_e):.2f}")
    print(f"Naive merge:        {np.mean(ppl_n):.2f} ± {np.std(ppl_n):.2f}")
    print(f"Canonical merge:    {np.mean(ppl_c):.2f} ± {np.std(ppl_c):.2f}")

    # Ratios
    ratios_vs_naive = [c / n for c, n in zip(ppl_c, ppl_n)]
    ratios_vs_ensemble = [c / e for c, e in zip(ppl_c, ppl_e)]

    print(f"\nCanonical / Naive:     {np.mean(ratios_vs_naive):.4f} ± {np.std(ratios_vs_naive):.4f}")
    print(f"Canonical / Ensemble:  {np.mean(ratios_vs_ensemble):.4f} ± {np.std(ratios_vs_ensemble):.4f}")

    # How much of naive→ensemble improvement is captured by canonical merging?
    # improvement_fraction = (naive - canonical) / (naive - ensemble)
    fracs = [(n - c) / (n - e) for n, c, e in zip(ppl_n, ppl_c, ppl_e)]
    print(f"\nFraction of ensemble gain captured by canonical merge:")
    print(f"  {np.mean(fracs):.1%} ± {np.std(fracs):.1%}")

    print("\nPer-pair results:")
    print(f"{'Seeds':>12}  {'Ensemble':>10}  {'Naive':>10}  {'Canonical':>10}  {'C/N':>6}  {'C/E':>6}")
    for r in results:
        print(
            f"  ({r['seed_a']},{r['seed_b']}):   "
            f"{r['ppl_ensemble']:>10.2f}  "
            f"{r['ppl_naive']:>10.2f}  "
            f"{r['ppl_canonical']:>10.2f}  "
            f"{r['ppl_canonical']/r['ppl_naive']:>6.3f}  "
            f"{r['ppl_canonical']/r['ppl_ensemble']:>6.3f}"
        )

    # %%
