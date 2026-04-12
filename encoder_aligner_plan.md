# Aligner Encoder Plan

This note captures ideas for adding symmetry-specific evidence encoders in front of the Sinkhorn aligners. It is intentionally separate from the main block-symmetry refactor plan because it is a plausible follow-on design rather than part of the minimum refactor scope.

## Goal

Keep a generic Sinkhorn transport matcher, but let each symmetry compress its owned evidence blocks into a compact descriptor before alignment.

The intended pipeline is:

1. Gather the owned evidence blocks for a symmetry.
2. Encode them into a compact descriptor tensor.
3. Feed the descriptor tensor to a generic Sinkhorn-based aligner.
4. Apply the resulting transport to the symmetry's owned parameters.

Conceptually:

`owned evidence blocks -> symmetry-specific encoder -> compact descriptor -> generic Sinkhorn matcher -> transport matrix`

This reduces parameter count, improves regularization, and avoids very large raw `known_size` projections.

## Generic Interface

The generic aligner should operate on compressed descriptors rather than raw evidence. A clean target interface is:

- encoder output shape for ordinary symmetries: `(batch, descriptor_size, unknown_size)`
- encoder output shape for head symmetry: `(batch, descriptor_size, num_heads)`

The Sinkhorn matcher can then remain generic:

- input: descriptor tensor
- output: soft doubly-stochastic transport matrix

The existing `DimensionAligner` logic can survive as the transport stage, but it should conceptually sit after an encoder rather than consuming raw evidence directly.

## Why Encoders Help

Without compression, some current or planned aligners become very large.

Examples for BERT-base style dimensions:

- `model`: raw known size on the order of vocab or other very large evidence width
- `mlp`: raw known size from multiple large hidden-space blocks
- `head`: raw per-head flattened bundle is roughly 196,800 scalars per head

Using a structured encoder before Sinkhorn should:

- reduce parameter count
- reduce compute in the aligner itself
- regularize the transport prediction
- better respect the block structure than a giant dense projection

## Symmetry-Specific Encoder Sketch

### Model

Inputs:

- embedding tables
- tied model-owned decoder-side blocks when applicable
- hidden-owned bias blocks
- LayerNorm blocks

Encoder:

- family-wise low-rank probes for large matrix families
- small vector encoders for bias and LayerNorm blocks
- small fusion MLP over the concatenated summaries

Suggested descriptor size:

- `128` or `256`

### QK

Inputs:

- query weight
- key weight
- query bias
- key bias

Encoder:

- shared low-rank probes over the `d_model x d_head` weight blocks
- small projections for the two bias vectors
- small bottleneck MLP over the concatenated summaries

Suggested descriptor size:

- `32` or `64`

Sharing:

- one shared encoder for all `qk` instances when dimensions match

### OV

Inputs:

- value weight
- output weight slice
- value bias

Encoder:

- same general pattern as `qk`, but with its own learned weights

Suggested descriptor size:

- `32` or `64`

Sharing:

- one shared encoder for all `ov` instances when dimensions match

### MLP

Inputs:

- intermediate weight
- output weight
- intermediate bias

Do not include the layer output bias if it is owned by hidden space rather than the MLP symmetry itself.

Encoder:

- low-rank probes for the two matrix families
- small projection for the owned bias block
- small fusion MLP

Suggested descriptor size:

- `128`

### Decoder

Inputs:

- transform dense weight
- transform dense bias
- transform LayerNorm blocks
- decoder weight

Encoder:

- one path for hidden-owned decoder-side blocks
- one path for the decoder weight family
- small fusion MLP

Suggested descriptor size:

- `128` or `256`

### Head

Inputs per head, after model-space and within-head canonicalization:

- query weight
- key weight
- value weight
- output weight slice
- query bias
- key bias
- value bias

Do not include:

- attention output bias
- LayerNorm blocks
- other hidden-owned parameters with no head axis

Encoder:

- family-wise low-rank probes or projections for each head-owned family
- concatenate the resulting summaries
- apply a small shared bottleneck MLP to produce one descriptor per head

Suggested descriptor size:

- `64`

This should be shared across heads within a layer and ideally across layers if the dimensions match and the inductive bias is acceptable.

## Concrete Head Encoder Example

For one BERT-base head, the raw owned bundle is approximately:

- query weight: `768 x 64`
- key weight: `768 x 64`
- value weight: `768 x 64`
- output weight slice: `64 x 768`
- query, key, value biases: `3 x 64`

Flattening that directly gives roughly `196,800` scalars per head, which is too large for a direct dense aligner input.

Instead, use:

- low-rank probes with ranks `8` and `8` for each weight family
- small bias projections of width `8`
- a shared MLP from the concatenated family summaries to a descriptor of size `64`

This keeps the head descriptor encoder on the order of tens of thousands of parameters per layer rather than millions.

## Transport Sizes After Compression

If the generic Sinkhorn matcher uses compressed descriptors, its parameter count is approximately:

`2 * descriptor_size * unknown_size`

Examples:

- `model` with descriptor size `128` and unknown size `768`: about `196,608`
- `qk` or `ov` with descriptor size `32` and unknown size `64`: about `4,096`
- `mlp` with descriptor size `128` and unknown size `3072`: about `786,432`
- `head` with descriptor size `64` and `12` heads: about `1,536`

These are much smaller than using raw known dimensions directly.

## Sharing Strategy

Recommended default:

- share encoder weights across instances of the same symmetry family when dimensions match
- keep separate encoders for `model`, `qk`, `ov`, `mlp`, `decoder`, and `head`
- only introduce per-layer encoders if there is clear evidence that shared encoders underfit

This keeps the design regularized and avoids needless parameter growth.

## Execution Order

The intended order for canonicalization remains:

1. Apply the model-space transport.
2. Apply within-symmetry transports such as `qk`, `ov`, `mlp`, and `decoder`.
3. Build head descriptors from the now-canonicalized head-owned blocks.
4. Predict the soft Sinkhorn head transport.
5. Apply the head transport along the head axis.
6. Optionally harden the head transport to a true permutation for exact evaluation or export.

This keeps each aligner focused on a single symmetry problem.

## Scope Guidance

This encoder plan is probably beyond the minimum viable refactor. A pragmatic rollout would be:

1. Complete the block-native storage and transformation refactor first.
2. Add encoder support to the generic aligner interface.
3. Introduce structured encoders for the highest-leverage cases first: `model`, `mlp`, and `head`.
4. Add encoders for `qk`, `ov`, and `decoder` only if needed.

## Open Questions

1. Which symmetry families should share encoder weights across layers, and which should remain layer-specific?
2. Should the output-weight family in head descriptors be stored or normalized in a transposed convention to keep head-axis transport uniform?
3. Should soft head transport always be hardened for evaluation, or only for exact invariance tests and export?
4. Is a single generic transport matcher sufficient for all symmetries once encoders are introduced, or do some symmetries need specialized transport heads?
