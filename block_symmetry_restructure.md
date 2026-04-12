# Proposed Block-Structured Symmeters

## Motivation

The current `Symmeters` format is optimized for flat row storage. In practice, most width-`d_model` rows are stored in the `model` symmetry, and downstream code has to recover semantic groupings such as:

- query/key blocks for a head
- value/output blocks for a head
- MLP intermediate/output blocks for a layer
- decoder-hidden transform blocks

This makes training-time code more complex than it needs to be. Scope builders and canonicalization logic spend a significant amount of effort finding, validating, and regrouping rows that already belong together semantically.

The proposed refactor changes the internal representation so each symmetry stores its own structured parameter blocks directly.

## Core Representation

Instead of representing most tensors as flat row collections, represent each symmetry as a stack of blocks with shape:

$$
(\text{num_blocks}, \text{known_dim}, \text{unknown_dim})
$$

where:

- `num_blocks` is the number of semantically distinct blocks in the symmetry
- `known_dim` is the dimension shared with the left-side transform
- `unknown_dim` is the dimension canonicalized by the symmetry-specific right-side transform

Examples:

- `qk`: blocks are `[query, key]`
- `ov`: blocks are `[value, output]`
- `mlp`: blocks are `[intermediate, output]`
- decoder-hidden symmetry: blocks are the hidden-dimension blocks used by the decoder transform
- model symmetry: blocks are embedding- or vocab-conditioned blocks used to infer the hidden canonical basis

## Bias Convention

Biases are structurally different from weight rows, but they can still be folded into this representation by appending each bias vector as the final row of its block.

That gives each block the affine form:

$$
(\text{known_dim} + 1, \text{unknown_dim})
$$

with the final row reserved for bias.

This is attractive because:

- the block remains a single object
- the aligner sees weight and bias evidence together
- the bias is always in a fixed, interpretable location

For blocks without a natural bias, a zero padding row can be added so the affine convention is uniform.

## Canonicalization Math

Let:

- $P_{model} \in \mathbb{R}^{d \times d}$ be the hidden-space canonicalization matrix
- $P_{sym} \in \mathbb{R}^{k \times k}$ be the symmetry-specific matrix for some symmetry of width $k$

For affine blocks, define the augmented model transform:

$$
\tilde P_{model} =
\begin{bmatrix}
P_{model} & 0 \\
0 & 1
\end{bmatrix}
\in \mathbb{R}^{(d+1) \times (d+1)}
$$

This applies the model transform to the weight rows while preserving the final bias row as an affine coordinate.

For a non-model symmetry block tensor:

$$
X_{sym} \in \mathbb{R}^{B \times (d+1) \times k}
$$

the canonicalization is:

$$
X'_{sym}[b] = \tilde P_{model} \, X_{sym}[b] \, P_{sym}
$$

for each block $b$.

Equivalently, the left action is broadcast over the block axis, while the right action is shared across all blocks in the symmetry.

This is preferable to flattening the blocks into one large matrix and allowing an unrestricted left multiplication over the concatenated rows.

## Model Symmetry

Under the same convention, the `model` symmetry would also be block-structured. A likely form is:

$$
X_{model} \in \mathbb{R}^{B_m \times (v + 1) \times d}
$$

where:

- $v$ is vocab size
- the final row can be a real bias row or a zero padding row, depending on the block
- blocks might include embeddings and unembedding/output blocks

In the tied case, one plausible block ordering is:

- embeddings block
- unembeddings/output block

The model aligner then predicts $P_{model}$ from these model blocks.

## Why This Simplifies Training Code

This refactor moves complexity away from scope recovery and into the base representation.

Benefits:

- scope builders no longer need to mine `model` for query/key, value/output, or MLP blocks
- canonicalization code can operate on structured blocks directly
- the relationship between a symmetry and its conditioning evidence becomes explicit
- batched training becomes more natural because each sample exposes the same family of block tensors

In particular, current logic such as regrouping `model` rows by equivalence prefixes becomes unnecessary or dramatically smaller.

## Tradeoffs

This is not a free simplification. It changes where the complexity lives.

Costs:

- serialization becomes more opinionated
- deserialization must understand block layouts rather than plain flat row streams
- transformation code must support broadcast left actions and symmetry-specific right actions
- save/load formats may need migration or compatibility shims

So the refactor is best justified if training-time use of canonicalization is a central goal.

## Recommended Direction

1. Treat non-model symmetries as explicit affine block tensors.
2. Reserve the final row of each block for bias or zero padding.
3. Define model-side action through the augmented matrix $\tilde P_{model}$.
4. Keep the right action symmetry-specific: $P_{sym}$.
5. Predict each symmetry matrix from the structured blocks directly, rather than reconstructing conditioning blocks from `model`.

## Questions/Answers

- Exact block layouts for each symmetry family (`qk`, `ov`, `mlp`, decoder-hidden, model)
    - `qk`: blocks are `[query, key]`
    - `ov`: blocks are `[value, output]`
    - `mlp`: blocks are `[intermediate, output]`
- Whether tied embeddings and unembeddings should live in one symmetry or two linked blocks
    - If they are tied then they are the same symmetry. If they are untied then there is a separate symmetry for the MLM head that needs to be canonicalized.
- How much backward compatibility is needed for current serialized checkpoints
    - No backward compatibility is needed.
- Whether `Symmeters` should natively store block tensors, or whether a new structured container should be introduced alongside it
    - Symmeters should be updated to natively store block tensors.