# Autoencoder Loss For Augmented Residual-Space Vectors

This note documents a loss for training an autoencoder that compresses augmented vectors of the form

$$
x = \begin{bmatrix} v \\ b \end{bmatrix} \in \mathbb{R}^{769},
$$

where:

- $v \in \mathbb{R}^{768}$ is the residual-space part
- $b \in \mathbb{R}$ is the affine bias slot

The goal is not just reconstruction. The goal is for the 768-dimensional latent to behave like the transformer residual space under permutations of residual coordinates.

## Objects

Let:

- $E : \mathbb{R}^{769} \to \mathbb{R}^{768}$ be the encoder
- $D : \mathbb{R}^{768} \to \mathbb{R}^{769}$ be the decoder
- $P \in \mathbb{R}^{768 \times 768}$ be a permutation matrix acting on residual coordinates
- 
$$
G_P = \begin{bmatrix}
P & 0 \\
0 & 1
\end{bmatrix}
$$

be the corresponding action on the 769-dimensional augmented space

The correct symmetry is not a pure 768-dimensional permutation on the original input. It is $G_P$, which permutes the residual coordinates and leaves the affine slot fixed.

## Recommended Loss

Use

$$
\mathcal{L}
=
\lambda_{\mathrm{rec}} \, \mathcal{L}_{\mathrm{rec}}
+
\lambda_{\mathrm{enc}} \, \mathcal{L}_{\mathrm{enc-eq}}
+
\lambda_{\mathrm{dec}} \, \mathcal{L}_{\mathrm{dec-eq}}
+
\lambda_{\mathrm{anchor}} \, \mathcal{L}_{\mathrm{anchor}}.
$$

### 1. Reconstruction

$$
\mathcal{L}_{\mathrm{rec}} = \|D(E(x)) - x\|_2^2
$$

This makes the autoencoder reconstruct the augmented vector.

In practice, it is better to balance the residual coordinates and the scalar bias slot explicitly:

$$
\mathcal{L}_{\mathrm{rec}}
=
\frac{1}{768}\|\hat v - v\|_2^2 + \gamma (\hat b - b)^2,
$$

where $D(E(x)) = [\hat v; \hat b]$ and $\gamma \approx 1$ is a reasonable starting point.

Without this normalization, the model may ignore the single bias coordinate because it is outnumbered by the 768 residual coordinates.

### 2. Encoder Equivariance

$$
\mathcal{L}_{\mathrm{enc-eq}} = \|E(G_P x) - P E(x)\|_2^2
$$

This says: if the augmented input is transformed by the correct residual-space permutation, then the latent should transform by the corresponding residual permutation as well.

This is the key term that pushes the latent to behave like residual space rather than like an arbitrary learned bottleneck.

### 3. Decoder Equivariance

Let $z = E(x)$. Then

$$
\mathcal{L}_{\mathrm{dec-eq}} = \|D(P z) - G_P D(z)\|_2^2.
$$

This says: if the latent is permuted as residual space, decoding should match the correctly transformed augmented vector.

This makes the decoder commute with the intended symmetry.

### 4. Residual-Basis Anchor

$$
\mathcal{L}_{\mathrm{anchor}} = \|E([v;0]) - v\|_2^2
$$

This term anchors the latent basis to the residual basis on zero-bias inputs.

It matters because autoencoders are otherwise only identified up to an arbitrary invertible latent change of basis. Without an anchor, reconstruction and equivariance can still allow a latent basis that is not aligned with residual coordinates.

## Minimal Practical Version

If a lighter loss is needed, use:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\alpha \, \mathcal{L}_{\mathrm{enc-eq}}
+
\beta \, \mathcal{L}_{\mathrm{anchor}}.
$$

This is the smallest version that still aims at a residual-aligned latent rather than mere compression.

## Recommended Starting Weights

For the full loss, a reasonable starting point is:

- $\lambda_{\mathrm{rec}} = 1$
- $\lambda_{\mathrm{enc}} = 1$
- $\lambda_{\mathrm{dec}} = 1$
- $\lambda_{\mathrm{anchor}} = 0.25$

For the lighter version:

- $\alpha = 1$
- $\beta = 0.25$

These are starting values, not principled constants. They should be tuned against both reconstruction quality and permutation consistency.

## Why Reconstruction Alone Is Not Enough

A plain autoencoder can reconstruct well while still learning a latent basis unrelated to residual coordinates. If $(E, D)$ works, then so does $(A E, D A^{-1})$ for any invertible latent transform $A$.

That means low reconstruction loss alone does not justify treating the latent coordinates as residual coordinates.

The equivariance and anchor terms are there to remove that ambiguity.

## What To Validate

After training, validate three things separately:

1. Reconstruction quality: $D(E(x)) \approx x$
2. Encoder consistency: $E(G_P x) \approx P E(x)$
3. Decoder consistency: $D(P E(x)) \approx G_P x$

If reconstruction is good but the equivariance checks fail, the latent is not behaving like residual space even if the autoencoder compresses well.