import math

import torch
import torch.nn as nn

from lib.canonical_scope import BatchedCanonicalizationScope, CanonicalizationScope
from lib.utils import sinkhorn
from lib.serial_params import Symmeters


class DimensionAligner(torch.nn.Module):
    
    def __init__(self, known_size, unknown_size, sinkhorn_iters=20):
        super().__init__()
        self.known_size = known_size
        self.unknown_size = unknown_size
        self.sinkhorn_iters = sinkhorn_iters
        
        self.W_q = nn.Parameter(torch.empty(known_size, unknown_size))
        self.W_k = nn.Parameter(torch.empty(known_size, unknown_size))
        nn.init.normal_(self.W_q, std=1.0 / math.sqrt(known_size))
        nn.init.normal_(self.W_k, std=1.0 / math.sqrt(known_size))
        
    
    def forward(self, W_x, tau=1.0): # W_x shape = (batch_size, known_size, unknown_size)
        W_xt = W_x.permute(0, 2, 1) # (batch_size, unknown_size, known_size)
        Q = W_xt @ self.W_q  # (batch, unknown_size, unknown_size)
        K = W_xt @ self.W_k  # (batch, unknown_size, unknown_size)
        attention_logits = (Q @ K.permute(0, 2, 1)) / math.sqrt(self.unknown_size)

        # Sinkhorn normalization -> soft permutation matrix P
        P = sinkhorn(attention_logits / tau, n_iters=self.sinkhorn_iters)
        return P.transpose(-1, -2)


class Canonicalizer(torch.nn.Module):
    
    def __init__(self, symmeters: Symmeters, sinkhorn_iters=20):
        super().__init__()
        self.symmeters = symmeters
        self.d_model = symmeters["model"].shape[1]
        self.dimension_aligners = nn.ModuleDict()
        self.vocab_size = symmeters["vocab"].shape[1]
        self.dimension_aligners["model"] = DimensionAligner(self.vocab_size, self.d_model, sinkhorn_iters=sinkhorn_iters)
        
        for symmetry, parameters in symmeters.items():
            if symmetry in {"model", "vocab"} or parameters.vectors is None:
                continue
            prefixes = symmeters.get_equivalence_class(symmetry)
            if not prefixes:
                continue
            self.dimension_aligners[symmetry] = DimensionAligner(
                self.d_model * len(prefixes),
                parameters.shape[1],
                sinkhorn_iters=sinkhorn_iters,
            )
        
        
    def _canonicalize_scope(self, scope: CanonicalizationScope, tau=1.0):
        canonicalized = scope.symmeters.clone()
        model_params = scope.symmeters["model"]
        model_vectors = model_params.vectors
        model_matrix = self.dimension_aligners["model"](
            scope.model_conditioning_vectors.unsqueeze(0),
            tau=tau,
        ).squeeze(0)
        canonicalized_model_vectors = torch.index_copy(
            model_vectors,
            0,
            scope.model_active_indices,
            scope.model_active_vectors @ model_matrix,
        )
        canonicalized["model"] = model_params.with_vectors(canonicalized_model_vectors)

        for symmetry in scope.active_symmetry_names:
            if symmetry not in self.dimension_aligners:
                raise ValueError(f"Canonicalizer has no aligner for active symmetry {symmetry}.")

            conditioning_block = scope.model_conditioning_block(
                symmetry,
                model_vectors=canonicalized["model"].vectors,
            )
            symmetry_matrix = self.dimension_aligners[symmetry](
                conditioning_block.unsqueeze(0),
                tau=tau,
            ).squeeze(0)
            canonicalized.apply_square_matrix(symmetry_matrix, symmetry)

        return canonicalized

    def _canonicalize_batched_scope(self, scope: BatchedCanonicalizationScope, tau=1.0):
        canonicalized = [single_scope.symmeters.clone() for single_scope in scope.scopes]
        model_matrices = self.dimension_aligners["model"](
            scope.model_conditioning_vectors,
            tau=tau,
        )

        for idx, (single_scope, model_matrix) in enumerate(zip(scope.scopes, model_matrices)):
            model_params = single_scope.symmeters["model"]
            model_vectors = model_params.vectors
            canonicalized_model_vectors = torch.index_copy(
                model_vectors,
                0,
                single_scope.model_active_indices,
                single_scope.model_active_vectors @ model_matrix,
            )
            canonicalized[idx]["model"] = model_params.with_vectors(canonicalized_model_vectors)

        for symmetry in scope.active_symmetry_names:
            if symmetry not in self.dimension_aligners:
                raise ValueError(f"Canonicalizer has no aligner for active symmetry {symmetry}.")

            conditioning_block = scope.model_conditioning_block(
                symmetry,
                model_vectors=[item["model"].vectors for item in canonicalized],
            )
            symmetry_matrices = self.dimension_aligners[symmetry](conditioning_block, tau=tau)
            for item, symmetry_matrix in zip(canonicalized, symmetry_matrices):
                item.apply_square_matrix(symmetry_matrix, symmetry)

        return canonicalized

    def forward(self, scope: CanonicalizationScope | BatchedCanonicalizationScope, tau=1.0):
        """Canonicalize one scope or a batch of compatible scopes."""

        if isinstance(scope, BatchedCanonicalizationScope):
            return self._canonicalize_batched_scope(scope, tau=tau)
        if isinstance(scope, CanonicalizationScope):
            return self._canonicalize_scope(scope, tau=tau)
        raise TypeError(
            "Canonicalizer.forward expects CanonicalizationScope or BatchedCanonicalizationScope."
        )
        