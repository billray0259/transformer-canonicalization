from dataclasses import dataclass
from typing import Sequence

import torch

from lib.serial_params import Symmeters


@dataclass(frozen=True)
class CanonicalizationScope:
    """Explicit description of a scoped canonicalization task."""

    symmeters: Symmeters
    model_conditioning_indices: torch.Tensor
    model_active_indices: torch.Tensor
    active_symmetry_names: tuple[str, ...] = ()

    def __post_init__(self):
        if "model" not in self.symmeters:
            raise ValueError("CanonicalizationScope requires a model symmetry.")

        model_vectors = self.symmeters["model"].vectors
        if model_vectors is None:
            raise ValueError("CanonicalizationScope requires non-empty model vectors.")

        conditioning_indices = torch.as_tensor(
            self.model_conditioning_indices,
            device=model_vectors.device,
            dtype=torch.long,
        )
        active_indices = torch.as_tensor(
            self.model_active_indices,
            device=model_vectors.device,
            dtype=torch.long,
        )
        if conditioning_indices.ndim != 1 or active_indices.ndim != 1:
            raise ValueError("CanonicalizationScope indices must be one-dimensional.")
        if conditioning_indices.numel() == 0:
            raise ValueError("CanonicalizationScope requires at least one conditioning row.")
        if active_indices.numel() == 0:
            raise ValueError("CanonicalizationScope requires at least one active row.")

        row_count = model_vectors.shape[0]
        if torch.any((conditioning_indices < 0) | (conditioning_indices >= row_count)):
            raise ValueError("CanonicalizationScope conditioning indices are out of bounds.")
        if torch.any((active_indices < 0) | (active_indices >= row_count)):
            raise ValueError("CanonicalizationScope active indices are out of bounds.")

        active_symmetry_names = tuple(self.active_symmetry_names)
        invalid_symmetry_names = [
            symmetry_name
            for symmetry_name in active_symmetry_names
            if symmetry_name in {"model", "vocab"}
        ]
        if invalid_symmetry_names:
            raise ValueError(
                f"CanonicalizationScope active symmetries must exclude model/vocab, got {invalid_symmetry_names}."
            )

        missing_symmetry_names = [
            symmetry_name
            for symmetry_name in active_symmetry_names
            if symmetry_name not in self.symmeters
        ]
        if missing_symmetry_names:
            raise ValueError(
                f"CanonicalizationScope references missing symmetries {missing_symmetry_names}."
            )

        object.__setattr__(self, "model_conditioning_indices", conditioning_indices)
        object.__setattr__(self, "model_active_indices", active_indices)
        object.__setattr__(self, "active_symmetry_names", active_symmetry_names)

    @property
    def model_conditioning_vectors(self):
        return self.symmeters["model"].vectors.index_select(0, self.model_conditioning_indices)

    @property
    def model_active_vectors(self):
        return self.symmeters["model"].vectors.index_select(0, self.model_active_indices)

    def model_conditioning_block(self, symmetry_name, model_vectors=None):
        if symmetry_name not in self.active_symmetry_names:
            raise ValueError(f"Symmetry {symmetry_name} is not active in this scope.")

        if symmetry_name not in self.symmeters:
            raise ValueError(f"Symmetry {symmetry_name} is missing from the scoped symmeters.")

        model_vectors = self.symmeters["model"].vectors if model_vectors is None else model_vectors
        row_width = self.symmeters[symmetry_name].shape[1]
        prefixes = self.symmeters.get_equivalence_class(symmetry_name)
        if not prefixes:
            raise ValueError(f"Symmetry {symmetry_name} has no equivalence class prefixes.")

        index_blocks = []
        for prefix in prefixes:
            indices = self.symmeters._equivalence_row_index_tensor(
                "model",
                prefix,
                model_vectors.device,
            )
            if indices.numel() == 0:
                raise ValueError(
                    f"Symmetry {symmetry_name} expected model rows for prefix {prefix}, but none were found."
                )
            if indices.numel() != row_width:
                raise ValueError(
                    f"Symmetry {symmetry_name} expected {row_width} model rows for prefix {prefix}, got {indices.numel()}."
                )
            index_blocks.append(indices)

        index_blocks.sort(key=lambda indices: int(indices[0]))
        return torch.cat(
            [model_vectors.index_select(0, indices).transpose(0, 1) for indices in index_blocks],
            dim=0,
        )


@dataclass(frozen=True)
class BatchedCanonicalizationScope:
    """Batch of compatible canonicalization scopes for batched aligner evaluation."""

    scopes: tuple[CanonicalizationScope, ...]

    def __post_init__(self):
        scopes = tuple(self.scopes)
        if not scopes:
            raise ValueError("BatchedCanonicalizationScope requires at least one scope.")

        reference_scope = scopes[0]
        reference_conditioning_shape = tuple(reference_scope.model_conditioning_vectors.shape)
        reference_active_shape = tuple(reference_scope.model_active_vectors.shape)
        reference_active_symmetries = reference_scope.active_symmetry_names

        for scope in scopes[1:]:
            if scope.active_symmetry_names != reference_active_symmetries:
                raise ValueError(
                    "BatchedCanonicalizationScope requires identical active symmetry sets across scopes."
                )
            if tuple(scope.model_conditioning_vectors.shape) != reference_conditioning_shape:
                raise ValueError(
                    "BatchedCanonicalizationScope requires model conditioning vectors with identical shapes."
                )
            if tuple(scope.model_active_vectors.shape) != reference_active_shape:
                raise ValueError(
                    "BatchedCanonicalizationScope requires model active vectors with identical shapes."
                )
            for symmetry_name in reference_active_symmetries:
                if tuple(scope.model_conditioning_block(symmetry_name).shape) != tuple(reference_scope.model_conditioning_block(symmetry_name).shape):
                    raise ValueError(
                        f"BatchedCanonicalizationScope requires identical conditioning block shapes for symmetry {symmetry_name}."
                    )

        object.__setattr__(self, "scopes", scopes)

    @classmethod
    def from_scopes(cls, scopes: Sequence[CanonicalizationScope]):
        return cls(tuple(scopes))

    @property
    def batch_size(self):
        return len(self.scopes)

    @property
    def active_symmetry_names(self):
        return self.scopes[0].active_symmetry_names

    @property
    def model_conditioning_vectors(self):
        return torch.stack([scope.model_conditioning_vectors for scope in self.scopes])

    @property
    def model_active_vectors(self):
        return torch.stack([scope.model_active_vectors for scope in self.scopes])

    def model_conditioning_block(self, symmetry_name, model_vectors=None):
        if model_vectors is None:
            return torch.stack([
                scope.model_conditioning_block(symmetry_name)
                for scope in self.scopes
            ])

        if isinstance(model_vectors, torch.Tensor):
            if model_vectors.ndim != 3 or model_vectors.shape[0] != self.batch_size:
                raise ValueError(
                    "BatchedCanonicalizationScope model_vectors tensor must have shape (batch, rows, d_model)."
                )
            return torch.stack([
                scope.model_conditioning_block(symmetry_name, model_vectors=model_vectors[idx])
                for idx, scope in enumerate(self.scopes)
            ])

        if len(model_vectors) != self.batch_size:
            raise ValueError("BatchedCanonicalizationScope model_vectors sequence must match batch size.")

        return torch.stack([
            scope.model_conditioning_block(symmetry_name, model_vectors=vectors)
            for scope, vectors in zip(self.scopes, model_vectors)
        ])