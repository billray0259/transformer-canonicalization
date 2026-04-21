import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.serial_params import ParameterComponent, Symmeters
from lib.utils import sinkhorn


def _flatten_component_for_axis(
    component: ParameterComponent,
    axis_name: str,
    bank_axis_name: str | None = None,
) -> torch.Tensor | None:
    axis_indices = component.axis_indices(axis_name)
    if len(axis_indices) != 1:
        return None
    axis_index = axis_indices[0]

    if bank_axis_name is not None:
        bank_axis_indices = component.axis_indices(bank_axis_name)
        if len(bank_axis_indices) != 1:
            return None
        bank_axis_index = bank_axis_indices[0]
        tensor = component.tensor.movedim((bank_axis_index, axis_index), (0, -1))
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])

    tensor = component.tensor.movedim(axis_index, -1)
    return tensor.reshape(-1, tensor.shape[-1])


def _module_key(symmetry_name: str) -> str:
    return symmetry_name.replace(".", "__dot__")


# class DimensionAligner(nn.Module):
#     def __init__(self, known_size: int, hidden_size: int, unknown_size: int, sinkhorn_iters: int = 20):
#         super().__init__()
#         self.known_size = known_size
#         self.hidden_size = hidden_size
#         self.unknown_size = unknown_size
#         self.sinkhorn_iters = sinkhorn_iters

#         self.descriptor = nn.Linear(known_size, hidden_size)
#         self.gelu = nn.GELU()
#         self.W_q = nn.Parameter(torch.empty(hidden_size, unknown_size))
#         self.W_k = nn.Parameter(torch.empty(hidden_size, unknown_size))
#         nn.init.normal_(self.W_q, std=1.0 / math.sqrt(hidden_size))
#         nn.init.normal_(self.W_k, std=1.0 / math.sqrt(hidden_size))

#     def forward(self, evidence: torch.Tensor, tau: float = 1.0):
#         evidence_t = evidence.permute(0, 2, 1)
#         evidence_t = self.descriptor(evidence_t)
#         evidence_t = self.gelu(evidence_t)
#         queries = evidence_t @ self.W_q
#         keys = evidence_t @ self.W_k
#         logits = (queries @ keys.permute(0, 2, 1)) / math.sqrt(self.hidden_size)
#         transport = sinkhorn(logits / tau, n_iters=self.sinkhorn_iters)
#         return transport.transpose(-1, -2)

class PermutationAligner(nn.Module):
    def __init__(self, known_size, unknown_size, sinkhorn_iters=20):
        super().__init__()
        # Learned prototypes: one per canonical position
        self.prototypes = nn.Parameter(torch.randn(known_size, unknown_size) / math.sqrt(known_size))
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, evidence, tau=1.0):
        # evidence: (batch, known_size, unknown_size)
        rows = evidence.permute(0, 2, 1)  # (batch, unknown_size, known_size)
        logits = (rows @ self.prototypes) / tau
        transport = sinkhorn(logits, n_iters=self.sinkhorn_iters)
        
        return transport, logits
    
class RotationAligner(nn.Module):
    def __init__(self, known_size, unknown_size):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(unknown_size, known_size) / math.sqrt(known_size))

    def forward(self, evidence):
        # evidence: (batch, known_size, unknown_size)
        # M = prototypes^T @ evidence, batched
        M = (self.prototypes)[None] @ evidence  # (batch, unknown, unknown)
        U, S, Vh = torch.linalg.svd(M)
        R = U @ Vh
        return R

class HeadDescriptorEncoder(nn.Module):
    def __init__(self, component_specs, axis_name: str, descriptor_size: int = 64):
        super().__init__()
        self.axis_name = axis_name
        self.projectors = nn.ModuleDict()
        self.projector_specs: list[tuple[str, str, str]] = []
        projected_width = 0
        for symmetry_name, component_name, component in component_specs:
            flattened = _flatten_component_for_axis(component, axis_name)
            if flattened is None:
                continue
            feature_dim = flattened.shape[0]
            projector_key = f"{_module_key(symmetry_name)}::{component_name.replace('.', '__dot__')}"
            self.projectors[projector_key] = nn.Linear(feature_dim, descriptor_size // 2, bias=False)
            self.projector_specs.append((projector_key, symmetry_name, component_name))
            projected_width += descriptor_size // 2
        self.mlp = nn.Sequential(
            nn.Linear(projected_width, descriptor_size),
            nn.GELU(),
            nn.Linear(descriptor_size, descriptor_size),
        )

    def forward(self, symmeters: Symmeters) -> torch.Tensor:
        projected = []
        for projector_key, symmetry_name, component_name in self.projector_specs:
            component = symmeters.component(symmetry_name, component_name)
            flattened = _flatten_component_for_axis(component, self.axis_name)
            projected.append(self.projectors[projector_key](flattened.T))
        return self.mlp(torch.cat(projected, dim=-1))


class HeadAligner(nn.Module):
    def __init__(self, component_specs, axis_name: str, descriptor_size: int = 64, sinkhorn_iters: int = 20):
        super().__init__()
        self.encoder = HeadDescriptorEncoder(component_specs, axis_name=axis_name, descriptor_size=descriptor_size)
        self.sinkhorn_iters = sinkhorn_iters
        self.scale = math.sqrt(descriptor_size)

    def forward(self, symmeters: Symmeters, tau: float = 1.0):
        descriptors = self.encoder(symmeters)
        logits = (descriptors @ descriptors.T) / self.scale
        transport = sinkhorn(logits.unsqueeze(0) / tau, n_iters=self.sinkhorn_iters)[0].squeeze(0).T
        return transport


class CascadingTemplateCanonicalizer(nn.Module):
    def __init__(self, order: Sequence[str], templates: dict[str, torch.Tensor]):
        super().__init__()
        self.order = tuple(order)
        missing = [symmetry_name for symmetry_name in self.order if symmetry_name not in templates]
        if missing:
            raise ValueError(f"Missing templates for symmetries {missing}.")

        self._buffer_names: dict[str, str | dict[str, str]] = {}
        for symmetry_name in self.order:
            template = templates[symmetry_name]
            if isinstance(template, dict):
                self._buffer_names[symmetry_name] = {}
                for kind, tensor in template.items():
                    buffer_name = f"template__{_module_key(symmetry_name)}__{kind}"
                    self.register_buffer(buffer_name, tensor.detach().clone())
                    self._buffer_names[symmetry_name][kind] = buffer_name
                continue

            buffer_name = f"template__{_module_key(symmetry_name)}"
            self.register_buffer(buffer_name, template.detach().clone())
            self._buffer_names[symmetry_name] = buffer_name

    def template(self, symmetry_name: str) -> torch.Tensor:
        if symmetry_name not in self._buffer_names:
            raise KeyError(f"Unknown symmetry {symmetry_name}.")
        buffer_names = self._buffer_names[symmetry_name]
        if isinstance(buffer_names, dict):
            return {kind: getattr(self, buffer_name) for kind, buffer_name in buffer_names.items()}
        return getattr(self, buffer_names)

    @staticmethod
    def procrustes_align(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        matrix = source.transpose(-1, -2) @ target
        u, _, vh = torch.linalg.svd(matrix)
        sign, _ = torch.linalg.slogdet(u @ vh)
        u[..., :, -1] *= sign.unsqueeze(-1)
        return u @ vh

    @staticmethod
    def permutation_align(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        from scipy.optimize import linear_sum_assignment

        scores = (source.T @ target).detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(scores, maximize=True)
        return torch.eye(source.shape[1], device=source.device, dtype=source.dtype)[col_ind]

    @staticmethod
    def head_evidence(symmeters: Symmeters, symmetry_name: str) -> dict[str, torch.Tensor]:
        prefix = symmetry_name.rsplit(".", 1)[0]
        return {
            kind: Canonicalizer._evidence_tensor(symmeters, f"{prefix}.{kind}").detach().float()
            for kind in ("qk", "ov")
        }

    @staticmethod
    def apply_head_permutation(evidence: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("h...,hj->j...", evidence, matrix)

    @classmethod
    def head_permutation_align(cls, target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> torch.Tensor:
        from scipy.optimize import linear_sum_assignment

        num_heads = source["qk"].shape[0]
        total_cost = torch.zeros((num_heads, num_heads), device=source["qk"].device, dtype=source["qk"].dtype)
        for kind in ("qk", "ov"):
            t, s = target[kind], source[kind]
            K, D = t.shape[-2], t.shape[-1]
            M = torch.einsum("jkd,ike->ijde", s, t)
            S = torch.linalg.svdvals(M.flatten(0, 1)).unflatten(0, (num_heads, num_heads))
            total_cost += (t.pow(2).sum((-2, -1))[:, None] + s.pow(2).sum((-2, -1))[None, :] - 2 * S.sum(-1)) / (K * D)
        _, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
        return torch.eye(num_heads, device=source["qk"].device, dtype=source["qk"].dtype)[col_ind]

    @staticmethod
    def head_descriptor(symmeters: Symmeters, symmetry_name: str) -> torch.Tensor:
        pieces = []
        for _, _, component in symmeters.components_with_axis(symmetry_name):
            flattened = _flatten_component_for_axis(component, symmetry_name)
            if flattened is not None:
                pieces.append(flattened.float())
        if not pieces:
            raise ValueError(f"No head evidence for {symmetry_name}.")
        return torch.cat(pieces, dim=0)

    def infer_transform(self, symmeters: Symmeters, symmetry_name: str) -> torch.Tensor:
        template = self.template(symmetry_name)
        if symmetry_name.endswith(".head"):
            evidence = {
                kind: tensor.to(device=template[kind].device, dtype=template[kind].dtype)
                for kind, tensor in self.head_evidence(symmeters, symmetry_name).items()
            }
            return self.head_permutation_align(template, evidence)

        evidence = Canonicalizer._evidence_tensor(symmeters, symmetry_name).detach().to(device=template.device, dtype=template.dtype)
        return self.procrustes_align(template, evidence)

    @staticmethod
    def apply_symmetry_transform(symmeters: Symmeters, symmetry_name: str, matrix: torch.Tensor):
        if symmetry_name.endswith(".head"):
            symmeters.apply_head_transport(symmetry_name, matrix)
        else:
            symmeters.apply_transform(symmetry_name, matrix)
        return symmeters

    def canonicalize_with_transforms(self, symmeters: Symmeters) -> tuple[Symmeters, dict[str, torch.Tensor]]:
        if not isinstance(symmeters, Symmeters):
            raise TypeError("CascadingTemplateCanonicalizer.forward expects Symmeters.")

        canonicalized = symmeters.clone()
        inferred: dict[str, torch.Tensor] = {}
        for symmetry_name in self.order:
            matrix = self.infer_transform(canonicalized, symmetry_name)
            self.apply_symmetry_transform(canonicalized, symmetry_name, matrix)
            inferred[symmetry_name] = matrix
        return canonicalized, inferred

    def forward(self, symmeters: Symmeters) -> Symmeters:
        canonicalized, _ = self.canonicalize_with_transforms(symmeters)
        return canonicalized

    def save(self, path: str):
        torch.save(
            {
                "order": self.order,
                "templates": {
                    symmetry_name: {
                        kind: tensor.detach().cpu()
                        for kind, tensor in self.template(symmetry_name).items()
                    } if isinstance(self.template(symmetry_name), dict) else self.template(symmetry_name).detach().cpu()
                    for symmetry_name in self.order
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None):
        data = torch.load(path, map_location=map_location)
        return cls(data["order"], data["templates"])


class Canonicalizer(nn.Module):
    def __init__(self, symmeters: Symmeters, sinkhorn_iters: int = 20, head_descriptor_size: int = 64):
        super().__init__()
        self.dimension_aligners = nn.ModuleDict()
        self.head_aligners = nn.ModuleDict()

        for symmetry_name in symmeters.ordered_transform_names():
            if symmetry_name.endswith(".head"):
                component_specs = symmeters.components_with_axis(symmetry_name)
                if component_specs:
                    self.head_aligners[_module_key(symmetry_name)] = HeadAligner(
                        component_specs,
                        axis_name=symmetry_name,
                        descriptor_size=head_descriptor_size,
                        sinkhorn_iters=sinkhorn_iters,
                    )
                continue

            try:
                evidence = self._evidence_tensor(symmeters, symmetry_name)
            except ValueError:
                continue
            self.dimension_aligners[_module_key(symmetry_name)] = PermutationAligner(
                known_size=evidence.shape[-2],
                unknown_size=evidence.shape[-1],
                sinkhorn_iters=sinkhorn_iters,
            )

    @staticmethod
    def _evidence_tensor(symmeters: Symmeters, symmetry_name: str) -> torch.Tensor:
        bank_axis_name = symmeters.transform_bank_axis(symmetry_name)

        if bank_axis_name is None:
            evidence_components = [
                flattened
                for component in symmeters.owned_components(symmetry_name).values()
                for flattened in [_flatten_component_for_axis(component, symmetry_name)]
                if flattened is not None
            ]
            if not evidence_components:
                raise ValueError(f"No evidence components found for symmetry {symmetry_name}.")
            return torch.cat(evidence_components, dim=0)

        evidence_components = [
            flattened
            for component in symmeters.owned_components(symmetry_name).values()
            for flattened in [_flatten_component_for_axis(component, symmetry_name, bank_axis_name=bank_axis_name)]
            if flattened is not None
        ]
        if not evidence_components:
            raise ValueError(f"No evidence components found for symmetry {symmetry_name}.")
        return torch.cat(evidence_components, dim=1)

    @staticmethod
    def _normalize_active_symmetry_names(
        symmeters: Symmeters,
        active_symmetry_names: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        if "model" not in symmeters:
            raise ValueError("Canonicalizer requires a model symmetry.")

        normalized = tuple(active_symmetry_names) if active_symmetry_names is not None else tuple(
            symmetry_name
            for symmetry_name in symmeters.ordered_transform_names()
            if symmetry_name != "model"
        )
        missing = [symmetry_name for symmetry_name in normalized if symmetry_name not in symmeters]
        if missing:
            raise ValueError(f"Canonicalizer received missing active symmetries {missing}.")
        return normalized

    def _canonicalize_single(
        self,
        symmeters: Symmeters,
        active_symmetry_names: Sequence[str] | None = None,
        tau: float = 1.0,
    ):
        normalized_active_symmetry_names = self._normalize_active_symmetry_names(
            symmeters,
            active_symmetry_names=active_symmetry_names,
        )
        canonicalized = symmeters.clone()
        for symmetry_name in ("model", *normalized_active_symmetry_names):
            module_key = _module_key(symmetry_name)
            if module_key in self.dimension_aligners:
                evidence = self._evidence_tensor(canonicalized, symmetry_name)
                if evidence.ndim == 2:
                    matrix = self.dimension_aligners[module_key](evidence.unsqueeze(0), tau=tau).squeeze(0)
                else:
                    matrix = self.dimension_aligners[module_key](evidence, tau=tau)
                canonicalized.apply_transform(symmetry_name, matrix)
                continue

            if module_key in self.head_aligners:
                matrix = self.head_aligners[module_key](canonicalized, tau=tau)
                canonicalized.apply_head_transport(symmetry_name, matrix)
        return canonicalized

    def forward(
        self,
        symmeters: Symmeters,
        tau: float = 1.0,
        active_symmetry_names: Sequence[str] | None = None,
    ):
        if not isinstance(symmeters, Symmeters):
            raise TypeError("Canonicalizer.forward expects Symmeters.")
        return self._canonicalize_single(
            symmeters,
            active_symmetry_names=active_symmetry_names,
            tau=tau,
        )
        