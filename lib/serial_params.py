from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

import torch


_LAYER_SYMMETRY_RE = re.compile(r"^L(?P<layer>\d+)\.(?P<kind>qk|ov|mlp|head)$")
_BANKED_SYMMETRY_RE = re.compile(r"^L(?P<layer>\d+)\.(?P<kind>qk|ov)$")


def _bank_axis_name(symmetry_name: str) -> str | None:
    """Map a banked symmetry name to its associated head-axis name.

    Args:
        symmetry_name: Symmetry identifier to inspect, such as ``L0.qk`` or ``L3.ov``.

    Returns:
        The corresponding per-layer head symmetry name when ``symmetry_name`` is a
        banked ``qk`` or ``ov`` symmetry, otherwise ``None``.

    Process:
        Match the input name against the banked-symmetry pattern and, on success,
        rewrite the suffix to ``head`` for the same layer.
    """
    match = _BANKED_SYMMETRY_RE.match(symmetry_name)
    if match is None:
        return None
    return f"L{match.group('layer')}.head"


@dataclass(frozen=True)
class ParameterComponent:
    tensor: torch.Tensor
    axes: tuple[str, ...]
    kind: str
    layout: str
    parameter_keys: tuple[str, ...]

    def __post_init__(self):
        """Validate basic structural invariants after dataclass initialization.

        Args:
            None.

        Returns:
            None.

        Process:
            Confirm that ``tensor`` is a ``torch.Tensor`` and that the number of
            declared axis names matches the tensor rank. Raise ``ValueError`` when
            either invariant is violated.
        """
        if not isinstance(self.tensor, torch.Tensor):
            raise ValueError("ParameterComponent tensor must be a torch.Tensor.")
        if self.tensor.ndim != len(self.axes):
            raise ValueError("ParameterComponent axes must have one entry per tensor dimension.")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | ParameterComponent):
        """Build a component from a serialized payload or return an existing one.

        Args:
            payload: Either an existing ``ParameterComponent`` or a mapping with at
                least ``tensor`` and ``axes`` entries plus optional metadata.

        Returns:
            A normalized ``ParameterComponent`` instance.

        Process:
            Pass through existing instances unchanged, validate mapping payloads,
            normalize ``parameter_keys`` to a tuple, then construct a new component
            while filling missing metadata with default values.
        """
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("ParameterComponent payloads must be mappings.")
        if "tensor" not in payload or "axes" not in payload:
            raise ValueError("ParameterComponent payloads must include tensor and axes.")

        parameter_keys = payload.get("parameter_keys", ())
        if isinstance(parameter_keys, str):
            normalized_parameter_keys = (parameter_keys,)
        else:
            normalized_parameter_keys = tuple(parameter_keys)

        return cls(
            tensor=payload["tensor"],
            axes=tuple(payload["axes"]),
            kind=payload.get("kind", "unknown"),
            layout=payload.get("layout", "identity"),
            parameter_keys=normalized_parameter_keys,
        )

    def with_tensor(self, tensor: torch.Tensor):
        """Return a copy of the component with a different tensor value.

        Args:
            tensor: Replacement tensor to attach to the component metadata.

        Returns:
            A new ``ParameterComponent`` carrying the provided tensor and the same
            axis, kind, layout, and parameter-key metadata.

        Process:
            Reconstruct the dataclass with the new tensor while preserving all other
            fields from the current instance.
        """
        return ParameterComponent(
            tensor=tensor,
            axes=self.axes,
            kind=self.kind,
            layout=self.layout,
            parameter_keys=self.parameter_keys,
        )

    def axis_indices(self, axis_name: str) -> list[int]:
        """Return every tensor dimension whose axis label matches ``axis_name``.

        Args:
            axis_name: Axis label to search for in ``self.axes``.

        Returns:
            A list of integer dimension indices where the axis label matches.

        Process:
            Scan ``self.axes`` in order and collect the positions whose stored name
            equals the requested axis name.
        """
        return [idx for idx, name in enumerate(self.axes) if name == axis_name]

    def has_axis(self, axis_name: str) -> bool:
        """Check whether the component references a named symmetry axis.

        Args:
            axis_name: Axis label to look for.

        Returns:
            ``True`` when the axis label appears in ``self.axes`` and ``False``
            otherwise.

        Process:
            Perform a membership test against the stored axis-name tuple.
        """
        return axis_name in self.axes

    def to_payload(self) -> dict[str, Any]:
        """Serialize the component into the persisted payload format.

        Args:
            None.

        Returns:
            A dictionary containing the tensor and all metadata fields needed to
            reconstruct the component later.

        Process:
            Copy the current dataclass fields into a plain dictionary without
            modifying the underlying tensor.
        """
        return {
            "tensor": self.tensor,
            "axes": self.axes,
            "kind": self.kind,
            "layout": self.layout,
            "parameter_keys": self.parameter_keys,
        }


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Clone a tensor while preserving its gradient requirement.

    Args:
        tensor: Tensor to duplicate.

    Returns:
        A detached clone with ``requires_grad`` restored to match the input tensor.

    Process:
        Detach the tensor from any computation graph, clone its storage, and then
        reapply the original ``requires_grad`` flag.
    """
    clone = tensor.detach().clone()
    return clone.requires_grad_(tensor.requires_grad)


class Symmeters(dict):
    def __init__(
        self,
        symmetry_names: Iterable[str] = (),
        symmetry_dict: dict[str, dict[str, ParameterComponent | Mapping[str, Any]]] | None = None,
    ):
        """Initialize a symmetry-to-component mapping.

        Args:
            symmetry_names: Symmetry names to pre-create with empty component maps.
            symmetry_dict: Optional initial mapping whose component payloads are
                normalized into ``ParameterComponent`` instances.

        Returns:
            None.

        Process:
            Start from an empty dictionary, create any requested empty symmetry
            entries, then feed provided component mappings through ``__setitem__`` so
            normalization rules are applied consistently.
        """
        super().__init__()
        for symmetry_name in symmetry_names:
            super().__setitem__(symmetry_name, {})
        if symmetry_dict is not None:
            for symmetry_name, components in symmetry_dict.items():
                self[symmetry_name] = components

    @property
    def symmetry_names(self) -> list[str]:
        """Return the symmetry names currently stored in the mapping.

        Args:
            None.

        Returns:
            A list containing the dictionary keys in iteration order.

        Process:
            Materialize the current key view into a list.
        """
        return list(self.keys())

    @classmethod
    def from_symmetry_dict(cls, symmetry_dict: dict[str, dict[str, ParameterComponent | Mapping[str, Any]]]):
        """Construct a ``Symmeters`` instance from an existing symmetry mapping.

        Args:
            symmetry_dict: Mapping from symmetry names to component payloads or
                ``ParameterComponent`` instances.

        Returns:
            A populated ``Symmeters`` instance.

        Process:
            Seed the new object with the symmetry names and delegate payload
            normalization to the initializer.
        """
        return cls(symmetry_names=symmetry_dict.keys(), symmetry_dict=symmetry_dict)

    def __setitem__(self, key: str, value: dict[str, ParameterComponent | Mapping[str, Any]]):
        """Store a symmetry entry after normalizing all component payloads.

        Args:
            key: Symmetry name being assigned.
            value: Mapping from component names to ``ParameterComponent`` objects or
                payload dictionaries.

        Returns:
            None.

        Process:
            Validate that the payload is dictionary-shaped, convert every component
            payload via ``ParameterComponent.from_payload``, and then store the
            normalized mapping in the underlying dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Symmetry payloads must be dictionaries of parameter components.")
        super().__setitem__(key, {component_name: ParameterComponent.from_payload(spec) for component_name, spec in value.items()})

    def __add__(self, other):
        """Combine two symmetry collections into a new merged instance.

        Args:
            other: Another ``Symmeters`` object to merge into this one.

        Returns:
            A new ``Symmeters`` containing all symmetries and components from both
            operands, or ``NotImplemented`` for unsupported operand types.

        Process:
            Clone the left-hand mapping structure, add each symmetry from ``other``,
            and reject duplicate component names within the same symmetry.
        """
        if not isinstance(other, Symmeters):
            return NotImplemented

        combined = Symmeters.from_symmetry_dict(
            {
                symmetry_name: dict(components)
                for symmetry_name, components in self.items()
            }
        )
        for symmetry_name, components in other.items():
            combined.add_symmetry(symmetry_name)
            for component_name, component in components.items():
                if component_name in combined[symmetry_name]:
                    raise ValueError(f"Duplicate parameter component {component_name} in symmetry {symmetry_name}.")
                combined[symmetry_name][component_name] = component
        return combined

    def add_symmetry(self, symmetry_name: str):
        """Ensure that a symmetry entry exists and return the collection.

        Args:
            symmetry_name: Symmetry identifier to add when missing.

        Returns:
            ``self`` for fluent-style chaining.

        Process:
            Create an empty component dictionary for the symmetry only if the key is
            not already present.
        """
        if symmetry_name not in self:
            super().__setitem__(symmetry_name, {})
        return self

    def add_component(
        self,
        symmetry_name: str,
        component_name: str,
        tensor: torch.Tensor,
        axes: Iterable[str],
        *,
        kind: str,
        layout: str,
        parameter_keys: str | Iterable[str],
    ):
        """Create and register a new parameter component under a symmetry.

        Args:
            symmetry_name: Symmetry that owns the component.
            component_name: Name to assign to the component within that symmetry.
            tensor: Tensor payload for the component.
            axes: Axis labels aligned with the tensor dimensions.
            kind: Logical component type metadata.
            layout: Layout metadata describing how the tensor is organized.
            parameter_keys: Source parameter name or names represented by the
                component.

        Returns:
            ``self`` for fluent-style chaining.

        Process:
            Ensure the symmetry exists, reject duplicate component names, normalize
            the provided metadata through ``ParameterComponent.from_payload``, and
            store the result in the symmetry entry.
        """
        self.add_symmetry(symmetry_name)
        if component_name in self[symmetry_name]:
            raise ValueError(f"ParameterComponent {component_name} already exists in symmetry {symmetry_name}.")
        self[symmetry_name][component_name] = ParameterComponent.from_payload(
            {
                "tensor": tensor,
                "axes": axes,
                "kind": kind,
                "layout": layout,
                "parameter_keys": parameter_keys,
            }
        )
        return self

    def iter_components(self):
        """Yield every stored component together with its symmetry and name.

        Args:
            None.

        Returns:
            An iterator of ``(symmetry_name, component_name, component)`` tuples.

        Process:
            Traverse the outer symmetry mapping and each inner component mapping in
            iteration order, yielding one tuple per component.
        """
        for symmetry_name, components in self.items():
            for component_name, component in components.items():
                yield symmetry_name, component_name, component

    def owned_components(self, symmetry_name: str) -> dict[str, ParameterComponent]:
        """Return the component mapping owned by a specific symmetry.

        Args:
            symmetry_name: Symmetry whose components should be retrieved.

        Returns:
            The component dictionary for that symmetry, or an empty dictionary when
            the symmetry is absent.

        Process:
            Perform a dictionary lookup with an empty-dictionary default.
        """
        return self.get(symmetry_name, {})

    def components_with_axis(self, axis_name: str):
        """Collect all components that reference a given axis label.

        Args:
            axis_name: Axis label to search for.

        Returns:
            A list of ``(symmetry_name, component_name, component)`` tuples for every
            component whose axes include ``axis_name``.

        Process:
            Iterate over every stored component and keep only those for which
            ``component.has_axis(axis_name)`` is true.
        """
        return [
            (symmetry_name, component_name, component)
            for symmetry_name, component_name, component in self.iter_components()
            if component.has_axis(axis_name)
        ]

    def component(self, symmetry_name: str, component_name: str) -> ParameterComponent:
        """Fetch a specific component by symmetry name and component name.

        Args:
            symmetry_name: Symmetry containing the component.
            component_name: Component name within that symmetry.

        Returns:
            The matching ``ParameterComponent`` instance.

        Process:
            Index directly into the nested dictionary structure, allowing the usual
            ``KeyError`` behavior when a name is missing.
        """
        return self[symmetry_name][component_name]

    def tensor(self, symmetry_name: str, component_name: str) -> torch.Tensor:
        """Fetch the tensor payload for a named component.

        Args:
            symmetry_name: Symmetry containing the component.
            component_name: Component name within that symmetry.

        Returns:
            The tensor stored on the requested component.

        Process:
            Resolve the component from the nested mapping and return its ``tensor``
            field.
        """
        return self[symmetry_name][component_name].tensor

    def get_component(self, component_name: str, symmetry_name: str | None = None) -> ParameterComponent | None:
        """Look up a component optionally scoped to one symmetry.

        Args:
            component_name: Component name to search for.
            symmetry_name: Optional symmetry to limit the lookup to.

        Returns:
            The matching ``ParameterComponent`` when found, otherwise ``None``.

        Process:
            If a symmetry is provided, search only that inner mapping; otherwise,
            scan all symmetry entries until the first matching component name is
            found.
        """
        if symmetry_name is not None:
            return self.get(symmetry_name, {}).get(component_name)
        for components in self.values():
            if component_name in components:
                return components[component_name]
        return None

    def has_component(self, component_name: str, symmetry_name: str | None = None) -> bool:
        """Report whether a component exists in the collection.

        Args:
            component_name: Component name to search for.
            symmetry_name: Optional symmetry to restrict the search to.

        Returns:
            ``True`` when ``get_component`` finds a matching component and ``False``
            otherwise.

        Process:
            Delegate the lookup to ``get_component`` and convert the result to a
            boolean presence check.
        """
        return self.get_component(component_name, symmetry_name=symmetry_name) is not None

    def symmetry_size(self, symmetry_name: str) -> int:
        """Infer the common dimension size associated with a symmetry axis.

        Args:
            symmetry_name: Symmetry axis whose size should be determined.

        Returns:
            The unique integer dimension shared by every matching tensor axis.

        Process:
            Inspect all components carrying the requested axis, collect the
            corresponding tensor extents, and verify that at least one axis exists and
            that all observed sizes agree.
        """
        sizes = {
            component.tensor.shape[axis_index]
            for _, _, component in self.components_with_axis(symmetry_name)
            for axis_index, axis_name in enumerate(component.axes)
            if axis_name == symmetry_name
        }
        if not sizes:
            raise ValueError(f"Symmetry {symmetry_name} has no attached tensor axis.")
        if len(sizes) != 1:
            raise ValueError(f"Symmetry {symmetry_name} has inconsistent sizes: {sorted(sizes)}")
        return sizes.pop()

    def transform_bank_axis(self, symmetry_name: str) -> str | None:
        """Return the head-axis name needed for a banked symmetry transform.

        Args:
            symmetry_name: Symmetry to convert into its associated bank axis.

        Returns:
            The bank axis name when the symmetry supports banked transforms and at
            least one component carries that axis, otherwise ``None``.

        Process:
            Derive the candidate axis name via ``_bank_axis_name`` and keep it only
            if components in the collection actually reference that axis.
        """
        bank_axis_name = _bank_axis_name(symmetry_name)
        if bank_axis_name is None:
            return None
        return bank_axis_name if self.components_with_axis(bank_axis_name) else None

    @staticmethod
    def _apply_axis_transform(tensor: torch.Tensor, matrix: torch.Tensor, axis_index: int) -> torch.Tensor:
        """Apply a square transform matrix along one tensor axis.

        Args:
            tensor: Tensor to transform.
            matrix: Square transform matrix whose input and output dimensions match
                the selected axis.
            axis_index: Dimension index along which to apply the transform.

        Returns:
            A tensor with the selected axis right-multiplied by ``matrix``.

        Process:
            Move the target axis to the end, perform matrix multiplication on that
            trailing dimension, and then move the axis back to its original position.
        """
        moved = tensor.movedim(axis_index, -1)
        transformed = moved @ matrix
        return transformed.movedim(-1, axis_index)

    @staticmethod
    def _apply_banked_axis_transform(
        tensor: torch.Tensor,
        matrix: torch.Tensor,
        bank_axis_index: int,
        axis_index: int,
    ) -> torch.Tensor:
        """Apply one transform per bank along a banked tensor axis.

        Args:
            tensor: Tensor to transform.
            matrix: Banked square transforms with shape ``(banks, in_dim, out_dim)``.
            bank_axis_index: Dimension identifying which bank each slice belongs to.
            axis_index: Dimension along which each bank-specific transform is applied.

        Returns:
            A tensor transformed independently within each bank.

        Process:
            Move the bank axis to the front and the transformed axis to the end,
            apply a bank-wise einsum against the stacked matrices, and restore the
            original axis order.
        """
        moved = tensor.movedim((bank_axis_index, axis_index), (0, -1))
        transformed = torch.einsum("b...i,bij->b...j", moved, matrix)
        return transformed.movedim((0, -1), (bank_axis_index, axis_index))

    @staticmethod
    def _attention_dual_roles(symmetry_name: str) -> tuple[str, str] | None:
        if symmetry_name.endswith(".qk"):
            return ("query", "key")
        if symmetry_name.endswith(".ov"):
            return ("value", "output")
        return None

    @classmethod
    def _attention_dual_role(cls, symmetry_name: str, component_name: str, component: ParameterComponent) -> str | None:
        roles = cls._attention_dual_roles(symmetry_name)
        if roles is None:
            return None

        names = (component_name, *component.parameter_keys)
        if symmetry_name.endswith(".qk"):
            if any(".query." in name or name.startswith("query.") for name in names):
                return "query"
            if any(".key." in name or name.startswith("key.") for name in names):
                return "key"
            return None

        if any(".self.value." in name or ".value." in name or name.startswith("value.") for name in names):
            return "value"
        if any(".output.dense.weight" in name or name.startswith("output.dense.weight") for name in names):
            return "output"
        return None

    def apply_transform(self, symmetry_name: str, matrix: torch.Tensor):
        """Apply a symmetry transform to every component carrying that axis.

        Args:
            symmetry_name: Symmetry axis to transform.
            matrix: Either a square matrix for a shared transform or a rank-3 bank of
                square matrices for per-head transforms.

        Returns:
            ``self`` after updating matching component tensors in place.

        Process:
            Validate that the symmetry exists, distinguish between shared and banked
            transforms by matrix rank, cast the transform to each tensor's device and
            dtype, verify shape compatibility, and replace every affected component
            with an updated tensor.
        """
        if symmetry_name not in self and not self.components_with_axis(symmetry_name):
            raise ValueError(f"Symmetry {symmetry_name} not found.")

        if matrix.ndim not in {2, 3}:
            raise ValueError(
                f"Expected a square matrix or bank of square matrices for symmetry {symmetry_name}, got shape {tuple(matrix.shape)}."
            )

        bank_axis_name = self.transform_bank_axis(symmetry_name) if matrix.ndim == 3 else None
        if matrix.ndim == 3 and bank_axis_name is None:
            raise ValueError(f"Symmetry {symmetry_name} does not support banked transforms.")

        matched_any_axis = False
        for symmetry_key, components in self.items():
            for component_name, component in list(components.items()):
                axis_indices = component.axis_indices(symmetry_name)
                if not axis_indices:
                    continue
                matched_any_axis = True
                tensor = component.tensor
                cast_matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)
                transformed = tensor

                if cast_matrix.ndim == 3:
                    if len(axis_indices) != 1:
                        raise ValueError(
                            f"Banked transform for symmetry {symmetry_name} requires exactly one matching axis in parameter component {component_name}."
                        )

                    bank_axis_indices = component.axis_indices(bank_axis_name)
                    if len(bank_axis_indices) != 1:
                        raise ValueError(
                            f"Banked transform for symmetry {symmetry_name} requires parameter component {component_name} to carry bank axis {bank_axis_name}."
                        )

                    axis_index = axis_indices[0]
                    bank_axis_index = bank_axis_indices[0]
                    if cast_matrix.shape[1] != cast_matrix.shape[2]:
                        raise ValueError(
                            f"Banked transform for symmetry {symmetry_name} must be square per bank, got shape {tuple(cast_matrix.shape)}."
                        )
                    if transformed.shape[axis_index] != cast_matrix.shape[1]:
                        raise ValueError(
                            f"Banked transform shape {tuple(cast_matrix.shape)} does not match symmetry axis {axis_index} of parameter component {component_name} with shape {tuple(transformed.shape)}."
                        )
                    if transformed.shape[bank_axis_index] != cast_matrix.shape[0]:
                        raise ValueError(
                            f"Banked transform shape {tuple(cast_matrix.shape)} does not match bank axis {bank_axis_index} of parameter component {component_name} with shape {tuple(transformed.shape)}."
                        )
                    transformed = self._apply_banked_axis_transform(
                        transformed,
                        cast_matrix,
                        bank_axis_index,
                        axis_index,
                    )
                    components[component_name] = component.with_tensor(transformed)
                    continue

                for axis_index in axis_indices:
                    if transformed.shape[axis_index] != cast_matrix.shape[0] or cast_matrix.shape[0] != cast_matrix.shape[1]:
                        raise ValueError(
                            f"Matrix shape {tuple(cast_matrix.shape)} does not match axis {axis_index} of parameter component {component_name} with shape {tuple(transformed.shape)}."
                        )
                    transformed = self._apply_axis_transform(transformed, cast_matrix, axis_index)
                components[component_name] = component.with_tensor(transformed)

        if not matched_any_axis and symmetry_name in self:
            return self
        return self

    def apply_attention_dual_transform(self, symmetry_name: str, matrix: torch.Tensor):
        roles = self._attention_dual_roles(symmetry_name)
        if roles is None:
            return self.apply_transform(symmetry_name, matrix)

        if symmetry_name not in self and not self.components_with_axis(symmetry_name):
            raise ValueError(f"Symmetry {symmetry_name} not found.")

        if matrix.ndim not in {2, 3}:
            raise ValueError(
                f"Expected a square matrix or bank of square matrices for symmetry {symmetry_name}, got shape {tuple(matrix.shape)}."
            )

        bank_axis_name = self.transform_bank_axis(symmetry_name) if matrix.ndim == 3 else None
        if matrix.ndim == 3 and bank_axis_name is None:
            raise ValueError(f"Symmetry {symmetry_name} does not support banked transforms.")

        matched_any_axis = False
        for _, components in self.items():
            for component_name, component in list(components.items()):
                axis_indices = component.axis_indices(symmetry_name)
                if not axis_indices:
                    continue
                matched_any_axis = True
                tensor = component.tensor
                cast_matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)
                dual_matrix = torch.linalg.inv(cast_matrix).transpose(-1, -2)
                role = self._attention_dual_role(symmetry_name, component_name, component)
                component_matrix = dual_matrix if role == roles[1] else cast_matrix
                transformed = tensor

                if component_matrix.ndim == 3:
                    if len(axis_indices) != 1:
                        raise ValueError(
                            f"Banked transform for symmetry {symmetry_name} requires exactly one matching axis in parameter component {component_name}."
                        )

                    bank_axis_indices = component.axis_indices(bank_axis_name)
                    if len(bank_axis_indices) != 1:
                        raise ValueError(
                            f"Banked transform for symmetry {symmetry_name} requires parameter component {component_name} to carry bank axis {bank_axis_name}."
                        )

                    axis_index = axis_indices[0]
                    bank_axis_index = bank_axis_indices[0]
                    if component_matrix.shape[1] != component_matrix.shape[2]:
                        raise ValueError(
                            f"Banked transform for symmetry {symmetry_name} must be square per bank, got shape {tuple(component_matrix.shape)}."
                        )
                    if transformed.shape[axis_index] != component_matrix.shape[1]:
                        raise ValueError(
                            f"Banked transform shape {tuple(component_matrix.shape)} does not match symmetry axis {axis_index} of parameter component {component_name} with shape {tuple(transformed.shape)}."
                        )
                    if transformed.shape[bank_axis_index] != component_matrix.shape[0]:
                        raise ValueError(
                            f"Banked transform shape {tuple(component_matrix.shape)} does not match bank axis {bank_axis_index} of parameter component {component_name} with shape {tuple(transformed.shape)}."
                        )
                    transformed = self._apply_banked_axis_transform(
                        transformed,
                        component_matrix,
                        bank_axis_index,
                        axis_index,
                    )
                    components[component_name] = component.with_tensor(transformed)
                    continue

                for axis_index in axis_indices:
                    if transformed.shape[axis_index] != component_matrix.shape[0] or component_matrix.shape[0] != component_matrix.shape[1]:
                        raise ValueError(
                            f"Matrix shape {tuple(component_matrix.shape)} does not match axis {axis_index} of parameter component {component_name} with shape {tuple(transformed.shape)}."
                        )
                    transformed = self._apply_axis_transform(transformed, component_matrix, axis_index)
                components[component_name] = component.with_tensor(transformed)

        if not matched_any_axis and symmetry_name in self:
            return self
        return self

    def apply_qk_dual_transform(self, symmetry_name: str, matrix: torch.Tensor):
        return self.apply_attention_dual_transform(symmetry_name, matrix)

    def apply_ov_dual_transform(self, symmetry_name: str, matrix: torch.Tensor):
        return self.apply_attention_dual_transform(symmetry_name, matrix)

    def apply_head_transport(self, layer: int | str, matrix: torch.Tensor):
        """Apply a transform to a layer's head symmetry.

        Args:
            layer: Either an integer layer index or an explicit head symmetry name.
            matrix: Transform matrix or bank of matrices to apply.

        Returns:
            ``self`` after delegating to ``apply_transform`` for the resolved head
            symmetry name.

        Process:
            Convert integer layer indices into ``L{layer}.head`` names and then reuse
            the general symmetry-transform path.
        """
        symmetry_name = layer if isinstance(layer, str) else f"L{layer}.head"
        return self.apply_transform(symmetry_name, matrix)

    def ordered_transform_names(self) -> list[str]:
        """Return symmetry names in the order transforms should be applied.

        Args:
            None.

        Returns:
            A list of symmetry names ordered with global symmetries first, then layer
            symmetries in layer-and-kind order, then any remaining names except
            ``vocab``.

        Process:
            Add ``model`` and ``decoder`` when present, enumerate layer indices from
            recognized layer symmetry names, append known layer suffixes in a fixed
            order, and finally append any leftover non-``vocab`` symmetries.
        """
        ordered: list[str] = []
        for symmetry_name in ("model", "decoder"):
            if symmetry_name in self:
                ordered.append(symmetry_name)

        layer_indices = sorted(
            {
                int(match.group("layer"))
                for symmetry_name in self
                for match in [_LAYER_SYMMETRY_RE.match(symmetry_name)]
                if match is not None
            }
        )
        for layer_idx in layer_indices:
            for suffix in ("qk", "ov", "mlp", "head"):
                symmetry_name = f"L{layer_idx}.{suffix}"
                if symmetry_name in self:
                    ordered.append(symmetry_name)

        seen = set(ordered)
        ordered.extend(
            symmetry_name
            for symmetry_name in self
            if symmetry_name not in seen and symmetry_name != "vocab"
        )
        return ordered

    def apply_transforms(self, transforms: dict[str, torch.Tensor]):
        """Apply a collection of named transforms in canonical order.

        Args:
            transforms: Mapping from symmetry names to transform tensors, with an
                optional shared ``head`` entry used as a fallback for all head
                symmetries.

        Returns:
            ``self`` after applying every available transform.

        Process:
            Iterate through ``ordered_transform_names()``, choose the symmetry-specific
            transform when available, fall back to the shared ``head`` transform for
            head symmetries, and dispatch to either ``apply_head_transport`` or
            ``apply_transform``.
        """
        shared_head_transform = transforms.get("head")
        for symmetry_name in self.ordered_transform_names():
            matrix = transforms.get(symmetry_name)
            if matrix is None and symmetry_name.endswith(".head"):
                matrix = shared_head_transform
            if matrix is None:
                continue
            if symmetry_name.endswith(".head"):
                self.apply_head_transport(symmetry_name, matrix)
            elif symmetry_name.endswith((".qk", ".ov")):
                self.apply_attention_dual_transform(symmetry_name, matrix)
            else:
                self.apply_transform(symmetry_name, matrix)
        return self

    def clone(self):
        """Deep-clone the collection and all stored component tensors.

        Args:
            None.

        Returns:
            A new ``Symmeters`` with cloned tensors and preserved metadata.

        Process:
            Rebuild the symmetry dictionary while cloning each component tensor via
            ``_clone_tensor`` and preserving all component metadata.
        """
        return Symmeters.from_symmetry_dict(
            {
                symmetry_name: {
                    component_name: component.with_tensor(_clone_tensor(component.tensor))
                    for component_name, component in components.items()
                }
                for symmetry_name, components in self.items()
            }
        )

    def save(self, path: str):
        """Persist the symmetry collection to disk in the native format.

        Args:
            path: Destination filesystem path for the serialized payload.

        Returns:
            None.

        Process:
            Materialize every component to its payload dictionary, wrap the result in
            the current ``format_version`` envelope, and write it with ``torch.save``.
        """
        torch.save(
            {
                "format_version": 2,
                "symmetries": {
                    symmetry_name: {
                        component_name: component.to_payload()
                        for component_name, component in components.items()
                    }
                    for symmetry_name, components in self.items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str):
        """Load a symmetry collection from disk.

        Args:
            path: Filesystem path containing a serialized symmetry payload.

        Returns:
            A ``Symmeters`` instance reconstructed from the saved data.

        Process:
            Read the file with ``torch.load``, verify that the payload uses the
            supported native block format, and rebuild the collection from the stored
            symmetry dictionary.
        """
        data = torch.load(path)
        if data.get("format_version") != 2:
            raise ValueError("This project now persists only the block-native symmetry format (format_version=2).")
        return cls.from_symmetry_dict(data["symmetries"])

    def __str__(self):
        """Render a readable summary of symmetries and component metadata.

        Args:
            None.

        Returns:
            A stringified dictionary where tensors are represented by shape tuples
            instead of full values.

        Process:
            Convert each component to its payload form, replace tensor values with
            their shapes for readability, and stringify the resulting nested mapping.
        """
        return str(
            {
                symmetry_name: {
                    component_name: {
                        key: value if key != "tensor" else tuple(value.shape)
                        for key, value in component.to_payload().items()
                    }
                    for component_name, component in components.items()
                }
                for symmetry_name, components in self.items()
            }
        )