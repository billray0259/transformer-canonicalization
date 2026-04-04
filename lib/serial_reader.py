import torch

from lib.serial_params import Symmeters, NamedSerialParameters



class SerializedParameterReader:
    """Consumes serialized parameter rows in the expected order."""

    def __init__(
        self,
        serialized_params: NamedSerialParameters | Symmeters,
        symmetry_name: str | None = None,
    ) -> None:
        """Initialize the reader over one serialized parameter symmetry."""
        if isinstance(serialized_params, Symmeters):
            assert symmetry_name is not None, "A symmetry name is required for Symmeters inputs."
            serialized_params = serialized_params.get(symmetry_name, NamedSerialParameters())
        self.names = serialized_params.names
        self.vectors = serialized_params.vectors
        if self.vectors is None:
            self.vectors = torch.empty((0, 0))
        assert len(self.names) == self.vectors.shape[0], "Serialized names and vectors must have the same length."
        self.index = 0

    def peek(self) -> str | None:
        """Return the next row name without advancing the cursor."""
        if self.index >= len(self.names):
            return None
        return self.names[self.index]

    def startswith(self, prefix: str) -> bool:
        """Check whether the next row name starts with the given prefix."""
        name = self.peek()
        return name is not None and name.startswith(prefix)

    def has_prefix(self, prefix: str) -> bool:
        """Check whether any row name starts with the given prefix."""
        return any(name.startswith(prefix) for name in self.names)

    def take(self, expected_names: list[str]) -> torch.Tensor:
        """Consume a fixed sequence of named rows and return their tensor block."""
        end = self.index + len(expected_names)
        actual_names = self.names[self.index:end]
        assert actual_names == expected_names, f"Expected rows {expected_names}, got {actual_names}."
        rows = self.vectors[self.index:end]
        self.index = end
        return rows

    def take_optional_layernorm(self, prefix: str) -> torch.Tensor | None:
        """Consume a LayerNorm weight/bias pair when present."""
        if self.peek() != f"{prefix}.weight":
            return None
        return self.take([f"{prefix}.weight", f"{prefix}.bias"])

    def read_matrix(self, prefix: str, row_count: int) -> torch.Tensor:
        """Read a standard row-wise matrix block."""
        return self.take([f"{prefix}.{i}" for i in range(row_count)])

    def read_head_matrix(self, prefix: str, num_heads: int, head_dim: int) -> torch.Tensor:
        """Read a head-indexed matrix block using the serialized attention naming scheme."""
        return self.take([
            f"{prefix}.head.{head_idx}.{row_idx}"
            for head_idx in range(num_heads)
            for row_idx in range(head_dim)
        ])

    def read_bias(self, prefix: str) -> torch.Tensor:
        """Read a standalone bias row."""
        return self.take([f"{prefix}.bias"])[0]

    def read_head_bias(self, prefix: str, head_idx: int) -> torch.Tensor:
        """Read one head-partitioned bias row."""
        return self.take([f"{prefix}.head.{head_idx}.bias"])[0]

    def read_optional_layernorm(self, prefix: str) -> torch.Tensor | None:
        """Read a LayerNorm weight/bias pair when present."""
        return self.take_optional_layernorm(prefix)

    def assert_done(self) -> None:
        """Assert that the entire serialized symmetry has been consumed."""
        assert self.index == len(self.names), f"Unused serialized rows remain starting at {self.peek()}."


class SymmetersReader:
    """Lazily exposes per-symmetry readers over Symmeters."""

    def __init__(self, serialized_params: NamedSerialParameters | Symmeters) -> None:
        if isinstance(serialized_params, NamedSerialParameters):
            serialized_params = Symmeters.from_symmetry_dict({"model": serialized_params})
        self.symmeters = serialized_params
        self.readers = {
            symmetry_name: SerializedParameterReader(serialized_params, symmetry_name)
            for symmetry_name in serialized_params.symmetry_names
        }

    def __getitem__(self, symmetry_name: str) -> SerializedParameterReader:
        if symmetry_name not in self.readers:
            self.readers[symmetry_name] = SerializedParameterReader(self.symmeters, symmetry_name)
        return self.readers[symmetry_name]

    def assert_done(self) -> None:
        for reader in self.readers.values():
            reader.assert_done()


class SerializedParameterOverrides(dict):
    """Collects functional_call overrides while reading serialized parameter symmetries."""

    def __init__(self, serialized_params: NamedSerialParameters | Symmeters) -> None:
        super().__init__()
        self.readers = SymmetersReader(serialized_params)

    def has_prefix(self, prefix: str, symmetry: str | None = None) -> bool:
        if symmetry is not None:
            return self.readers[symmetry].has_prefix(prefix)
        return any(self.readers[symmetry_name].has_prefix(prefix) for symmetry_name in self.readers.symmeters.symmetry_names)

    def matrix(
        self,
        key: str,
        row_count: int,
        *,
        symmetry: str = "model",
        src: str | None = None,
        transpose: bool = False,
    ) -> torch.Tensor:
        value = self.readers[symmetry].read_matrix(src or key, row_count)
        self[key] = value.T if transpose else value
        return self[key]

    def head_matrix(
        self,
        key: str,
        *,
        num_heads: int,
        head_dim: int,
        symmetry: str = "model",
        src: str | None = None,
        transpose: bool = False,
    ) -> torch.Tensor:
        value = self.readers[symmetry].read_head_matrix(src or key, num_heads, head_dim)
        self[key] = value.T if transpose else value
        return self[key]

    def bias(self, key: str, *, symmetry: str = "model", src: str | None = None) -> torch.Tensor:
        self[key] = self.readers[symmetry].read_bias(src or key.removesuffix(".bias"))
        return self[key]

    def head_bias(
        self,
        key: str,
        *,
        symmetry_names: list[str],
        src: str | None = None,
    ) -> torch.Tensor:
        prefix = src or key.removesuffix(".bias")
        self[key] = torch.cat(
            [
                self.readers[symmetry_name].read_head_bias(prefix, head_idx)
                for head_idx, symmetry_name in enumerate(symmetry_names)
            ]
        )
        return self[key]

    def optional_layernorm(
        self,
        key_prefix: str,
        *,
        symmetry: str = "model",
        src: str | None = None,
    ) -> torch.Tensor | None:
        rows = self.readers[symmetry].read_optional_layernorm(src or key_prefix)
        if rows is None:
            return None
        self[f"{key_prefix}.weight"] = rows[0]
        self[f"{key_prefix}.bias"] = rows[1]
        return rows

    def assert_done(self) -> None:
        self.readers.assert_done()
