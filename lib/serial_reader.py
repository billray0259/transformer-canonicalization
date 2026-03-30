import torch

from lib.serial_params import NamedSerialParameters



class SerializedParameterReader:
    """Consumes serialized parameter rows in the expected order."""

    @staticmethod
    def split_matrix_and_bias(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split serialized rows into a matrix block and optional inline bias."""
        # rows: (row_count, hidden_size + 1) -> matrix: (row_count, hidden_size)
        matrix = rows[:, :-1]
        # bias_column: (row_count,), where NaN means "no inline bias"
        bias_column = rows[:, -1]
        has_bias = ~torch.isnan(bias_column)
        assert bool(has_bias.all() or (~has_bias).all()), "Serialized rows mix padded and non-padded bias columns."
        bias = None if bool((~has_bias).all()) else bias_column
        return matrix, bias

    def __init__(self, serialized_params: NamedSerialParameters) -> None:
        """Initialize the reader over a flat serialized parameter stream."""
        self.names = serialized_params.names
        self.vectors = serialized_params.vectors
        assert self.vectors is not None, "Serialized parameters must include vectors."
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

    def read_matrix(self, prefix: str, row_count: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read a standard row-wise matrix block with an optional inline bias column."""
        # Expected serialized block shape: (row_count, hidden_size + 1)
        rows = self.take([f"{prefix}.{i}" for i in range(row_count)])
        return self.split_matrix_and_bias(rows)

    def read_head_matrix(self, prefix: str, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read a head-indexed matrix block using the serialized attention naming scheme."""
        # Head-indexed blocks still flatten to (num_heads * head_dim, hidden_size + 1)
        rows = self.take([
            f"{prefix}.head.{head_idx}.{row_idx}"
            for head_idx in range(num_heads)
            for row_idx in range(head_dim)
        ])
        return self.split_matrix_and_bias(rows)

    def read_bias(self, prefix: str) -> torch.Tensor:
        """Read a standalone bias row padded with a trailing NaN sentinel."""
        rows = self.take([f"{prefix}.bias"])
        # rows[0, :-1]: (hidden_size,)
        assert torch.isnan(rows[0, -1]), f"Expected padded bias row for {prefix}."
        return rows[0, :-1]

    def read_optional_layernorm(self, prefix: str) -> torch.Tensor | None:
        """Read a LayerNorm weight/bias pair when present and validate sentinel padding."""
        rows = self.take_optional_layernorm(prefix)
        if rows is None:
            return None
        # rows: (2, hidden_size + 1) -> returned tensor: (2, hidden_size)
        assert torch.isnan(rows[:, -1]).all(), f"Expected padded LayerNorm rows for {prefix}."
        return rows[:, :-1]

    def assert_done(self) -> None:
        """Assert that the entire serialized stream has been consumed."""
        assert self.index == len(self.names), f"Unused serialized rows remain starting at {self.peek()}."
