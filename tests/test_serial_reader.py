import pytest
import torch

from lib.serial_params import NamedSerialParameters
from lib.serial_reader import SerializedParameterReader


def make_reader(names, rows):
    return SerializedParameterReader(NamedSerialParameters.from_vector_list(names, [rows]))


def test_split_matrix_and_bias_returns_matrix_and_inline_bias():
    rows = torch.tensor(
        [
            [1.0, 2.0, 10.0],
            [3.0, 4.0, 20.0],
        ]
    )

    matrix, bias = SerializedParameterReader.split_matrix_and_bias(rows)

    torch.testing.assert_close(matrix, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(bias, torch.tensor([10.0, 20.0]))


def test_split_matrix_and_bias_returns_none_when_bias_column_is_nan():
    rows = torch.tensor(
        [
            [1.0, 2.0, float("nan")],
            [3.0, 4.0, float("nan")],
        ]
    )

    matrix, bias = SerializedParameterReader.split_matrix_and_bias(rows)

    torch.testing.assert_close(matrix, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert bias is None


def test_split_matrix_and_bias_rejects_mixed_bias_padding():
    rows = torch.tensor(
        [
            [1.0, 2.0, float("nan")],
            [3.0, 4.0, 5.0],
        ]
    )

    with pytest.raises(AssertionError, match="mix padded and non-padded"):
        SerializedParameterReader.split_matrix_and_bias(rows)


def test_peek_startswith_and_take_advance_reader_cursor():
    reader = make_reader(
        ["block.0", "block.1", "tail.bias"],
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, float("nan")],
            ]
        ),
    )

    assert reader.peek() == "block.0"
    assert reader.startswith("block")

    rows = reader.take(["block.0", "block.1"])

    torch.testing.assert_close(rows, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert reader.peek() == "tail.bias"
    assert not reader.startswith("block")


def test_take_rejects_unexpected_names():
    reader = make_reader(["block.0"], torch.tensor([[1.0, 2.0]]))

    with pytest.raises(AssertionError, match="Expected rows"):
        reader.take(["other.0"])


def test_take_optional_layernorm_returns_rows_and_none_when_absent():
    reader = make_reader(
        ["norm.weight", "norm.bias", "next.0"],
        torch.tensor(
            [
                [1.0, 2.0, float("nan")],
                [3.0, 4.0, float("nan")],
                [5.0, 6.0, float("nan")],
            ]
        ),
    )

    rows = reader.take_optional_layernorm("norm")
    missing = reader.take_optional_layernorm("missing")

    torch.testing.assert_close(
        rows,
        torch.tensor(
            [
                [1.0, 2.0, float("nan")],
                [3.0, 4.0, float("nan")],
            ]
        ),
        equal_nan=True,
    )
    assert missing is None
    assert reader.peek() == "next.0"


def test_read_matrix_reads_prefixed_rows_and_splits_inline_bias():
    reader = make_reader(
        ["proj.0", "proj.1"],
        torch.tensor(
            [
                [1.0, 2.0, 10.0],
                [3.0, 4.0, 20.0],
            ]
        ),
    )

    matrix, bias = reader.read_matrix("proj", 2)

    torch.testing.assert_close(matrix, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(bias, torch.tensor([10.0, 20.0]))


def test_read_head_matrix_reads_head_indexed_rows_in_expected_order():
    reader = make_reader(
        [
            "attn.head.0.0",
            "attn.head.0.1",
            "attn.head.1.0",
            "attn.head.1.1",
        ],
        torch.tensor(
            [
                [1.0, 2.0, 10.0],
                [3.0, 4.0, 20.0],
                [5.0, 6.0, 30.0],
                [7.0, 8.0, 40.0],
            ]
        ),
    )

    matrix, bias = reader.read_head_matrix("attn", num_heads=2, head_dim=2)

    torch.testing.assert_close(
        matrix,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
    )
    torch.testing.assert_close(bias, torch.tensor([10.0, 20.0, 30.0, 40.0]))


def test_read_bias_requires_padded_nan_sentinel():
    reader = make_reader(["proj.bias"], torch.tensor([[1.0, 2.0, 3.0]]))

    with pytest.raises(AssertionError, match="Expected padded bias row"):
        reader.read_bias("proj")


def test_read_optional_layernorm_strips_padding_and_validates_nan_sentinel():
    reader = make_reader(
        ["norm.weight", "norm.bias"],
        torch.tensor(
            [
                [1.0, 2.0, float("nan")],
                [3.0, 4.0, float("nan")],
            ]
        ),
    )

    rows = reader.read_optional_layernorm("norm")

    torch.testing.assert_close(rows, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_read_optional_layernorm_rejects_non_nan_padding():
    reader = make_reader(
        ["norm.weight", "norm.bias"],
        torch.tensor(
            [
                [1.0, 2.0, float("nan")],
                [3.0, 4.0, 5.0],
            ]
        ),
    )

    with pytest.raises(AssertionError, match="Expected padded LayerNorm rows"):
        reader.read_optional_layernorm("norm")


def test_assert_done_rejects_unconsumed_rows():
    reader = make_reader(
        ["block.0", "block.1"],
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )
    reader.take(["block.0"])

    with pytest.raises(AssertionError, match="Unused serialized rows remain"):
        reader.assert_done()


def test_assert_done_passes_after_full_consumption():
    reader = make_reader(["block.0"], torch.tensor([[1.0, 2.0]]))

    reader.take(["block.0"])

    reader.assert_done()