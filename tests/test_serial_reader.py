import pytest
import torch

from lib.serial_params import Symmeters, NamedSerialParameters
from lib.serial_reader import (
    SymmetersReader,
    SerializedParameterOverrides,
    SerializedParameterReader,
)


def make_named(names, rows):
    return NamedSerialParameters.from_vector_list(names, [torch.tensor(rows, dtype=torch.float32)])


def test_named_reader_reads_matrix_bias_and_layernorm_in_order():
    reader = SerializedParameterReader(
        make_named(
            ["proj.0", "proj.1", "proj.bias", "norm.weight", "norm.bias"],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        )
    )

    torch.testing.assert_close(reader.read_matrix("proj", 2), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(reader.read_bias("proj"), torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(reader.read_optional_layernorm("norm"), torch.tensor([[7.0, 8.0], [9.0, 10.0]]))
    reader.assert_done()


def test_symmeters_reader_selects_requested_symmetry():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": make_named(["block.0"], [[1.0, 2.0]]),
            "L0.H0.qk": make_named(["attn.bias"], [[3.0, 4.0]]),
        }
    )

    model_reader = SerializedParameterReader(symmeters, "model")
    qk_reader = SerializedParameterReader(symmeters, "L0.H0.qk")
    missing_reader = SerializedParameterReader(symmeters, "decoder")

    torch.testing.assert_close(model_reader.read_matrix("block", 1), torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(qk_reader.read_bias("attn"), torch.tensor([3.0, 4.0]))
    assert missing_reader.peek() is None
    missing_reader.assert_done()


def test_symmeters_reader_caches_readers_and_checks_all_symmetries():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": make_named(["block.0"], [[1.0, 2.0]]),
            "L0.H0.qk": make_named(["attn.bias"], [[3.0, 4.0]]),
        }
    )

    readers = SymmetersReader(symmeters)

    assert readers["model"] is readers["model"]
    assert readers["L0.H0.qk"] is readers["L0.H0.qk"]
    torch.testing.assert_close(readers["model"].read_matrix("block", 1), torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(readers["L0.H0.qk"].read_bias("attn"), torch.tensor([3.0, 4.0]))
    readers.assert_done()


def test_reader_can_reassemble_head_partitioned_biases():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "L0.H0.qk": make_named(["attn.head.0.bias"], [[1.0, 2.0]]),
            "L0.H1.qk": make_named(["attn.head.1.bias"], [[3.0, 4.0]]),
        }
    )

    overrides = SerializedParameterOverrides(symmeters)

    torch.testing.assert_close(
        overrides.head_bias(
            "attn.bias",
            symmetry_names=["L0.H0.qk", "L0.H1.qk"],
        ),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    overrides.assert_done()


def test_reader_startswith_and_wrapper_supports_flat_inputs_and_missing_symmetries():
    readers = SymmetersReader(make_named(["block.0"], [[1.0, 2.0]]))

    assert readers["model"].startswith("block")
    assert not readers["model"].startswith("other")
    assert readers["decoder"].peek() is None
    readers["decoder"].assert_done()


def test_overrides_context_reads_and_stores_values():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": make_named(["proj.weight.0", "proj.weight.1", "norm.weight", "norm.bias"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            "proj.weight": make_named(["proj.bias"], [[9.0, 10.0]]),
        }
    )

    overrides = SerializedParameterOverrides(symmeters)

    overrides.matrix("proj.weight", 2)
    overrides.bias("proj.bias", symmetry="proj.weight")
    overrides.optional_layernorm("norm")

    torch.testing.assert_close(overrides["proj.weight"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(overrides["proj.bias"], torch.tensor([9.0, 10.0]))
    torch.testing.assert_close(overrides["norm.weight"], torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(overrides["norm.bias"], torch.tensor([7.0, 8.0]))
    overrides.assert_done()


def test_overrides_has_prefix_can_check_specific_symmetries():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": make_named(["residual.0"], [[1.0, 2.0]]),
            "decoder": make_named(["decoder.weight.0"], [[3.0, 4.0]]),
        }
    )

    overrides = SerializedParameterOverrides(symmeters)

    assert overrides.has_prefix("decoder.weight")
    assert overrides.has_prefix("decoder.weight", symmetry="decoder")
    assert not overrides.has_prefix("decoder.weight", symmetry="model")


def test_read_head_matrix_uses_head_indexed_order():
    reader = SerializedParameterReader(
        make_named(
            [
                "attn.head.0.0",
                "attn.head.0.1",
                "attn.head.1.0",
                "attn.head.1.1",
            ],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        )
    )

    torch.testing.assert_close(
        reader.read_head_matrix("attn", num_heads=2, head_dim=2),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )
    reader.assert_done()


def test_take_rejects_unexpected_names():
    reader = SerializedParameterReader(make_named(["block.0"], [[1.0, 2.0]]))

    with pytest.raises(AssertionError, match="Expected rows"):
        reader.take(["other.0"])


def test_assert_done_rejects_unconsumed_rows():
    reader = SerializedParameterReader(make_named(["block.0", "block.1"], [[1.0, 2.0], [3.0, 4.0]]))
    reader.take(["block.0"])

    with pytest.raises(AssertionError, match="Unused serialized rows remain"):
        reader.assert_done()