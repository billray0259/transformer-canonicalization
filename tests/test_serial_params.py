import pytest
import torch

from lib.serial_params import NamedSerialParameters


def test_constructor_rejects_mismatched_name_count():
    with pytest.raises(AssertionError, match="same length"):
        NamedSerialParameters(names=["only"], vectors=torch.randn(2, 3))


def test_empty_instance_has_no_names_and_no_vectors():
    params = NamedSerialParameters()

    assert params.names == []
    assert params.vectors is None


def test_from_vector_list_concatenates_lazily_and_caches_result():
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second = torch.tensor([[5.0, 6.0]])
    params = NamedSerialParameters.from_vector_list(
        ["block.0", "block.1", "block.2"],
        [first, second],
    )

    assert params._vectors is None

    vectors = params.vectors

    torch.testing.assert_close(vectors, torch.cat([first, second], dim=0))
    assert params._vector_list == [vectors]
    assert params.vectors.data_ptr() == vectors.data_ptr()


def test_add_combines_names_and_vectors_without_mutating_inputs():
    left = NamedSerialParameters.from_vector_list(["left.0"], [torch.tensor([[1.0, 2.0]])])
    right = NamedSerialParameters.from_vector_list(
        ["right.0", "right.1"],
        [torch.tensor([[3.0, 4.0], [5.0, 6.0]])],
    )

    combined = left + right

    assert combined.names == ["left.0", "right.0", "right.1"]
    torch.testing.assert_close(
        combined.vectors,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert left.names == ["left.0"]
    assert right.names == ["right.0", "right.1"]


def test_adding_non_serial_parameters_raises_type_error():
    params = NamedSerialParameters.from_vector_list(["row.0"], [torch.tensor([[1.0, 2.0]])])

    with pytest.raises(TypeError):
        _ = params + 1
