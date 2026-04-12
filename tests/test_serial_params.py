import pytest
import torch

from lib.serial_params import Symmeters


def test_symmeters_add_merges_block_components_without_row_metadata():
    left = Symmeters(["model"])
    left.add_component(
        "model",
        "left.weight",
        torch.tensor([[1.0, 2.0]]),
        axes=("rows", "model"),
        kind="weight",
        layout="identity",
        parameter_keys="left.weight",
    )

    right = Symmeters(["model", "L0.mlp"])
    right.add_component(
        "model",
        "right.bias",
        torch.tensor([3.0, 4.0]),
        axes=("model",),
        kind="bias",
        layout="identity",
        parameter_keys="right.bias",
    )
    right.add_component(
        "L0.mlp",
        "mlp.bias",
        torch.tensor([5.0, 6.0]),
        axes=("L0.mlp",),
        kind="bias",
        layout="identity",
        parameter_keys="mlp.bias",
    )

    combined = left + right

    assert combined.symmetry_names == ["model", "L0.mlp"]
    torch.testing.assert_close(combined.tensor("model", "left.weight"), torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(combined.tensor("model", "right.bias"), torch.tensor([3.0, 4.0]))
    assert combined.component("L0.mlp", "mlp.bias").axes == ("L0.mlp",)


def test_apply_transform_updates_linked_model_axis_and_owned_axis():
    symmeters = Symmeters(["model", "L0.qk"])
    symmeters.add_component(
        "model",
        "residual.weight",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        axes=("rows", "model"),
        kind="weight",
        layout="identity",
        parameter_keys="residual.weight",
    )
    symmeters.add_component(
        "L0.qk",
        "query.weight",
        torch.tensor(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
        axes=("L0.head", "L0.qk", "model"),
        kind="weight",
        layout="head_rows",
        parameter_keys="query.weight",
    )
    symmeters.add_component(
        "L0.qk",
        "query.bias",
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        axes=("L0.head", "L0.qk"),
        kind="bias",
        layout="head_bias",
        parameter_keys="query.bias",
    )

    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    symmeters.apply_transform("model", swap)
    symmeters.apply_transform("L0.qk", torch.stack([swap, torch.eye(2)]))

    torch.testing.assert_close(symmeters.tensor("model", "residual.weight"), torch.tensor([[2.0, 1.0], [4.0, 3.0]]))
    torch.testing.assert_close(
        symmeters.tensor("L0.qk", "query.weight"),
        torch.tensor(
            [
                [[40.0, 30.0], [20.0, 10.0]],
                [[60.0, 50.0], [80.0, 70.0]],
            ]
        ),
    )
    torch.testing.assert_close(symmeters.tensor("L0.qk", "query.bias"), torch.tensor([[6.0, 5.0], [7.0, 8.0]]))


def test_apply_head_transport_updates_all_head_stacked_components():
    symmeters = Symmeters(["L0.qk", "L0.ov", "L0.head"])
    symmeters.add_component(
        "L0.qk",
        "query.bias",
        torch.tensor([[1.0], [2.0]]),
        axes=("L0.head", "L0.qk"),
        kind="bias",
        layout="head_bias",
        parameter_keys="query.bias",
    )
    symmeters.add_component(
        "L0.ov",
        "value.weight",
        torch.tensor([[[10.0, 11.0]], [[20.0, 21.0]]]),
        axes=("L0.head", "L0.ov", "model"),
        kind="weight",
        layout="head_rows",
        parameter_keys="value.weight",
    )

    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    symmeters.apply_head_transport(0, swap)

    torch.testing.assert_close(symmeters.tensor("L0.qk", "query.bias"), torch.tensor([[2.0], [1.0]]))
    torch.testing.assert_close(
        symmeters.tensor("L0.ov", "value.weight"),
        torch.tensor([[[20.0, 21.0]], [[10.0, 11.0]]]),
    )


def test_apply_transform_handles_repeated_model_axis_for_tied_blocks():
    symmeters = Symmeters(["model"])
    symmeters.add_component(
        "model",
        "cls.predictions.transform.dense.weight",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        axes=("model", "model"),
        kind="weight",
        layout="identity",
        parameter_keys="cls.predictions.transform.dense.weight",
    )

    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    symmeters.apply_transform("model", swap)

    torch.testing.assert_close(
        symmeters.tensor("model", "cls.predictions.transform.dense.weight"),
        torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
    )


def test_apply_qk_dual_transform_uses_inverse_transpose_for_key_components():
    symmeters = Symmeters(["L0.qk"])
    symmeters.add_component(
        "L0.qk",
        "query.weight",
        torch.eye(2).unsqueeze(0),
        axes=("L0.head", "rows", "L0.qk"),
        kind="weight",
        layout="head_rows",
        parameter_keys="query.weight",
    )
    symmeters.add_component(
        "L0.qk",
        "key.weight",
        torch.eye(2).unsqueeze(0),
        axes=("L0.head", "rows", "L0.qk"),
        kind="weight",
        layout="head_rows",
        parameter_keys="key.weight",
    )
    symmeters.add_component(
        "L0.qk",
        "query.bias",
        torch.tensor([[1.0, 1.0]]),
        axes=("L0.head", "L0.qk"),
        kind="bias",
        layout="head_bias",
        parameter_keys="query.bias",
    )
    symmeters.add_component(
        "L0.qk",
        "key.bias",
        torch.tensor([[1.0, 1.0]]),
        axes=("L0.head", "L0.qk"),
        kind="bias",
        layout="head_bias",
        parameter_keys="key.bias",
    )

    transform = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    symmeters.apply_qk_dual_transform("L0.qk", transform)

    torch.testing.assert_close(symmeters.tensor("L0.qk", "query.weight"), transform)
    torch.testing.assert_close(
        symmeters.tensor("L0.qk", "key.weight"),
        torch.tensor([[[0.5, 0.0], [0.0, 0.25]]]),
    )
    torch.testing.assert_close(symmeters.tensor("L0.qk", "query.bias"), torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(symmeters.tensor("L0.qk", "key.bias"), torch.tensor([[0.5, 0.25]]))


def test_apply_ov_dual_transform_uses_inverse_transpose_for_output_components():
    symmeters = Symmeters(["L0.ov"])
    symmeters.add_component(
        "L0.ov",
        "value.weight",
        torch.eye(2).unsqueeze(0),
        axes=("L0.head", "rows", "L0.ov"),
        kind="weight",
        layout="head_rows",
        parameter_keys="value.weight",
    )
    symmeters.add_component(
        "L0.ov",
        "output.dense.weight",
        torch.eye(2).unsqueeze(0),
        axes=("L0.head", "rows", "L0.ov"),
        kind="weight",
        layout="head_rows_transposed",
        parameter_keys="output.dense.weight",
    )
    symmeters.add_component(
        "L0.ov",
        "value.bias",
        torch.tensor([[1.0, 1.0]]),
        axes=("L0.head", "L0.ov"),
        kind="bias",
        layout="head_bias",
        parameter_keys="value.bias",
    )

    transform = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    symmeters.apply_ov_dual_transform("L0.ov", transform)

    torch.testing.assert_close(symmeters.tensor("L0.ov", "value.weight"), transform)
    torch.testing.assert_close(
        symmeters.tensor("L0.ov", "output.dense.weight"),
        torch.tensor([[[0.5, 0.0], [0.0, 0.25]]]),
    )
    torch.testing.assert_close(symmeters.tensor("L0.ov", "value.bias"), torch.tensor([[2.0, 4.0]]))


def test_clone_save_and_load_roundtrip(tmp_path):
    symmeters = Symmeters(["model", "L0.head"])
    symmeters.add_component(
        "model",
        "weight",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
        axes=("rows", "model"),
        kind="weight",
        layout="identity",
        parameter_keys="weight",
    )

    cloned = symmeters.clone()
    with torch.no_grad():
        cloned["model"]["weight"].tensor[0, 0] = 99.0

    torch.testing.assert_close(symmeters.tensor("model", "weight"), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    path = tmp_path / "symmeters.pt"
    symmeters.save(path)
    loaded = Symmeters.load(path)

    torch.testing.assert_close(loaded.tensor("model", "weight"), symmeters.tensor("model", "weight"))
    assert loaded.component("model", "weight").axes == ("rows", "model")


def test_apply_transform_rejects_missing_symmetry():
    symmeters = Symmeters(["model"])

    with pytest.raises(ValueError, match="Symmetry missing not found"):
        symmeters.apply_transform("missing", torch.eye(2))
