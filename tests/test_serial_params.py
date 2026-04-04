import pytest
import torch

from lib.serial_params import Symmeters, NamedSerialParameters


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
    params = NamedSerialParameters.from_vector_list(["block.0", "block.1", "block.2"], [first, second])

    assert params._vectors is None
    torch.testing.assert_close(params.vectors, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert params._vector_list == [params.vectors]


def test_filter_preserves_row_structure():
    params = NamedSerialParameters.from_vector_list(
        ["keep.0", "drop.0", "keep.1"],
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
    )

    filtered = params.filter(lambda name: name.startswith("keep"))

    assert filtered.names == ["keep.0", "keep.1"]
    torch.testing.assert_close(filtered.vectors, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_symmeters_add_merges_matching_symmetries():
    left = Symmeters.from_symmetry_dict(
        {"model": NamedSerialParameters.from_vector_list(["left.0"], [torch.tensor([[1.0, 2.0]])])}
    )
    right = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(["right.0"], [torch.tensor([[3.0, 4.0]])]),
            "L0.mlp": NamedSerialParameters.from_vector_list(["bias.bias"], [torch.tensor([[5.0, 6.0]])]),
        }
    )
    right.set_equivalence_class("L0.mlp", ["bert.encoder.layer.0.intermediate.dense"])

    combined = left + right

    assert combined.symmetry_names == ["model", "L0.mlp"]
    assert combined["model"].names == ["left.0", "right.0"]
    torch.testing.assert_close(combined["model"].vectors, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(combined["L0.mlp"].vectors, torch.tensor([[5.0, 6.0]]))
    assert combined.get_equivalence_class("L0.mlp") == {"bert.encoder.layer.0.intermediate.dense"}


def test_symmeters_add_rejects_mismatched_equivalence_classes():
    left = Symmeters.from_symmetry_dict({"L0.mlp": NamedSerialParameters()})
    right = Symmeters.from_symmetry_dict({"L0.mlp": NamedSerialParameters()})
    left.set_equivalence_class("L0.mlp", ["left.prefix"])
    right.set_equivalence_class("L0.mlp", ["right.prefix"])

    with pytest.raises(ValueError, match="Mismatched equivalence classes"):
        left + right


def test_symmeters_add_named_routes_into_model_symmetry():
    symmeters = Symmeters([]) + NamedSerialParameters.from_vector_list(
        ["row.0"],
        [torch.tensor([[1.0, 2.0]])],
    )

    assert symmeters.symmetry_names == ["model"]
    torch.testing.assert_close(symmeters["model"].vectors, torch.tensor([[1.0, 2.0]]))


def test_arbitrary_symmetry_name_is_accepted():
    symmeters = Symmeters.from_symmetry_dict({"bad-stream": NamedSerialParameters()})

    assert symmeters.symmetry_names == ["bad-stream"]


def test_equivalent_model_rows_filters_model_symmetry_from_metadata():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.head.0.0", "proj.head.0.1", "other.0"],
                [torch.tensor([[1.0], [2.0], [3.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters(),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["proj.head.0"])

    filtered = symmeters.equivalent_model_rows("L0.H0.qk")

    assert filtered.names == ["proj.head.0.0", "proj.head.0.1"]


def test_equivalent_model_rows_do_not_overmatch_numeric_suffixes():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.head.1.0", "proj.head.10.0", "proj.head.11.0"],
                [torch.tensor([[1.0], [2.0], [3.0]])],
            ),
            "L0.H1.qk": NamedSerialParameters(),
        }
    )
    symmeters.set_equivalence_class("L0.H1.qk", ["proj.head.1"])

    filtered = symmeters.equivalent_model_rows("L0.H1.qk")

    assert filtered.names == ["proj.head.1.0"]


def test_equivalent_model_rows_returns_empty_without_prefixes_or_model_stream():
    no_prefixes = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(["row.0"], [torch.tensor([[1.0]])]),
            "L0.H0.qk": NamedSerialParameters(),
        }
    )
    no_model = Symmeters.from_symmetry_dict({"L0.H0.qk": NamedSerialParameters()})
    no_model.set_equivalence_class("L0.H0.qk", ["proj.head.0"])

    assert no_prefixes.equivalent_model_rows("L0.H0.qk").names == []
    assert no_model.equivalent_model_rows("L0.H0.qk").names == []


def test_apply_square_matrix_updates_symmetry_and_matching_model_rows():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.head.0.0", "proj.head.0.1", "other.0"],
                [torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 6.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(
                ["bias.0"],
                [torch.tensor([[2.0, 3.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["proj.head.0"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "L0.H0.qk")

    torch.testing.assert_close(symmeters["L0.H0.qk"].vectors, torch.tensor([[3.0, 2.0]]))
    torch.testing.assert_close(
        symmeters["model"].vectors,
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [5.0, 6.0]]),
    )


def test_apply_square_matrix_updates_matching_rows_in_same_symmetry():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "decoder": NamedSerialParameters.from_vector_list(
                [
                    "cls.predictions.transform.dense.weight.0",
                    "cls.predictions.transform.dense.weight.1",
                    "cls.predictions.transform.dense.bias",
                    "cls.predictions.decoder.weight.0",
                    "cls.predictions.decoder.weight.1",
                ],
                [torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [5.0, 6.0], [7.0, 8.0]])],
            )
        }
    )
    symmeters.set_equivalence_class("decoder", ["cls.predictions.transform.dense.weight"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "decoder")

    torch.testing.assert_close(
        symmeters["decoder"].vectors,
        torch.tensor([[3.0, 4.0], [1.0, 2.0], [20.0, 10.0], [6.0, 5.0], [8.0, 7.0]]),
    )


def test_apply_square_matrix_updates_decoder_bridge_rows_when_model_is_permuted():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["model.0", "model.1"],
                [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
            ),
            "decoder": NamedSerialParameters.from_vector_list(
                [
                    "cls.predictions.transform.dense.weight.0",
                    "cls.predictions.transform.dense.weight.1",
                    "cls.predictions.transform.dense.bias",
                    "cls.predictions.decoder.weight.0",
                ],
                [torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("decoder", ["cls.predictions.transform.dense.weight"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "model")

    torch.testing.assert_close(symmeters["model"].vectors, torch.tensor([[2.0, 1.0], [4.0, 3.0]]))
    torch.testing.assert_close(
        symmeters["decoder"].vectors,
        torch.tensor([[6.0, 5.0], [8.0, 7.0], [9.0, 10.0], [11.0, 12.0]]),
    )


def test_apply_square_matrix_ignores_same_symmetry_bias_prefix_when_model_has_exact_block():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.0", "proj.1"],
                [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
            ),
            "L0.mlp": NamedSerialParameters.from_vector_list(
                ["proj.bias"],
                [torch.tensor([[5.0, 6.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.mlp", ["proj"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "L0.mlp")

    torch.testing.assert_close(symmeters["L0.mlp"].vectors, torch.tensor([[6.0, 5.0]]))
    torch.testing.assert_close(symmeters["model"].vectors, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


def test_apply_square_matrix_rejects_missing_symmetry():
    symmeters = Symmeters.from_symmetry_dict({"model": NamedSerialParameters()})

    with pytest.raises(ValueError, match="Symmetry missing not found"):
        symmeters.apply_square_matrix(torch.eye(2), "missing")


def test_apply_square_matrix_is_noop_for_empty_symmetries():
    symmeters = Symmeters.from_symmetry_dict({"L0.H0.qk": NamedSerialParameters()})

    returned = symmeters.apply_square_matrix(torch.eye(2), "L0.H0.qk")

    assert returned is symmeters
    assert symmeters["L0.H0.qk"].vectors is None


def test_apply_square_matrix_updates_symmetry_without_model_equivalence_metadata():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "L0.H0.qk": NamedSerialParameters.from_vector_list(
                ["bias.0"],
                [torch.tensor([[2.0, 3.0]])],
            )
        }
    )

    symmeters.apply_square_matrix(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), "L0.H0.qk")

    torch.testing.assert_close(symmeters["L0.H0.qk"].vectors, torch.tensor([[3.0, 2.0]]))


def test_apply_square_matrix_updates_each_equivalent_prefix_block_independently():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["query.head.0.0", "query.head.0.1", "key.head.0.0", "key.head.0.1"],
                [torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(
                ["query.bias", "key.bias"],
                [torch.tensor([[5.0, 6.0], [7.0, 8.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["query.head.0", "key.head.0"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "L0.H0.qk")

    torch.testing.assert_close(symmeters["L0.H0.qk"].vectors, torch.tensor([[6.0, 5.0], [8.0, 7.0]]))
    torch.testing.assert_close(
        symmeters["model"].vectors,
        torch.tensor([[2.0, 20.0], [1.0, 10.0], [4.0, 40.0], [3.0, 30.0]]),
    )


def test_apply_square_matrix_preserves_gradients_to_matrix():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.head.0.0", "proj.head.0.1"],
                [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(
                ["bias.0"],
                [torch.tensor([[5.0, 6.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["proj.head.0"])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    symmeters.apply_square_matrix(matrix, "L0.H0.qk")
    loss = symmeters["L0.H0.qk"].vectors.square().sum() + symmeters["model"].vectors.square().sum()
    loss.backward()

    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()
    assert matrix.grad.abs().sum() > 0


def test_apply_square_matrix_ignores_equivalence_prefixes_without_matching_rows():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["other.0"],
                [torch.tensor([[5.0, 6.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(
                ["bias.0"],
                [torch.tensor([[2.0, 3.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["missing.prefix"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "L0.H0.qk")

    torch.testing.assert_close(symmeters["L0.H0.qk"].vectors, torch.tensor([[3.0, 2.0]]))
    torch.testing.assert_close(symmeters["model"].vectors, torch.tensor([[5.0, 6.0]]))


def test_apply_square_matrix_rejects_prefixes_with_unexpected_row_counts():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.head.0.0", "proj.head.0.1", "proj.head.0.2"],
                [torch.tensor([[1.0], [2.0], [3.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(
                ["bias.0"],
                [torch.tensor([[2.0, 3.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["proj.head.0"])

    with pytest.raises(ValueError, match="matched 3 rows in symmetry model, expected 2"):
        symmeters.apply_square_matrix(torch.eye(2), "L0.H0.qk")


def test_apply_square_matrix_applies_model_symmetry_equivalence_rows_on_the_other_side():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.0", "proj.1", "other.bias"],
                [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
            )
        }
    )
    symmeters.set_equivalence_class("model", ["proj"])
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    symmeters.apply_square_matrix(matrix, "model")

    torch.testing.assert_close(symmeters["model"].vectors, torch.tensor([[4.0, 3.0], [2.0, 1.0], [6.0, 5.0]]))


def test_apply_attention_head_matrix_rejects_missing_layers_and_wrong_shapes():
    symmeters = Symmeters.from_symmetry_dict({"model": NamedSerialParameters.from_vector_list(["row.0"], [torch.tensor([[1.0, 2.0]])])})

    with pytest.raises(ValueError, match="No attention heads found"):
        symmeters.apply_attention_head_matrix(torch.eye(1), layer_idx=0)

    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                [
                    "attn.self.query.weight.head.0.0",
                    "attn.self.key.weight.head.0.0",
                    "attn.self.value.weight.head.0.0",
                    "attn.output.dense.weight.head.0.0",
                ],
                [torch.tensor([[1.0], [2.0], [3.0], [4.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(["q.bias"], [torch.tensor([[5.0]])]),
            "L0.H0.ov": NamedSerialParameters.from_vector_list(["v.bias"], [torch.tensor([[6.0]])]),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["attn.self.query.weight.head.0", "attn.self.key.weight.head.0"])
    symmeters.set_equivalence_class("L0.H0.ov", ["attn.self.value.weight.head.0", "attn.output.dense.weight.head.0"])

    with pytest.raises(ValueError, match=r"Expected a \(1, 1\) matrix"):
        symmeters.apply_attention_head_matrix(torch.eye(2), layer_idx=0)


def test_apply_attention_head_matrix_updates_model_and_auxiliary_symmetries():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                [
                    "attn.self.query.weight.head.0.0",
                    "attn.self.key.weight.head.0.0",
                    "attn.self.value.weight.head.0.0",
                    "attn.output.dense.weight.head.0.0",
                    "attn.self.query.weight.head.1.0",
                    "attn.self.key.weight.head.1.0",
                    "attn.self.value.weight.head.1.0",
                    "attn.output.dense.weight.head.1.0",
                ],
                [torch.tensor([[1.0], [2.0], [3.0], [4.0], [11.0], [12.0], [13.0], [14.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(["qk.0"], [torch.tensor([[5.0]])]),
            "L0.H0.ov": NamedSerialParameters.from_vector_list(["ov.0"], [torch.tensor([[6.0]])]),
            "L0.H1.qk": NamedSerialParameters.from_vector_list(["qk.1"], [torch.tensor([[15.0]])]),
            "L0.H1.ov": NamedSerialParameters.from_vector_list(["ov.1"], [torch.tensor([[16.0]])]),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["attn.self.query.weight.head.0", "attn.self.key.weight.head.0"])
    symmeters.set_equivalence_class("L0.H0.ov", ["attn.self.value.weight.head.0", "attn.output.dense.weight.head.0"])
    symmeters.set_equivalence_class("L0.H1.qk", ["attn.self.query.weight.head.1", "attn.self.key.weight.head.1"])
    symmeters.set_equivalence_class("L0.H1.ov", ["attn.self.value.weight.head.1", "attn.output.dense.weight.head.1"])

    symmeters.apply_attention_head_matrix(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), layer_idx=0)

    torch.testing.assert_close(
        symmeters["model"].vectors,
        torch.tensor([[11.0], [12.0], [13.0], [14.0], [1.0], [2.0], [3.0], [4.0]]),
    )
    torch.testing.assert_close(symmeters["L0.H0.qk"].vectors, torch.tensor([[15.0]]))
    torch.testing.assert_close(symmeters["L0.H0.ov"].vectors, torch.tensor([[16.0]]))
    torch.testing.assert_close(symmeters["L0.H1.qk"].vectors, torch.tensor([[5.0]]))
    torch.testing.assert_close(symmeters["L0.H1.ov"].vectors, torch.tensor([[6.0]]))


def test_apply_attention_head_matrix_preserves_gradients_to_matrix():
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                [
                    "attn.self.query.weight.head.0.0",
                    "attn.self.key.weight.head.0.0",
                    "attn.self.value.weight.head.0.0",
                    "attn.output.dense.weight.head.0.0",
                    "attn.self.query.weight.head.1.0",
                    "attn.self.key.weight.head.1.0",
                    "attn.self.value.weight.head.1.0",
                    "attn.output.dense.weight.head.1.0",
                ],
                [torch.tensor([[1.0], [2.0], [3.0], [4.0], [11.0], [12.0], [13.0], [14.0]])],
            ),
            "L0.H0.qk": NamedSerialParameters.from_vector_list(["qk.0"], [torch.tensor([[5.0]])]),
            "L0.H0.ov": NamedSerialParameters.from_vector_list(["ov.0"], [torch.tensor([[6.0]])]),
            "L0.H1.qk": NamedSerialParameters.from_vector_list(["qk.1"], [torch.tensor([[15.0]])]),
            "L0.H1.ov": NamedSerialParameters.from_vector_list(["ov.1"], [torch.tensor([[16.0]])]),
        }
    )
    symmeters.set_equivalence_class("L0.H0.qk", ["attn.self.query.weight.head.0", "attn.self.key.weight.head.0"])
    symmeters.set_equivalence_class("L0.H0.ov", ["attn.self.value.weight.head.0", "attn.output.dense.weight.head.0"])
    symmeters.set_equivalence_class("L0.H1.qk", ["attn.self.query.weight.head.1", "attn.self.key.weight.head.1"])
    symmeters.set_equivalence_class("L0.H1.ov", ["attn.self.value.weight.head.1", "attn.output.dense.weight.head.1"])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    symmeters.apply_attention_head_matrix(matrix, layer_idx=0)
    loss = (
        symmeters["model"].vectors.square().sum()
        + symmeters["L0.H0.qk"].vectors.square().sum()
        + symmeters["L0.H0.ov"].vectors.square().sum()
        + symmeters["L0.H1.qk"].vectors.square().sum()
        + symmeters["L0.H1.ov"].vectors.square().sum()
    )
    loss.backward()

    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()
    assert matrix.grad.abs().sum() > 0


def test_unsupported_additions_and_invalid_value_types_are_rejected():
    assert Symmeters([]).__add__(object()) is NotImplemented
    assert NamedSerialParameters().__add__(object()) is NotImplemented

    symmeters = Symmeters([])
    with pytest.raises(ValueError, match="instance of NamedSerialParameters"):
        symmeters["model"] = torch.tensor([[1.0, 2.0]])


def test_save_and_load_round_trip(tmp_path):
    params = NamedSerialParameters.from_vector_list(
        ["row.0", "row.1"],
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
    )
    path = tmp_path / "params.pt"

    params.save(path)
    loaded = NamedSerialParameters.load(path)

    assert loaded.names == ["row.0", "row.1"]
    torch.testing.assert_close(loaded.vectors, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_symmeters_save_and_load_round_trip_preserves_names_and_equivalence_classes(tmp_path):
    symmeters = Symmeters.from_symmetry_dict(
        {
            "model": NamedSerialParameters.from_vector_list(
                ["proj.0", "proj.1"],
                [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
            ),
            "L0.mlp": NamedSerialParameters.from_vector_list(
                ["bias.bias"],
                [torch.tensor([[5.0, 6.0]])],
            ),
        }
    )
    symmeters.set_equivalence_class("L0.mlp", ["proj"])
    path = tmp_path / "symmeters.pt"

    symmeters.save(path)
    loaded = Symmeters.load(path)

    assert loaded["model"].names == ["proj.0", "proj.1"]
    assert loaded["L0.mlp"].names == ["bias.bias"]
    assert loaded.get_equivalence_class("L0.mlp") == {"proj"}
    torch.testing.assert_close(loaded["model"].vectors, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(loaded["L0.mlp"].vectors, torch.tensor([[5.0, 6.0]]))


def test_filter_can_return_empty_and_matrix_multiplication_preserves_names():
    params = NamedSerialParameters.from_vector_list(
        ["row.0"],
        [torch.tensor([[1.0, 2.0]])],
    )

    filtered = params.filter(lambda _: False)
    right_product = params @ torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    left_product = torch.tensor([[4.0]]) @ params

    assert filtered.names == []
    assert filtered.vectors is None
    assert right_product.names == ["row.0"]
    assert left_product.names == ["row.0"]
    torch.testing.assert_close(right_product.vectors, torch.tensor([[2.0, 6.0]]))
    torch.testing.assert_close(left_product.vectors, torch.tensor([[4.0, 8.0]]))


def test_named_serial_parameters_item_access_and_assignment_cover_all_key_types():
    params = NamedSerialParameters.from_vector_list(
        ["row.0", "row.1"],
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
    )

    torch.testing.assert_close(params["row.0"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(params[1], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(params[:1], torch.tensor([[1.0, 2.0]]))

    params["row.0"] = torch.tensor([9.0, 8.0])
    params[1] = torch.tensor([7.0, 6.0])
    params[:1] = torch.tensor([[5.0, 4.0]])

    torch.testing.assert_close(params.vectors, torch.tensor([[5.0, 4.0], [7.0, 6.0]]))

    with pytest.raises(TypeError, match="Key must be a string, integer, or slice"):
        _ = params[1.5]
    with pytest.raises(TypeError, match="Key must be a string, integer, or slice"):
        params[1.5] = torch.tensor([0.0, 0.0])


def test_named_serial_parameters_inplace_matrix_products_update_vectors():
    params = NamedSerialParameters.from_vector_list(
        ["row.0", "row.1"],
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
    )

    right_result = params.inplace_right_matmul(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    torch.testing.assert_close(params.vectors, torch.tensor([[2.0, 4.0], [9.0, 12.0]]))
    assert right_result is params

    left_result = params.inplace_left_matmul(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    torch.testing.assert_close(params.vectors, torch.tensor([[4.0, 2.0], [12.0, 9.0]]))
    assert left_result is params


def test_string_representations_include_symmetry_and_shape_context():
    named = NamedSerialParameters.from_vector_list(
        ["row.0", "row.1"],
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
    )
    symmeters = Symmeters.from_symmetry_dict({"model": named})

    assert "(2, 2)" in str(named)
    assert "row.0" in str(named)
    assert "model" in str(symmeters)
