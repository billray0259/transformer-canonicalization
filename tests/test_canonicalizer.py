import pytest
import torch

from lib.canonicalizer import Canonicalizer, CascadingTemplateCanonicalizer
from lib.serial_params import Symmeters


def build_test_symmeters():
    symmeters = Symmeters(["model", "L0.qk"])
    symmeters.add_component(
        "model",
        "embeddings.weight",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        axes=("tokens", "model"),
        kind="weight",
        layout="identity",
        parameter_keys="embeddings.weight",
    )
    symmeters.add_component(
        "L0.qk",
        "query.bias",
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        axes=("rows", "L0.qk"),
        kind="bias",
        layout="identity",
        parameter_keys="query.bias",
    )
    return symmeters


def test_forward_accepts_single_symmeters_and_preserves_structure():
    symmeters = build_test_symmeters()
    canonicalizer = Canonicalizer(symmeters, sinkhorn_iters=2)

    canonicalized = canonicalizer(symmeters, tau=1.0)

    assert isinstance(canonicalized, Symmeters)
    assert canonicalized is not symmeters
    assert canonicalized.symmetry_names == symmeters.symmetry_names
    assert canonicalized.tensor("model", "embeddings.weight").shape == symmeters.tensor("model", "embeddings.weight").shape
    assert canonicalized.tensor("L0.qk", "query.bias").shape == symmeters.tensor("L0.qk", "query.bias").shape


def test_forward_rejects_missing_model_symmetry():
    symmeters = Symmeters(["L0.qk"])
    symmeters.add_component(
        "L0.qk",
        "query.bias",
        torch.tensor([[1.0, 2.0]]),
        axes=("rows", "L0.qk"),
        kind="bias",
        layout="identity",
        parameter_keys="query.bias",
    )
    canonicalizer = Canonicalizer(build_test_symmeters(), sinkhorn_iters=2)

    with pytest.raises(ValueError, match="requires a model symmetry"):
        canonicalizer(symmeters)


def test_forward_rejects_missing_active_symmetry_names():
    symmeters = build_test_symmeters()
    canonicalizer = Canonicalizer(symmeters, sinkhorn_iters=2)

    with pytest.raises(ValueError, match="missing active symmetries"):
        canonicalizer(symmeters, active_symmetry_names=("missing",))


def test_forward_rejects_non_symmeters_input():
    canonicalizer = Canonicalizer(build_test_symmeters(), sinkhorn_iters=2)

    with pytest.raises(TypeError, match="expects Symmeters"):
        canonicalizer([])


def test_cascading_template_canonicalizer_save_load_roundtrip(tmp_path):
    pytest.importorskip("scipy")

    symmeters = Symmeters(["model"])
    symmeters.add_component(
        "model",
        "embeddings.weight",
        torch.eye(2),
        axes=("tokens", "model"),
        kind="weight",
        layout="identity",
        parameter_keys="embeddings.weight",
    )

    canonicalizer = CascadingTemplateCanonicalizer(
        order=("model",),
        templates={"model": Canonicalizer._evidence_tensor(symmeters, "model")},
    )
    path = tmp_path / "cascade.pt"
    canonicalizer.save(str(path))

    loaded = CascadingTemplateCanonicalizer.load(str(path))
    canonicalized = loaded(symmeters)

    assert loaded.order == ("model",)
    torch.testing.assert_close(
        canonicalized.tensor("model", "embeddings.weight"),
        symmeters.tensor("model", "embeddings.weight"),
    )