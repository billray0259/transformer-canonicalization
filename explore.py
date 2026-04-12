# %% Load MultiBERT seed 0 and serialize it with the current code
from __future__ import annotations

import tempfile

import torch
from transformers import AutoModelForMaskedLM

from lib.serial_model import serialize_model
from lib.serial_params import ParameterComponent, Symmeters

torch.set_printoptions(sci_mode=False, precision=4)

model_name = "google/multiberts-seed_0"
seed0_model = AutoModelForMaskedLM.from_pretrained(model_name)
seed0_model.eval()
seed0_symmeters = serialize_model(seed0_model)

print("Loaded", model_name)
print("symmetry count =", len(seed0_symmeters.symmetry_names))
print("first 12 symmetry names =", seed0_symmeters.symmetry_names[:12])
print("model size =", seed0_symmeters.symmetry_size("model"))
print("L0.qk size =", seed0_symmeters.symmetry_size("L0.qk"))
print("L0.head size =", seed0_symmeters.symmetry_size("L0.head"))
print("L0.mlp size =", seed0_symmeters.symmetry_size("L0.mlp"))

# %%

for symmetry, components in seed0_symmeters.items():
    print(f"{symmetry}")
    for component_name, component in components.items():
        print(f"\t{tuple(component.tensor.shape)}\t{component_name}")



# %% ParameterComponent.from_payload on a real seed-0 component
component_name = "bert.embeddings.word_embeddings.weight"
component = seed0_symmeters.component("model", component_name)
payload = component.to_payload()
restored = ParameterComponent.from_payload(payload)

print("component_name =", component_name)
print("payload keys =", list(payload))
print("restored.axes =", restored.axes)
print("restored.kind =", restored.kind)
print("restored.layout =", restored.layout)
print("restored.parameter_keys =", restored.parameter_keys)
print("restored.axis_indices('model') =", restored.axis_indices("model"))
print("restored.has_axis('vocab_items') =", restored.has_axis("vocab_items"))
print("restored.tensor[:2, :8] =")
print(restored.tensor[:2, :8])


# %% Lookup helpers on the full serialized seed-0 object
print("owned_components('L0.qk') =")
print(list(seed0_symmeters.owned_components("L0.qk")))

query_bias = seed0_symmeters.get_component("bert.encoder.layer.0.attention.self.query.bias")
print("\nget_component('bert.encoder.layer.0.attention.self.query.bias').axes =", query_bias.axes)

head_axis_components = [
	(symmetry_name, component_name)
	for symmetry_name, component_name, _ in seed0_symmeters.components_with_axis("L0.head")[:8]
]
print("\nfirst components_with_axis('L0.head') =", head_axis_components)

print("\nsymmetry_size('model') =", seed0_symmeters.symmetry_size("model"))
print("symmetry_size('L0.qk') =", seed0_symmeters.symmetry_size("L0.qk"))
print("symmetry_size('L0.head') =", seed0_symmeters.symmetry_size("L0.head"))
print("transform_bank_axis('L0.qk') =", seed0_symmeters.transform_bank_axis("L0.qk"))


# %% Merging two Symmeters built from real seed-0 components
left = Symmeters.from_symmetry_dict(
	{
		"model": {
			"bert.embeddings.word_embeddings.weight": seed0_symmeters.component(
				"model",
				"bert.embeddings.word_embeddings.weight",
			),
		}
	}
)

right = Symmeters.from_symmetry_dict(
	{
		"model": {
			"bert.encoder.layer.0.attention.output.dense.bias": seed0_symmeters.component(
				"model",
				"bert.encoder.layer.0.attention.output.dense.bias",
			),
		},
		"L0.mlp": {
			"bert.encoder.layer.0.intermediate.dense.bias": seed0_symmeters.component(
				"L0.mlp",
				"bert.encoder.layer.0.intermediate.dense.bias",
			),
		},
	}
)

combined = left + right

print("left =")
print(left)
print("\nright =")
print(right)
print("\ncombined =")
print(combined)
print("\ncombined.tensor('model', 'bert.encoder.layer.0.attention.output.dense.bias')[:8] =")
print(combined.tensor("model", "bert.encoder.layer.0.attention.output.dense.bias")[:8])


# %% Shared model-axis transform on real seed-0 tensors
model_only = Symmeters.from_symmetry_dict(
	{
		"model": {
			"bert.embeddings.position_embeddings.weight": seed0_symmeters.component(
				"model",
				"bert.embeddings.position_embeddings.weight",
			),
			"cls.predictions.transform.dense.weight": seed0_symmeters.component(
				"model",
				"cls.predictions.transform.dense.weight",
			),
		}
	}
)

model_swap = torch.eye(seed0_symmeters.symmetry_size("model"))
model_swap[[0, 1]] = model_swap[[1, 0]]

print("Before apply_transform('model', model_swap):")
print("position_embeddings[:2, :8] =")
print(model_only.tensor("model", "bert.embeddings.position_embeddings.weight")[:2, :8])
print("\ncls.predictions.transform.dense.weight[:4, :4] =")
print(model_only.tensor("model", "cls.predictions.transform.dense.weight")[:4, :4])
print("\nmodel_swap[:4, :4] =")
print(model_swap[:4, :4])

model_only.apply_transform("model", model_swap)

print("\nAfter apply_transform('model', model_swap):")
print("position_embeddings[:2, :8] =")
print(model_only.tensor("model", "bert.embeddings.position_embeddings.weight")[:2, :8])
print("\ncls.predictions.transform.dense.weight[:4, :4] =")
print(model_only.tensor("model", "cls.predictions.transform.dense.weight")[:4, :4])


# %% Banked transform on the real L0.qk symmetry
qk_only = Symmeters.from_symmetry_dict(
	{
		"L0.qk": {
			"bert.encoder.layer.0.attention.self.query.weight": seed0_symmeters.component(
				"L0.qk",
				"bert.encoder.layer.0.attention.self.query.weight",
			),
			"bert.encoder.layer.0.attention.self.query.bias": seed0_symmeters.component(
				"L0.qk",
				"bert.encoder.layer.0.attention.self.query.bias",
			),
		},
		"L0.head": {},
	}
)

banked_qk = torch.eye(seed0_symmeters.symmetry_size("L0.qk")).repeat(seed0_symmeters.symmetry_size("L0.head"), 1, 1)
banked_qk[0, [0, 1]] = banked_qk[0, [1, 0]]

print("transform_bank_axis('L0.qk') =", qk_only.transform_bank_axis("L0.qk"))
print("Before apply_transform('L0.qk', banked_qk):")
print("query.bias[0, :8] =", qk_only.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.bias")[0, :8])
print("query.weight[0, :2, :8] =")
print(qk_only.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.weight")[0, :2, :8])
print("\nbanked_qk[0, :4, :4] =")
print(banked_qk[0, :4, :4])

qk_only.apply_transform("L0.qk", banked_qk)

print("\nAfter apply_transform('L0.qk', banked_qk):")
print("query.bias[0, :8] =", qk_only.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.bias")[0, :8])
print("query.weight[0, :2, :8] =")
print(qk_only.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.weight")[0, :2, :8])


# %% Head transport on real seed-0 tensors
head_only = Symmeters.from_symmetry_dict(
	{
		"L0.qk": {
			"bert.encoder.layer.0.attention.self.query.bias": seed0_symmeters.component(
				"L0.qk",
				"bert.encoder.layer.0.attention.self.query.bias",
			),
		},
		"L0.ov": {
			"bert.encoder.layer.0.attention.self.value.weight": seed0_symmeters.component(
				"L0.ov",
				"bert.encoder.layer.0.attention.self.value.weight",
			),
		},
		"L0.head": {},
	}
)

head_swap = torch.eye(seed0_symmeters.symmetry_size("L0.head"))
head_swap[[0, 1]] = head_swap[[1, 0]]

print("Before apply_head_transport(0, head_swap):")
print("query.bias[:2, :8] =")
print(head_only.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.bias")[:2, :8])
print("\nvalue.weight[:2, :2, :8] =")
print(head_only.tensor("L0.ov", "bert.encoder.layer.0.attention.self.value.weight")[:2, :2, :8])
print("\nhead_swap[:4, :4] =")
print(head_swap[:4, :4])

head_only.apply_head_transport(0, head_swap)

print("\nAfter apply_head_transport(0, head_swap):")
print("query.bias[:2, :8] =")
print(head_only.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.bias")[:2, :8])
print("\nvalue.weight[:2, :2, :8] =")
print(head_only.tensor("L0.ov", "bert.encoder.layer.0.attention.self.value.weight")[:2, :2, :8])


# %% ordered_transform_names and apply_transforms on real seed-0 tensors
ordered = Symmeters.from_symmetry_dict(
	{
		"model": {
			"bert.encoder.layer.0.attention.output.dense.bias": seed0_symmeters.component(
				"model",
				"bert.encoder.layer.0.attention.output.dense.bias",
			),
		},
		"L0.qk": {
			"bert.encoder.layer.0.attention.self.query.bias": seed0_symmeters.component(
				"L0.qk",
				"bert.encoder.layer.0.attention.self.query.bias",
			),
		},
		"L0.ov": {
			"bert.encoder.layer.0.attention.self.value.weight": seed0_symmeters.component(
				"L0.ov",
				"bert.encoder.layer.0.attention.self.value.weight",
			),
		},
		"L0.head": {},
		"L0.mlp": {
			"bert.encoder.layer.0.intermediate.dense.bias": seed0_symmeters.component(
				"L0.mlp",
				"bert.encoder.layer.0.intermediate.dense.bias",
			),
		},
	}
)

mlp_swap = torch.eye(seed0_symmeters.symmetry_size("L0.mlp"))
mlp_swap[[0, 1]] = mlp_swap[[1, 0]]
transforms = {
	"model": model_swap,
	"L0.mlp": mlp_swap,
	"head": head_swap,
}

print("ordered_transform_names() =", ordered.ordered_transform_names())
print("\nBefore apply_transforms(transforms):")
print("dense.bias[:8] =", ordered.tensor("model", "bert.encoder.layer.0.attention.output.dense.bias")[:8])
print("query.bias[:2, :8] =")
print(ordered.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.bias")[:2, :8])
print("\nvalue.weight[:2, :2, :8] =")
print(ordered.tensor("L0.ov", "bert.encoder.layer.0.attention.self.value.weight")[:2, :2, :8])
print("\nmlp.bias[:8] =", ordered.tensor("L0.mlp", "bert.encoder.layer.0.intermediate.dense.bias")[:8])

ordered.apply_transforms(transforms)

print("\nAfter apply_transforms(transforms):")
print("dense.bias[:8] =", ordered.tensor("model", "bert.encoder.layer.0.attention.output.dense.bias")[:8])
print("query.bias[:2, :8] =")
print(ordered.tensor("L0.qk", "bert.encoder.layer.0.attention.self.query.bias")[:2, :8])
print("\nvalue.weight[:2, :2, :8] =")
print(ordered.tensor("L0.ov", "bert.encoder.layer.0.attention.self.value.weight")[:2, :2, :8])
print("\nmlp.bias[:8] =", ordered.tensor("L0.mlp", "bert.encoder.layer.0.intermediate.dense.bias")[:8])


# %% clone, save, and load on a real seed-0 subset
roundtrip = Symmeters.from_symmetry_dict(
	{
		"model": {
			"bert.embeddings.position_embeddings.weight": seed0_symmeters.component(
				"model",
				"bert.embeddings.position_embeddings.weight",
			),
		},
		"L0.mlp": {
			"bert.encoder.layer.0.intermediate.dense.bias": seed0_symmeters.component(
				"L0.mlp",
				"bert.encoder.layer.0.intermediate.dense.bias",
			),
		},
	}
)

roundtrip_clone = roundtrip.clone()
with torch.no_grad():
	roundtrip_clone["model"]["bert.embeddings.position_embeddings.weight"].tensor[0, 0] = -999.0

print("Original position_embeddings[0, :8] =")
print(roundtrip.tensor("model", "bert.embeddings.position_embeddings.weight")[0, :8])
print("\nCloned position_embeddings[0, :8] after mutation =")
print(roundtrip_clone.tensor("model", "bert.embeddings.position_embeddings.weight")[0, :8])

with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
	roundtrip.save(handle.name)
	loaded = Symmeters.load(handle.name)
	print("\nLoaded position_embeddings[0, :8] =")
	print(loaded.tensor("model", "bert.embeddings.position_embeddings.weight")[0, :8])
	print("\nLoaded mlp.bias[:8] =")
	print(loaded.tensor("L0.mlp", "bert.encoder.layer.0.intermediate.dense.bias")[:8])