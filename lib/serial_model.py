from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForMaskedLM

from lib.serial_params import NamedSerialParameters, MultiStreamSerialParameters
from lib.serial_reader import SerializedParameterOverrides


class SerialAutoModelForMaskedLM(AutoModelForMaskedLM):
    """Adds serialization and differentiable deserialization helpers to MLM models."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "SerialAutoModelForMaskedLM":
        """Load a pretrained MLM and bind the serialization helpers to the concrete instance."""
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        for method_name in [
            "serialize_matrix",
            "serialize_bias",
            "serialize_head_biases",
            "serialize_layernorm",
            "serialize_embeddings",
            "serialize_attention",
            "serialize_encoder_layer",
            "serialize_encoder",
            "serialize_mlm_head",
            "serialize",
        ]:
            setattr(model, method_name, MethodType(getattr(cls, method_name), model))
        return model

    def serialize_matrix(
        self,
        matrix: torch.Tensor,
        name: str = "matrix",
        names: list[str] | None = None,
    ) -> NamedSerialParameters:
        """Serialize a matrix as row vectors."""
        assert matrix.shape[1] == self.config.hidden_size, "Matrix must have the same number of columns as the model's hidden size."
        if names is None:
            names = [f"{name}.{i}" for i in range(matrix.shape[0])]
        return NamedSerialParameters.from_vector_list(names, [matrix])

    def serialize_bias(self, bias: torch.Tensor, name: str | None = None) -> NamedSerialParameters:
        """Serialize a bias vector as a single row."""
        return NamedSerialParameters.from_vector_list(
            [f"{name}.bias" if name is not None else "bias"],
            [bias.unsqueeze(0)],
        )

    def serialize_head_biases(
        self,
        bias: torch.Tensor,
        *,
        num_heads: int,
        head_dim: int,
        name: str,
    ) -> NamedSerialParameters:
        """Serialize a bias vector as one row per attention head."""
        assert bias.shape[0] == num_heads * head_dim, "Bias must split evenly across attention heads."
        return NamedSerialParameters.from_vector_list(
            [f"{name}.head.{head_idx}.bias" for head_idx in range(num_heads)],
            [bias.reshape(num_heads, head_dim)],
        )

    def serialize_layernorm(self, layernorm: torch.nn.LayerNorm, name: str = "LayerNorm") -> NamedSerialParameters:
        """Serialize a LayerNorm module as separate weight and bias rows."""
        return NamedSerialParameters.from_vector_list(
            [f"{name}.weight", f"{name}.bias"],
            [layernorm.weight.unsqueeze(0), layernorm.bias.unsqueeze(0)],
        )

    def serialize_embeddings(self, name: str | None = None) -> NamedSerialParameters:
        """Serialize the embedding tables and optional embedding LayerNorm."""
        if name is None:
            name = f"{self.base_model_prefix}.embeddings"
        params = self.base_model.embeddings
        serialized = NamedSerialParameters()
        serialized += self.serialize_matrix(params.word_embeddings.weight, name=f"{name}.word_embeddings.weight")
        serialized += self.serialize_matrix(params.position_embeddings.weight, name=f"{name}.position_embeddings.weight")
        serialized += self.serialize_matrix(params.token_type_embeddings.weight, name=f"{name}.token_type_embeddings.weight")
        if params.LayerNorm is not None:
            serialized += self.serialize_layernorm(params.LayerNorm, name=f"{name}.LayerNorm")
        return serialized

    def serialize_attention(
        self,
        attention: Any,
        layer_idx: int,
        name: str | None = None,
    ) -> MultiStreamSerialParameters:
        """Serialize one attention block using the head-indexed row naming scheme."""
        if name is None:
            name = f"{self.base_model_prefix}.encoder.layer.{layer_idx}.attention"
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size
        qk_stream_names = [f"L{layer_idx}.H{head_idx}.qk" for head_idx in range(num_heads)]
        ov_stream_names = [f"L{layer_idx}.H{head_idx}.ov" for head_idx in range(num_heads)]
        serialized = MultiStreamSerialParameters(
            [
                "model",
                *qk_stream_names,
                *ov_stream_names,
            ]
        )

        def head_names(prefix: str) -> list[str]:
            return [
                f"{prefix}.head.{head_idx}.{row_idx}"
                for head_idx in range(num_heads)
                for row_idx in range(head_dim)
            ]

        for head_idx in range(num_heads):
            serialized.set_equivalence_class(
                qk_stream_names[head_idx],
                [
                    f"{name}.self.query.weight.head.{head_idx}",
                    f"{name}.self.key.weight.head.{head_idx}",
                ],
            )
            serialized.set_equivalence_class(
                ov_stream_names[head_idx],
                [
                    f"{name}.self.value.weight.head.{head_idx}",
                    f"{name}.output.dense.weight.head.{head_idx}",
                ],
            )

        for matrix_name in ["query", "key", "value"]:
            qkv = getattr(attention.self, matrix_name)
            stream_names = qk_stream_names if matrix_name in {"query", "key"} else ov_stream_names
            serialized["model"] += self.serialize_matrix(
                qkv.weight,
                names=head_names(f"{name}.self.{matrix_name}.weight"),
            )
            head_biases = self.serialize_head_biases(
                qkv.bias,
                num_heads=num_heads,
                head_dim=head_dim,
                name=f"{name}.self.{matrix_name}",
            )
            for head_idx, bias_name in enumerate(head_biases.names):
                serialized[stream_names[head_idx]] += NamedSerialParameters.from_vector_list(
                    [bias_name],
                    [head_biases.vectors[head_idx : head_idx + 1]],
                )

        serialized["model"] += self.serialize_matrix(
            attention.output.dense.weight.T,
            names=head_names(f"{name}.output.dense.weight"),
        )
        serialized["model"] += self.serialize_bias(attention.output.dense.bias, name=f"{name}.output.dense")
        if attention.output.LayerNorm is not None:
            serialized["model"] += self.serialize_layernorm(
                attention.output.LayerNorm,
                name=f"{name}.output.LayerNorm",
            )
        return serialized

    def serialize_encoder_layer(
        self,
        layer: Any,
        layer_idx: int,
        name: str | None = None,
    ) -> MultiStreamSerialParameters:
        """Serialize a single encoder layer including attention, MLP, and LayerNorm blocks."""
        if name is None:
            name = f"{self.base_model_prefix}.encoder.layer.{layer_idx}"
        serialized = self.serialize_attention(layer.attention, layer_idx=layer_idx, name=f"{name}.attention")
        serialized[f"L{layer_idx}.mlp"] = NamedSerialParameters()
        serialized.set_equivalence_class(
            f"L{layer_idx}.mlp",
            [f"{name}.intermediate.dense", f"{name}.output.dense.weight"],
        )
        serialized["model"] += self.serialize_matrix(
            layer.intermediate.dense.weight,
            name=f"{name}.intermediate.dense",
        )
        serialized[f"L{layer_idx}.mlp"] += self.serialize_bias(
            layer.intermediate.dense.bias,
            name=f"{name}.intermediate.dense",
        )
        serialized["model"] += self.serialize_matrix(
            layer.output.dense.weight.T,
            name=f"{name}.output.dense.weight",
        )
        serialized["model"] += self.serialize_bias(layer.output.dense.bias, name=f"{name}.output.dense")
        if layer.output.LayerNorm is not None:
            serialized["model"] += self.serialize_layernorm(layer.output.LayerNorm, name=f"{name}.output.LayerNorm")
        return serialized

    def serialize_encoder(self, name: str | None = None) -> MultiStreamSerialParameters:
        """Serialize every encoder layer in order."""
        if name is None:
            name = f"{self.base_model_prefix}.encoder"
        serialized = MultiStreamSerialParameters()
        for layer_idx, layer in enumerate(self.base_model.encoder.layer):
            serialized += self.serialize_encoder_layer(layer, layer_idx=layer_idx, name=f"{name}.layer.{layer_idx}")
        return serialized

    def serialize_mlm_head(self, name: str = "cls.predictions") -> MultiStreamSerialParameters:
        """Serialize the masked-language-model head."""
        params = self.cls.predictions
        serialized = MultiStreamSerialParameters(["model", "decoder", "vocab"])
        serialized["decoder"] += self.serialize_matrix(
            params.transform.dense.weight,
            name=f"{name}.transform.dense.weight",
        )
        serialized["decoder"] += self.serialize_bias(params.transform.dense.bias, name=f"{name}.transform.dense")
        if params.transform.LayerNorm is not None:
            serialized["decoder"] += self.serialize_layernorm(params.transform.LayerNorm, name=f"{name}.transform.LayerNorm")
        serialized["decoder"] += self.serialize_matrix(params.decoder.weight, name=f"{name}.decoder.weight")
        serialized["vocab"] += self.serialize_bias(params.decoder.bias, name=f"{name}.decoder")
        return serialized

    def serialize(self) -> MultiStreamSerialParameters:
        """Serialize embeddings, encoder layers, and MLM head into one flat row stream."""
        serialized = MultiStreamSerialParameters([])
        serialized += self.serialize_embeddings()
        serialized += self.serialize_encoder()
        serialized += self.serialize_mlm_head()
        if self.cls.predictions.decoder.weight.data_ptr() == self.base_model.embeddings.word_embeddings.weight.data_ptr():
            if "decoder" in serialized.stream_names:
                serialized["model"] += serialized["decoder"].filter(lambda name: "decoder.weight" not in name)
                serialized.set_equivalence_class(
                    "model",
                    [*serialized.get_equivalence_class("model"), "cls.predictions.transform.dense.weight"],
                )
                del serialized["decoder"]
        return serialized

    @classmethod
    def _deserialize_embeddings(
        cls,
        model: Any,
        overrides: SerializedParameterOverrides,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = f"{model.base_model_prefix}.embeddings"
        params = model.base_model.embeddings
        overrides.matrix(f"{name}.word_embeddings.weight", params.word_embeddings.num_embeddings)
        overrides.matrix(f"{name}.position_embeddings.weight", params.position_embeddings.num_embeddings)
        overrides.matrix(f"{name}.token_type_embeddings.weight", params.token_type_embeddings.num_embeddings)
        if overrides.optional_layernorm(f"{name}.LayerNorm") is None:
            params.LayerNorm = None

    @classmethod
    def _deserialize_attention(
        cls,
        attention: Any,
        overrides: SerializedParameterOverrides,
        layer_idx: int,
        name: str,
    ) -> None:
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size
        qk_stream_names = [f"L{layer_idx}.H{head_idx}.qk" for head_idx in range(num_heads)]
        ov_stream_names = [f"L{layer_idx}.H{head_idx}.ov" for head_idx in range(num_heads)]
        for matrix_name, suffix in [("query", "qk"), ("key", "qk"), ("value", "ov")]:
            overrides.head_matrix(
                f"{name}.self.{matrix_name}.weight",
                num_heads=num_heads,
                head_dim=head_dim,
            )
            overrides.head_bias(
                f"{name}.self.{matrix_name}.bias",
                stream_names=qk_stream_names if suffix == "qk" else ov_stream_names,
            )
        overrides.head_matrix(
            f"{name}.output.dense.weight",
            num_heads=num_heads,
            head_dim=head_dim,
            transpose=True,
        )
        overrides.bias(f"{name}.output.dense.bias")
        if overrides.optional_layernorm(f"{name}.output.LayerNorm") is None:
            attention.output.LayerNorm = None

    @classmethod
    def _deserialize_encoder_layer(
        cls,
        layer: Any,
        overrides: SerializedParameterOverrides,
        layer_idx: int,
        name: str,
    ) -> None:
        cls._deserialize_attention(layer.attention, overrides, layer_idx, name=f"{name}.attention")
        overrides.matrix(
            f"{name}.intermediate.dense.weight",
            layer.intermediate.dense.out_features,
            src=f"{name}.intermediate.dense",
        )
        overrides.bias(
            f"{name}.intermediate.dense.bias",
            stream=f"L{layer_idx}.mlp",
            src=f"{name}.intermediate.dense",
        )
        overrides.matrix(
            f"{name}.output.dense.weight",
            layer.output.dense.in_features,
            transpose=True,
        )
        overrides.bias(f"{name}.output.dense.bias")
        if overrides.optional_layernorm(f"{name}.output.LayerNorm") is None:
            layer.output.LayerNorm = None

    @classmethod
    def _deserialize_encoder(
        cls,
        model: Any,
        overrides: SerializedParameterOverrides,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = f"{model.base_model_prefix}.encoder"
        for layer_idx, layer in enumerate(model.base_model.encoder.layer):
            cls._deserialize_encoder_layer(layer, overrides, layer_idx, name=f"{name}.layer.{layer_idx}")

    @classmethod
    def _deserialize_mlm_head(
        cls,
        model: Any,
        overrides: SerializedParameterOverrides,
        tie_embeddings: bool,
        name: str = "cls.predictions",
    ) -> None:
        params = model.cls.predictions
        aux_stream = "model" if tie_embeddings else "decoder"
        overrides.matrix(f"{name}.transform.dense.weight", params.transform.dense.out_features, stream=aux_stream)
        overrides.bias(f"{name}.transform.dense.bias", stream=aux_stream)
        if overrides.optional_layernorm(f"{name}.transform.LayerNorm", stream=aux_stream) is None:
            params.transform.LayerNorm = None
        if not tie_embeddings:
            overrides.matrix(f"{name}.decoder.weight", params.decoder.out_features, stream="decoder")
        overrides.bias(f"{name}.decoder.bias", stream="vocab")

    @classmethod
    def load_serialized(
        cls,
        serialized_params: MultiStreamSerialParameters,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Build a shell model and differentiable overrides from serialized parameters."""
        if isinstance(serialized_params, NamedSerialParameters):
            raise TypeError("load_serialized expects MultiStreamSerialParameters.")
        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        overrides = SerializedParameterOverrides(serialized_params)
        should_tie_embeddings = not overrides.has_prefix("cls.predictions.decoder.weight")
        if not should_tie_embeddings:
            model.cls.predictions.decoder.weight = torch.nn.Parameter(model.cls.predictions.decoder.weight.detach().clone())
        cls._deserialize_embeddings(model, overrides)
        cls._deserialize_encoder(model, overrides)
        cls._deserialize_mlm_head(model, overrides, should_tie_embeddings)
        if should_tie_embeddings:
            model.cls.predictions.decoder.weight = model.base_model.embeddings.word_embeddings.weight
        overrides.assert_done()
        return model, overrides
