from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForMaskedLM

from lib.serial_params import NamedSerialParameters
from lib.serial_reader import SerializedParameterReader

class SerialAutoModelForMaskedLM(AutoModelForMaskedLM):
    """Adds serialization and differentiable deserialization helpers to MLM models."""
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load a pretrained MLM and bind the serialization helpers to the concrete instance."""
        # Load the model using the parent class's from_pretrained method
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # AutoModelForMaskedLM resolves to a concrete HF model class, so attach the
        # serialization helpers to the returned instance.
        model.serialize_matrix = MethodType(cls.serialize_matrix, model)
        model.serialize_bias = MethodType(cls.serialize_bias, model)
        model.serialize_layernorm = MethodType(cls.serialize_layernorm, model)
        model.serialize_embeddings = MethodType(cls.serialize_embeddings, model)
        model.serialize_attention = MethodType(cls.serialize_attention, model)
        model.serialize_encoder_layer = MethodType(cls.serialize_encoder_layer, model)
        model.serialize_encoder = MethodType(cls.serialize_encoder, model)
        model.serialize_mlm_head = MethodType(cls.serialize_mlm_head, model)
        model.serialize = MethodType(cls.serialize, model)

        return model
    
    def serialize_matrix(
        self,
        matrix: torch.Tensor,
        name: str = "matrix",
        bias: torch.Tensor | None = None,
        names: list[str] | None = None,
    ) -> NamedSerialParameters:
        """Serialize a matrix as row vectors with an optional inline bias column."""
        # matrix: (row_count, hidden_size)
        assert matrix.shape[1] == self.config.hidden_size, "Matrix must have the same number of columns as the model's hidden size."

        if names is None:
            names = [f"{name}.{i}" for i in range(matrix.shape[0])]
        else:
            assert len(names) == matrix.shape[0], "Names must match the number of matrix rows."
        vector_list = None
        if bias is None:
            # Pad with NaN sentinels to maintain consistent shape for concatenation, but don't include a bias name since there is no bias
            # Serialized rows: (row_count, hidden_size + 1)
            vector_list = [torch.cat([matrix, torch.full((matrix.shape[0], 1), float('nan'), device=matrix.device, dtype=matrix.dtype)], dim=1)]
        else:
            # Concatenate the bias to the matrix
            # bias.unsqueeze(1): (row_count, 1), so serialized rows stay (row_count, hidden_size + 1)
            vector_list = [torch.cat([matrix, bias.unsqueeze(1)], dim=1)]            
        
        return NamedSerialParameters.from_vector_list(names, vector_list)
    
    
    def serialize_bias(self, bias: torch.Tensor, name: str | None = None) -> NamedSerialParameters:
        """Serialize a bias vector as a single padded row."""
        assert bias.shape[0] == self.config.hidden_size, "Bias must have the same size as the model's hidden size."
        names = [f"{name}.bias" if name is not None else "bias"]
        # bias.unsqueeze(0): (1, hidden_size)
        bias = bias.unsqueeze(0)
        # Serialized bias row: (1, hidden_size + 1)
        vector_list = [torch.cat([bias, torch.full((1, 1), float('nan'), device=bias.device, dtype=bias.dtype)], dim=1)]
        return NamedSerialParameters.from_vector_list(names, vector_list)
        

    def serialize_layernorm(self, layernorm: torch.nn.LayerNorm, name: str = "LayerNorm") -> NamedSerialParameters:
        """Serialize a LayerNorm module as separate weight and bias rows."""
        # Each row starts as (1, hidden_size) and is padded to (1, hidden_size + 1)
        weight_row = layernorm.weight.unsqueeze(0)
        bias_row = layernorm.bias.unsqueeze(0)
        weight_row = torch.cat([weight_row, torch.full((1, 1), float('nan'), device=weight_row.device, dtype=weight_row.dtype)], dim=1)
        bias_row = torch.cat([bias_row, torch.full((1, 1), float('nan'), device=bias_row.device, dtype=bias_row.dtype)], dim=1)
        return NamedSerialParameters.from_vector_list([f"{name}.weight", f"{name}.bias"], [weight_row, bias_row])
                
        
    def serialize_embeddings(self, name: str = "embeddings") -> NamedSerialParameters:
        """Serialize the embedding tables and optional embedding LayerNorm."""
        params = self.base_model.embeddings
        
        serialized_params = NamedSerialParameters()
        
        if params.word_embeddings.weight.data_ptr() != self.cls.predictions.decoder.weight.data_ptr():
            # word_embeddings.weight: (vocab_size, hidden_size)
            serialized_params += self.serialize_matrix(params.word_embeddings.weight, name=f"{name}.word_embeddings.weight")
        
        # position/token_type embedding tables: (num_embeddings, hidden_size)
        serialized_params += self.serialize_matrix(params.position_embeddings.weight, name=f"{name}.position_embeddings.weight")
        serialized_params += self.serialize_matrix(params.token_type_embeddings.weight, name=f"{name}.token_type_embeddings.weight")
        
        if params.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(params.LayerNorm, name=f"{name}.LayerNorm")
        
        return serialized_params
    
    def serialize_attention(self, attention: Any, name: str = "attention") -> NamedSerialParameters:
        """Serialize one attention block using the head-indexed row naming scheme."""
        serialized_params = NamedSerialParameters()
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size

        def head_names(prefix: str) -> list[str]:
            return [
                f"{prefix}.head.{head_idx}.{row_idx}"
                for head_idx in range(num_heads)
                for row_idx in range(head_dim)
            ]
        
        # Query, Key, Value matrices and biases
        for matrix_name in ["query", "key", "value"]:
            qkv = getattr(attention.self, matrix_name)
            # qkv.weight: (hidden_size, hidden_size), qkv.bias: (hidden_size,)
            serialized_params += self.serialize_matrix(
                matrix=qkv.weight,
                bias=qkv.bias,
                names=head_names(f"{name}.self.{matrix_name}"),
            )
        
        # Output dense layer and bias
        # dense.weight.T: (hidden_size, hidden_size) so rows line up with the serialized scheme
        serialized_params += self.serialize_matrix(matrix=attention.output.dense.weight.T, names=head_names(f"{name}.output.dense.weight"))
        serialized_params += self.serialize_bias(bias=attention.output.dense.bias, name=f"{name}.output.dense")

        # Output LayerNorm if it exists
        if attention.output.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(attention.output.LayerNorm, name=f"{name}.output.LayerNorm")
        
        return serialized_params
    
    def serialize_encoder_layer(self, layer: Any, name: str = "encoder.layer") -> NamedSerialParameters:
        """Serialize a single encoder layer including attention, MLP, and LayerNorm blocks."""
        serialized_params = NamedSerialParameters()
        
        # Attention parameters
        serialized_params += self.serialize_attention(layer.attention, name=f"{name}.attention")
        
        # Intermediate dense layer and bias
        # intermediate.dense.weight: (intermediate_size, hidden_size)
        serialized_params += self.serialize_matrix(
            matrix=layer.intermediate.dense.weight,
            name=f"{name}.intermediate.dense",
            bias=layer.intermediate.dense.bias,
        )
        
        # Output dense layer and bias
        # output.dense.weight.T: (hidden_size, intermediate_size)
        serialized_params += self.serialize_matrix(matrix=layer.output.dense.weight.T, name=f"{name}.output.dense.weight")
        serialized_params += self.serialize_bias(bias=layer.output.dense.bias, name=f"{name}.output.dense")

        # Output LayerNorm if it exists
        if layer.output.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(layer.output.LayerNorm, name=f"{name}.output.LayerNorm")
        
        return serialized_params
    
    def serialize_encoder(self, name: str = "encoder") -> NamedSerialParameters:
        """Serialize every encoder layer in order."""
        serialized_params = NamedSerialParameters()
        
        for i, layer in enumerate(self.base_model.encoder.layer):
            serialized_params += self.serialize_encoder_layer(layer, name=f"{name}.layer.{i}")
        
        return serialized_params
    
    def serialize_mlm_head(self, name: str = "predictions") -> NamedSerialParameters:
        """Serialize the masked-language-model head."""
        params = self.cls.predictions
        serialized_params = NamedSerialParameters()
        
        # Transform dense layer and bias
        # transform.dense.weight: (hidden_size, hidden_size)
        serialized_params += self.serialize_matrix(
            matrix=params.transform.dense.weight,
            name=f"{name}.transform.dense.weight",
            bias=params.transform.dense.bias
        )
        
        # LayerNorm if it exists
        if params.transform.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(params.transform.LayerNorm, name=f"{name}.transform.LayerNorm")
        
        # Output decoder layer
        # decoder.weight: (vocab_size, hidden_size), bias: (vocab_size,)
        serialized_params += self.serialize_matrix(
            matrix=params.decoder.weight,
            name=f"{name}.decoder.weight",
            bias=params.bias
        )
        
        return serialized_params
    
    def serialize(self) -> NamedSerialParameters:
        """Serialize embeddings, encoder layers, and MLM head into one flat row stream."""
        serialized_params = NamedSerialParameters()
        
        # Serialize embeddings
        serialized_params += self.serialize_embeddings()
        
        # Serialize encoder layers
        serialized_params += self.serialize_encoder()
        
        # Serialize MLM head
        serialized_params += self.serialize_mlm_head()
        
        return serialized_params

    @classmethod
    def _deserialize_embeddings(
        cls,
        model: Any,
        reader: SerializedParameterReader,
        overrides: dict[str, torch.Tensor],
        name: str = "embeddings",
    ) -> bool:
        """Deserialize the embedding block into functional_call parameter overrides."""
        params = model.base_model.embeddings
        base_prefix = model.base_model_prefix
        word_embeddings_prefix = f"{name}.word_embeddings.weight"
        has_word_embeddings = reader.startswith(word_embeddings_prefix)
        if has_word_embeddings:
            # Break the default tied weight so untied serialized embeddings can be overridden independently.
            model.cls.predictions.decoder.weight = torch.nn.Parameter(
                model.cls.predictions.decoder.weight.detach().clone()
            )
            # word_weight: (vocab_size, hidden_size)
            word_weight, word_bias = reader.read_matrix(word_embeddings_prefix, params.word_embeddings.num_embeddings)
            assert word_bias is None, "Embedding tables should not include inline bias."
            overrides[f"{base_prefix}.embeddings.word_embeddings.weight"] = word_weight

        # position/token_type weights: (num_embeddings, hidden_size)
        position_weight, position_bias = reader.read_matrix(
            f"{name}.position_embeddings.weight",
            params.position_embeddings.num_embeddings,
        )
        token_type_weight, token_type_bias = reader.read_matrix(
            f"{name}.token_type_embeddings.weight",
            params.token_type_embeddings.num_embeddings,
        )
        assert position_bias is None and token_type_bias is None, "Embedding tables should not include inline bias."
        overrides[f"{base_prefix}.embeddings.position_embeddings.weight"] = position_weight
        overrides[f"{base_prefix}.embeddings.token_type_embeddings.weight"] = token_type_weight

        # layernorm_rows: (2, hidden_size) -> [0] weight, [1] bias
        layernorm_rows = reader.read_optional_layernorm(f"{name}.LayerNorm")
        if layernorm_rows is None:
            params.LayerNorm = None
        else:
            overrides[f"{base_prefix}.embeddings.LayerNorm.weight"] = layernorm_rows[0]
            overrides[f"{base_prefix}.embeddings.LayerNorm.bias"] = layernorm_rows[1]

        return not has_word_embeddings

    @classmethod
    def _deserialize_attention(
        cls,
        attention: Any,
        reader: SerializedParameterReader,
        overrides: dict[str, torch.Tensor],
        prefix: str,
        name: str = "attention",
    ) -> None:
        """Deserialize an attention block into functional_call parameter overrides."""
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size

        for matrix_name in ["query", "key", "value"]:
            # matrix: (hidden_size, hidden_size), bias: (hidden_size,)
            matrix, bias = reader.read_head_matrix(f"{name}.self.{matrix_name}", num_heads, head_dim)
            assert bias is not None, f"Expected inline bias for {name}.self.{matrix_name}."
            overrides[f"{prefix}.attention.self.{matrix_name}.weight"] = matrix
            overrides[f"{prefix}.attention.self.{matrix_name}.bias"] = bias

        # output_weight is read row-wise as (hidden_size, hidden_size) and transposed back for the module weight.
        output_weight, output_bias = reader.read_head_matrix(f"{name}.output.dense.weight", num_heads, head_dim)
        assert output_bias is None, f"Did not expect inline bias for {name}.output.dense.weight."
        overrides[f"{prefix}.attention.output.dense.weight"] = output_weight.T
        overrides[f"{prefix}.attention.output.dense.bias"] = reader.read_bias(f"{name}.output.dense")

        # layernorm_rows: (2, hidden_size) -> [0] weight, [1] bias
        layernorm_rows = reader.read_optional_layernorm(f"{name}.output.LayerNorm")
        if layernorm_rows is None:
            attention.output.LayerNorm = None
        else:
            overrides[f"{prefix}.attention.output.LayerNorm.weight"] = layernorm_rows[0]
            overrides[f"{prefix}.attention.output.LayerNorm.bias"] = layernorm_rows[1]

    @classmethod
    def _deserialize_encoder_layer(
        cls,
        layer: Any,
        reader: SerializedParameterReader,
        overrides: dict[str, torch.Tensor],
        prefix: str,
        name: str = "encoder.layer",
    ) -> None:
        """Deserialize one encoder layer into functional_call parameter overrides."""
        cls._deserialize_attention(layer.attention, reader, overrides, prefix, name=f"{name}.attention")

        # intermediate_weight: (intermediate_size, hidden_size), intermediate_bias: (intermediate_size,)
        intermediate_weight, intermediate_bias = reader.read_matrix(
            f"{name}.intermediate.dense",
            layer.intermediate.dense.out_features,
        )
        assert intermediate_bias is not None, f"Expected inline bias for {name}.intermediate.dense."
        overrides[f"{prefix}.intermediate.dense.weight"] = intermediate_weight
        overrides[f"{prefix}.intermediate.dense.bias"] = intermediate_bias

        # output_weight is read row-wise as (hidden_size, intermediate_size) and transposed back.
        output_weight, output_bias = reader.read_matrix(
            f"{name}.output.dense.weight",
            layer.output.dense.in_features,
        )
        assert output_bias is None, f"Did not expect inline bias for {name}.output.dense.weight."
        overrides[f"{prefix}.output.dense.weight"] = output_weight.T
        overrides[f"{prefix}.output.dense.bias"] = reader.read_bias(f"{name}.output.dense")

        # layernorm_rows: (2, hidden_size) -> [0] weight, [1] bias
        layernorm_rows = reader.read_optional_layernorm(f"{name}.output.LayerNorm")
        if layernorm_rows is None:
            layer.output.LayerNorm = None
        else:
            overrides[f"{prefix}.output.LayerNorm.weight"] = layernorm_rows[0]
            overrides[f"{prefix}.output.LayerNorm.bias"] = layernorm_rows[1]

    @classmethod
    def _deserialize_encoder(
        cls,
        model: Any,
        reader: SerializedParameterReader,
        overrides: dict[str, torch.Tensor],
        name: str = "encoder",
    ) -> None:
        """Deserialize every encoder layer into functional_call parameter overrides."""
        base_prefix = model.base_model_prefix
        for index, layer in enumerate(model.base_model.encoder.layer):
            cls._deserialize_encoder_layer(
                layer,
                reader,
                overrides,
                f"{base_prefix}.encoder.layer.{index}",
                name=f"{name}.layer.{index}",
            )

    @classmethod
    def _deserialize_mlm_head(
        cls,
        model: Any,
        reader: SerializedParameterReader,
        overrides: dict[str, torch.Tensor],
        name: str = "predictions",
    ) -> None:
        """Deserialize the MLM head into functional_call parameter overrides."""
        params = model.cls.predictions

        # transform_weight: (hidden_size, hidden_size), transform_bias: (hidden_size,)
        transform_weight, transform_bias = reader.read_matrix(
            f"{name}.transform.dense.weight",
            params.transform.dense.out_features,
        )
        assert transform_bias is not None, f"Expected inline bias for {name}.transform.dense.weight."
        overrides["cls.predictions.transform.dense.weight"] = transform_weight
        overrides["cls.predictions.transform.dense.bias"] = transform_bias

        # layernorm_rows: (2, hidden_size) -> [0] weight, [1] bias
        layernorm_rows = reader.read_optional_layernorm(f"{name}.transform.LayerNorm")
        if layernorm_rows is None:
            params.transform.LayerNorm = None
        else:
            overrides["cls.predictions.transform.LayerNorm.weight"] = layernorm_rows[0]
            overrides["cls.predictions.transform.LayerNorm.bias"] = layernorm_rows[1]

        # decoder_weight: (vocab_size, hidden_size), decoder_bias: (vocab_size,)
        decoder_weight, decoder_bias = reader.read_matrix(
            f"{name}.decoder.weight",
            params.decoder.out_features,
        )
        assert decoder_bias is not None, f"Expected inline bias for {name}.decoder.weight."
        overrides["cls.predictions.decoder.weight"] = decoder_weight
        overrides["cls.predictions.bias"] = decoder_bias
    
    @classmethod
    def load_serialized(
        cls,
        serialized_params: NamedSerialParameters,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Build a shell model and differentiable overrides from serialized parameters."""
        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        reader = SerializedParameterReader(serialized_params)
        # overrides maps state_dict names -> tensors with the same shapes as the target parameters.
        overrides = {}

        should_tie_embeddings = cls._deserialize_embeddings(model, reader, overrides)
        cls._deserialize_encoder(model, reader, overrides)
        cls._deserialize_mlm_head(model, reader, overrides)
        if should_tie_embeddings:
            model.cls.predictions.decoder.weight = model.base_model.embeddings.word_embeddings.weight

        reader.assert_done()
        return model, overrides
