import torch
from tqdm import tqdm

class Symmeters(dict):
    def __init__(self, symmetry_names=(), equivalence_classes=None):
        super().__init__({symmetry_name: NamedSerialParameters() for symmetry_name in symmetry_names})
        self.equivalence_classes = {
            symmetry_name: set(prefixes)
            for symmetry_name, prefixes in (equivalence_classes or {}).items()
        }
        self._equivalence_index_cache = {}
        self._equivalence_device_index_cache = {}

    @property
    def symmetry_names(self):
        return list(self.keys())
            
    @classmethod
    def from_symmetry_dict(cls, symmetry_dict, equivalence_classes=None):
        instance = cls(symmetry_dict.keys(), equivalence_classes=equivalence_classes)
        for symmetry_name, named_params in symmetry_dict.items():
            instance[symmetry_name] = named_params
        return instance
    
    def __add__(self, other):
        if isinstance(other, Symmeters):
            combined = Symmeters([])
            all_symmetry_names = [*self.keys(), *(name for name in other.keys() if name not in self)]
            for symmetry_name in all_symmetry_names:
                left_params = self.get(symmetry_name, NamedSerialParameters())
                right_params = other.get(symmetry_name, NamedSerialParameters())
                combined[symmetry_name] = left_params + right_params
            for symmetry_name in all_symmetry_names:
                prefixes1 = self.equivalence_classes.get(symmetry_name, set())
                prefixes2 = other.equivalence_classes.get(symmetry_name, set())
                if prefixes1 and prefixes2 and prefixes1 != prefixes2:
                    raise ValueError(f"Mismatched equivalence classes for symmetry {symmetry_name}.")
                if prefixes1 or prefixes2:
                    combined.equivalence_classes[symmetry_name] = set(prefixes1 or prefixes2)
            return combined
        if isinstance(other, NamedSerialParameters):
            return self + Symmeters.from_symmetry_dict({"model": other})
        return NotImplemented
    
    def __setitem__(self, key, value):
        if not isinstance(value, NamedSerialParameters):
            raise ValueError("Value must be an instance of NamedSerialParameters.")
        self._equivalence_index_cache.pop(key, None)
        for cache_key in [cache_key for cache_key in self._equivalence_device_index_cache if cache_key[0] == key]:
            del self._equivalence_device_index_cache[cache_key]
        super().__setitem__(key, value)
        
    
    def __getitem__(self, key) -> 'NamedSerialParameters':
        return super().__getitem__(key)

    def set_equivalence_class(self, symmetry_name, model_row_prefixes):
        self.equivalence_classes[symmetry_name] = set(model_row_prefixes)

    def get_equivalence_class(self, symmetry_name):
        return self.equivalence_classes.get(symmetry_name, set())

    @staticmethod
    def _equivalence_row_index(param_name, prefix):
        matched_prefix, dot, suffix = param_name.rpartition(".")
        if not dot or matched_prefix != prefix or not suffix.isdigit():
            return None
        return int(suffix)

    def _build_equivalence_row_indices(self, symmetry_name):
        matched_rows = {}
        for idx, param_name in enumerate(self[symmetry_name].names):
            prefix, dot, suffix = param_name.rpartition(".")
            if dot and suffix.isdigit():
                matched_rows.setdefault(prefix, []).append((int(suffix), idx))
        self._equivalence_index_cache[symmetry_name] = {
            prefix: torch.tensor([idx for _, idx in sorted(rows)], dtype=torch.long)
            for prefix, rows in matched_rows.items()
        }

    def _equivalence_row_index_tensor(self, symmetry_name, prefix, device):
        if symmetry_name not in self._equivalence_index_cache:
            self._build_equivalence_row_indices(symmetry_name)
        device_cache_key = (symmetry_name, str(device))
        if device_cache_key not in self._equivalence_device_index_cache:
            self._equivalence_device_index_cache[device_cache_key] = {
                prefix: indices.to(device=device)
                for prefix, indices in self._equivalence_index_cache[symmetry_name].items()
            }
        return self._equivalence_device_index_cache[device_cache_key].get(prefix, torch.empty(0, dtype=torch.long, device=device))

    def equivalent_model_rows(self, symmetry_name):
        prefixes = self.get_equivalence_class(symmetry_name)
        if not prefixes or "model" not in self:
            return NamedSerialParameters()
        return self["model"].filter(
            lambda name: any(self._equivalence_row_index(name, prefix) is not None for prefix in prefixes)
        )

    def _update_matching_rows(self, vectors, symmetry_name, prefixes, width, update_rows):
        matching_index_blocks = []
        updated_row_blocks = []
        for prefix in prefixes:
            matching_indices = self._equivalence_row_index_tensor(symmetry_name, prefix, vectors.device)
            if matching_indices.numel() == 0:
                continue
            if matching_indices.numel() != width:
                raise ValueError(
                    f"Equivalence class prefix {prefix} matched {matching_indices.numel()} rows in symmetry {symmetry_name}, "
                    f"expected {width}."
                )
            matching_index_blocks.append(matching_indices)
            updated_row_blocks.append(update_rows(vectors, matching_indices))
        if not matching_index_blocks:
            return vectors
        return torch.index_copy(vectors, 0, torch.cat(matching_index_blocks), torch.cat(updated_row_blocks))

    def apply_square_matrix(self, matrix, symmetry_name):
        if symmetry_name not in self:
            raise ValueError(f"Symmetry {symmetry_name} not found.")

        params = self[symmetry_name]
        if params.vectors is None:
            return self

        matrix = matrix.to(device=params.vectors.device, dtype=params.vectors.dtype)

        prefixes = self.get_equivalence_class(symmetry_name)

        transformed = params.vectors @ matrix
        if symmetry_name == "decoder":
            transformed = self._update_matching_rows(
                transformed,
                symmetry_name,
                prefixes,
                matrix.shape[0],
                lambda _vectors, indices, source=params.vectors: source.index_select(0, indices),
            )
        self[symmetry_name] = params.with_vectors(transformed)

        if symmetry_name == "model" and "decoder" in self:
            decoder = self["decoder"]
            self["decoder"] = decoder.with_vectors(
                self._update_matching_rows(
                    decoder.vectors,
                    "decoder",
                    self.get_equivalence_class("decoder"),
                    matrix.shape[0],
                    lambda vectors, indices: vectors.index_select(0, indices) @ matrix,
                )
            )

        if not prefixes:
            return self

        linked_symmetry_names = [symmetry_name]
        if symmetry_name != "model" and "model" in self:
            linked_symmetry_names.append("model")

        matrix_t = matrix.T

        for linked_symmetry_name in linked_symmetry_names:
            linked_params = self[linked_symmetry_name]
            if linked_params.vectors is None:
                continue
            self[linked_symmetry_name] = linked_params.with_vectors(
                self._update_matching_rows(
                    linked_params.vectors,
                    linked_symmetry_name,
                    prefixes,
                    matrix.shape[0],
                    lambda vectors, indices, matrix_t=matrix_t: matrix_t @ vectors.index_select(0, indices),
                )
            )

        return self

    def apply_attention_head_matrix(self, matrix, layer_idx):
        head_indices = sorted(
            int(symmetry_name.split(".")[1][1:])
            for symmetry_name in self
            if symmetry_name.startswith(f"L{layer_idx}.H") and symmetry_name.endswith(".qk")
        )
        if not head_indices:
            raise ValueError(f"No attention heads found for layer {layer_idx}.")
        if matrix.shape != (len(head_indices), len(head_indices)):
            raise ValueError(f"Expected a ({len(head_indices)}, {len(head_indices)}) matrix for layer {layer_idx} heads.")

        model_vectors = self["model"].vectors
        matrix = matrix.to(device=model_vectors.device, dtype=model_vectors.dtype)
        hidden_size = model_vectors.shape[1]
        heads = []
        specs = []
        for head_idx in head_indices:
            qk_name = f"L{layer_idx}.H{head_idx}.qk"
            ov_name = f"L{layer_idx}.H{head_idx}.ov"
            q_prefixes = self.get_equivalence_class(qk_name)
            ov_prefixes = self.get_equivalence_class(ov_name)
            prefixes = [
                next(prefix for prefix in q_prefixes if ".query." in prefix),
                next(prefix for prefix in q_prefixes if ".key." in prefix),
                next(prefix for prefix in ov_prefixes if ".value." in prefix),
                next(prefix for prefix in ov_prefixes if ".output." in prefix),
            ]
            index_blocks = [
                self._equivalence_row_index_tensor("model", prefix, model_vectors.device)
                for prefix in prefixes
            ]
            heads.append(
                torch.cat(
                    [*(model_vectors.index_select(0, indices).reshape(-1) for indices in index_blocks), self[qk_name].vectors.reshape(-1), self[ov_name].vectors.reshape(-1)]
                )
            )
            specs.append((index_blocks, qk_name, ov_name, self[qk_name].vectors.shape, self[ov_name].vectors.shape))

        permuted = matrix @ torch.stack(heads)
        updated_model_index_blocks = []
        updated_model_row_blocks = []
        updated_aux_vectors = {}
        for row, (index_blocks, qk_name, ov_name, qk_shape, ov_shape) in zip(permuted, specs):
            offset = 0
            for indices in index_blocks:
                block_size = indices.numel() * hidden_size
                updated_model_index_blocks.append(indices)
                updated_model_row_blocks.append(row[offset:offset + block_size].reshape(indices.numel(), hidden_size))
                offset += block_size
            qk_size = qk_shape.numel()
            updated_aux_vectors[qk_name] = row[offset:offset + qk_size].reshape(qk_shape)
            offset += qk_size
            ov_size = ov_shape.numel()
            updated_aux_vectors[ov_name] = row[offset:offset + ov_size].reshape(ov_shape)

        updated_model_vectors = torch.index_copy(
            model_vectors,
            0,
            torch.cat(updated_model_index_blocks),
            torch.cat(updated_model_row_blocks),
        )
        self["model"] = self["model"].with_vectors(updated_model_vectors)
        for symmetry_name, vectors in updated_aux_vectors.items():
            self[symmetry_name] = self[symmetry_name].with_vectors(vectors)

        return self
    
    def _apply_keyed_square_matrix(self, square_matrices, symmetry):
        if symmetry in square_matrices:
            self.apply_square_matrix(square_matrices[symmetry], symmetry)
            return True
        return False

    
    def apply_square_matrices(self, model, square_matrices):
        self._apply_keyed_square_matrix(square_matrices, "model")
        self._apply_keyed_square_matrix(square_matrices, "decoder")
        
        for layer_idx in tqdm(range(model.config.num_hidden_layers), desc="Applying square matrices to layers"):
            for head_idx in range(model.config.num_attention_heads):
                self._apply_keyed_square_matrix(square_matrices, f"L{layer_idx}.H{head_idx}.qk")
            
            for head_idx in range(model.config.num_attention_heads):
                self._apply_keyed_square_matrix(square_matrices, f"L{layer_idx}.H{head_idx}.ov")
            
            symmetry = f"L{layer_idx}.head"
            if symmetry in square_matrices:
                self.apply_attention_head_matrix(
                    square_matrices[symmetry],
                    layer_idx,
                )
            
            self._apply_keyed_square_matrix(square_matrices, f"L{layer_idx}.mlp")
            
    def clone(self):
        cloned = Symmeters.from_symmetry_dict(
            {
                symmetry_name: (
                    NamedSerialParameters()
                    if symmetry_params.vectors is None
                    else NamedSerialParameters.from_vector_list(
                        symmetry_params.names,
                        [symmetry_params.vectors.detach().clone()],
                    )
                )
                for symmetry_name, symmetry_params in self.items()
            },
            equivalence_classes={
                symmetry_name: set(prefixes)
                for symmetry_name, prefixes in self.equivalence_classes.items()
            },
        )
        cloned._equivalence_index_cache = {
            symmetry_name: prefix_map.copy()
            for symmetry_name, prefix_map in self._equivalence_index_cache.items()
        }
        cloned._equivalence_device_index_cache = {
            cache_key: prefix_map.copy()
            for cache_key, prefix_map in self._equivalence_device_index_cache.items()
        }
        return cloned
    
    def __str__(self):
        return str({f"{symmetry_name} {self.equivalence_classes.get(symmetry_name, [])}": str(named_params) for symmetry_name, named_params in self.items()})
    
    def save(self, path):
        torch.save({
            'symmetries': {
                symmetry_name: {
                    'names': named_params.names,
                    'vectors': named_params.vectors,
                }
                for symmetry_name, named_params in self.items()
            },
            'equivalence_classes': self.equivalence_classes
        }, path)
    
    @classmethod
    def load(cls, path):
        data = torch.load(path)
        symmetry_data = data.get('symmetries', data.get('streams', {}))
        instance = cls(symmetry_names=symmetry_data.keys(), equivalence_classes=data.get('equivalence_classes', {}))
        for symmetry_name, payload in symmetry_data.items():
            if isinstance(payload, dict):
                names = payload.get('names', [])
                vectors = payload.get('vectors')
            else:
                names = []
                vectors = payload
            instance[symmetry_name] = NamedSerialParameters() if vectors is None else NamedSerialParameters.from_vector_list(names, [vectors])
            instance[symmetry_name].names = names
        return instance

class NamedSerialParameters:
    
    def __init__(self, names=None, vectors=None):
        assert (names is None and vectors is None) or len(names) == vectors.shape[0], "Names and vectors must have the same length."
        self.names = names if names is not None else []
        self._vectors = vectors
        self._vector_list = [self._vectors] if self._vectors is not None else []
    
    @classmethod
    def from_vector_list(cls, names, vector_list):
        instance = cls()
        instance.names = names
        instance._vector_list = vector_list
        
        return instance

    def with_vectors(self, vectors):
        return NamedSerialParameters.from_vector_list(self.names, [vectors])
    
    def __add__(self, other):
        if isinstance(other, NamedSerialParameters):
            combined_names = self.names + other.names
            combined_vector_list = self._vector_list + other._vector_list
            return NamedSerialParameters.from_vector_list(combined_names, combined_vector_list)
        return NotImplemented

    @property
    def vectors(self):
        if len(self._vector_list) == 0:
            return self._vectors
        elif len(self._vector_list) == 1:
            return self._vector_list[0]
        
        self._vectors = torch.cat(self._vector_list, dim=0)
        self._vector_list = [self._vectors]
        return self._vectors
    
    @property
    def shape(self):
        return self.vectors.shape
    
    def save(self, path):
        torch.save({
            'names': self.names,
            'vectors': self.vectors
        }, path)
        
    @classmethod
    def load(cls, path):
        data = torch.load(path)
        return cls.from_vector_list(data['names'], [data['vectors']])
    
    def filter(self, lambda_fn):
        filtered_names = []
        filtered_vectors = []
        for name, vector in zip(self.names, [] if self.vectors is None else self.vectors):
            if lambda_fn(name):
                filtered_names.append(name)
                filtered_vectors.append(vector)
        if not filtered_vectors:
            return NamedSerialParameters()
        return NamedSerialParameters.from_vector_list(filtered_names, [torch.stack(filtered_vectors)])
    
    def __matmul__(self, matrix):
        new_vector_list = [self.vectors @ matrix]
        return NamedSerialParameters.from_vector_list(self.names, new_vector_list)

    def __rmatmul__(self, matrix):
        new_vector_list = [matrix @ self.vectors]
        return NamedSerialParameters.from_vector_list(self.names, new_vector_list)
    
    def __getitem__(self, key):
        if type(key) is str:
            idx = self.names.index(key)
            return self.vectors[idx]
        elif type(key) is int:
            return self.vectors[key]
        elif type(key) is slice:
            return self.vectors[key]
        else:
            raise TypeError("Key must be a string, integer, or slice.")
    
    def __setitem__(self, key, value):
        if type(key) is str:
            idx = self.names.index(key)
            self.vectors[idx] = value
        elif type(key) is int:
            self.vectors[key] = value
        elif type(key) is slice:
            self.vectors[key] = value
        else:
            raise TypeError("Key must be a string, integer, or slice.")
    
    def inplace_right_matmul(self, matrix):
        self._vectors = matrix @ self.vectors
        self._vector_list = [self._vectors]
        return self

    def inplace_left_matmul(self, matrix):
        self._vectors = self.vectors @ matrix
        self._vector_list = [self._vectors]
        return self
    
    def __str__(self):
        return f"{tuple(self.vectors.shape) if self.vectors is not None else None}: {[name for name in self.names[:5]]}{'... ' if len(self.names) > 5 else ''}{[name for name in self.names[-5:]] if len(self.names) > 5 else ''}"