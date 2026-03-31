import torch

class MultiStreamSerialParameters(dict):
    
    def __init__(self, stream_names=(), equivalence_classes=None):
        super().__init__({stream_name: NamedSerialParameters() for stream_name in stream_names})
        self.equivalence_classes = {
            stream_name: set(prefixes)
            for stream_name, prefixes in (equivalence_classes or {}).items()
        }

    @property
    def stream_names(self):
        return list(self.keys())
            
    @classmethod
    def from_stream_dict(cls, stream_dict, equivalence_classes=None):
        instance = cls(stream_dict.keys(), equivalence_classes=equivalence_classes)
        for stream_name, named_params in stream_dict.items():
            instance[stream_name] = named_params
        return instance
    
    def __add__(self, other):
        if isinstance(other, MultiStreamSerialParameters):
            combined = MultiStreamSerialParameters([])
            all_stream_names = [*self.keys(), *(name for name in other.keys() if name not in self)]
            for stream_name in all_stream_names:
                stream1 = self.get(stream_name, NamedSerialParameters())
                stream2 = other.get(stream_name, NamedSerialParameters())
                combined[stream_name] = stream1 + stream2
            for stream_name in all_stream_names:
                prefixes1 = self.equivalence_classes.get(stream_name, set())
                prefixes2 = other.equivalence_classes.get(stream_name, set())
                if prefixes1 and prefixes2 and prefixes1 != prefixes2:
                    raise ValueError(f"Mismatched equivalence classes for stream {stream_name}.")
                if prefixes1 or prefixes2:
                    combined.equivalence_classes[stream_name] = set(prefixes1 or prefixes2)
            return combined
        if isinstance(other, NamedSerialParameters):
            return self + MultiStreamSerialParameters.from_stream_dict({"model": other})
        return NotImplemented
    
    def __setitem__(self, key, value):
        if not isinstance(value, NamedSerialParameters):
            raise ValueError("Value must be an instance of NamedSerialParameters.")
        super().__setitem__(key, value)
        
    
    def __getitem__(self, key) -> 'NamedSerialParameters':
        return super().__getitem__(key)

    def set_equivalence_class(self, stream_name, model_row_prefixes):
        self.equivalence_classes[stream_name] = set(model_row_prefixes)

    def get_equivalence_class(self, stream_name):
        return self.equivalence_classes.get(stream_name, set())

    @staticmethod
    def _matches_equivalence_prefix(param_name, prefix):
        return param_name == prefix or param_name.startswith(f"{prefix}.")

    def equivalent_model_rows(self, stream_name):
        prefixes = self.get_equivalence_class(stream_name)
        if not prefixes or "model" not in self:
            return NamedSerialParameters()
        return self["model"].filter(
            lambda name: any(self._matches_equivalence_prefix(name, prefix) for prefix in prefixes)
        )

    def apply_square_matrix(self, matrix, stream_name):
        if stream_name not in self:
            raise ValueError(f"Stream {stream_name} not found.")

        self[stream_name] = self[stream_name] @ matrix

        prefixes = self.get_equivalence_class(stream_name)
        if stream_name == "model" or not prefixes or "model" not in self:
            return self

        model_vectors = self["model"].vectors
        for prefix in prefixes:
            matching_indices = [
                idx
                for idx, param_name in enumerate(self["model"].names)
                if self._matches_equivalence_prefix(param_name, prefix)
            ]
            if not matching_indices:
                continue
            if len(matching_indices) != matrix.shape[0]:
                raise ValueError(
                    f"Equivalence class prefix {prefix} matched {len(matching_indices)} model rows, "
                    f"expected {matrix.shape[0]}."
                )
            model_vectors[matching_indices] = matrix.T @ model_vectors[matching_indices]

        return self
    
    def __str__(self):
        return str({f"{stream_name} {self.equivalence_classes.get(stream_name, [])}": str(named_params) for stream_name, named_params in self.items()})

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