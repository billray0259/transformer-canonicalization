import torch

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