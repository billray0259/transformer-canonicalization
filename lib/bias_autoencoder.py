import torch

class BiasAutoencoder(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # (d_model+1, d_model) [I, 0]
        self.encoder = torch.nn.Parameter(torch.cat([torch.eye(d_model), torch.zeros((1, d_model))], dim=0))
        # (d_model, d_model+1) [I, 0]^T
        self.decoder = torch.nn.Parameter(torch.cat([torch.eye(d_model), torch.zeros((d_model, 1))], dim=1))
    
    def encode(self, x):
        return x @ self.encoder
    
    def decode(self, z):
        return z @ self.decoder
    
    def forward(self, x):
        return self.decode(self.encode(x))