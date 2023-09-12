from torch import nn, randn
import torch


class Projector(nn.Module):
    def __init__(self, width, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.proj = nn.Parameter(scale * randn(width, output_dim))
        #self.proj = nn.Parameter(torch.empty(width, output_dim))

    def forward(self, pooled):
        return (pooled @ self.proj).to(torch.float32)
