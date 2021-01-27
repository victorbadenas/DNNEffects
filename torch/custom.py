import torch
import torch.nn as nn


def abs(input):
    return torch.abs(input)


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return abs(input)


class PoolInDim(nn.Module):
    def __init__(self, dimension, input_size):
        super().__init__()
        self.dimension = dimension
        pool_factor = int(input_size / 64)
        self.pool = nn.MaxPool1d(pool_factor, return_indices=True)

    def forward(self, input):
        intermediate_shape = list(range(len(input.shape)))
        intermediate_shape[self.dimension] = len(intermediate_shape) - 1
        intermediate_shape[-1] = self.dimension
        input = input.clone()
        input = input.permute(intermediate_shape)
        out, indices = self.pool(input)
        return out.permute(intermediate_shape), indices
