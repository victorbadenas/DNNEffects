import torch
import torch.nn as nn
import torch.functional as F

class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        self.parameters = parameters

    def forward(self, input_tensor):
        return input_tensor
