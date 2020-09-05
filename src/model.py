import torch
import torch.nn as nn
import torch.functional as F

class Model(nn.Module):
    def __init__(self, training_parameters):
        super(Model, self).__init__()
        self.training_parameters = training_parameters
        self.fc_test = nn.Linear(128, 128)

    def forward(self, input_tensor):
        return self.fc_test(input_tensor)
