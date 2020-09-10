from prompt_toolkit import output
import torch
import torch.nn as nn
from custom import Abs, PoolInDim
from utils import unpoolParameters


# class Model(nn.Module):
#     def __init__(self, training_parameters):
#         super(Model, self).__init__()
#         self.training_parameters = training_parameters
#         self.adaptive_frontend = AdaptiveFrontend(training_parameters)
#         self.ZDNN = ZDNN(training_parameters)
#         self.backend = Backend(training_parameters)

#     def forward(self, input_tensor):
#         Z_tensor, pool_indices, residual_matrix = self.adaptive_frontend(input_tensor)
#         Z_hat_tensor = self.ZDNN(Z_tensor)
#         out_tensor = self.backend(Z_hat_tensor, pool_indices, residual_matrix)
#         return out_tensor


class Model(nn.Module):
    def __init__(self, training_parameters):
        super(Model, self).__init__()
        self.training_parameters = training_parameters
        frame_length = self.training_parameters.frame_length
        hidden_size = self.training_parameters.hidden_length
        hidden_size_half = int(hidden_size / 2)
        self.conv1 = nn.Conv1d(frame_length, frame_length, kernel_size=hidden_size_half, padding=hidden_size_half-1)
        self.dense1 = nn.Linear(in_features=hidden_size_half, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size, out_features=hidden_size_half)
        self.dense4 = nn.Linear(in_features=hidden_size_half, out_features=1)
        self.activation = nn.Tanh()

    def forward(self, input_tensor):
        output_tensor = self.activation(self.conv1(input_tensor))
        output_tensor = self.activation(self.dense1(output_tensor))
        output_tensor = self.activation(self.dense2(output_tensor))
        output_tensor = self.activation(self.dense3(output_tensor))
        output_tensor = self.activation(self.dense4(output_tensor))
        output_tensor = output_tensor + input_tensor
        return output_tensor


class Backend(nn.Module):
    def __init__(self, training_parameters):
        super(Backend, self).__init__()
        self.training_parameters = training_parameters
        unpoolparameters = unpoolParameters(64, self.training_parameters.frame_length)
        self.unpooling = nn.MaxUnpool1d(kernel_size=unpoolparameters['kernel_size'], stride=unpoolparameters['stride'], padding=unpoolparameters['padding'])
        self.dense1 = nn.Linear(in_features=128, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=64) 
        self.dense3 = nn.Linear(in_features=64, out_features=64)
        self.dense4 = nn.Linear(in_features=64, out_features=128)
        self.softplus = nn.Softplus()
        self.ReLU = nn.ReLU()
        self.inverseConv = nn.Linear(in_features=128, out_features=1)

    def forward(self, input_tensor, pool_indices, residual_matrix):
        out_tensor = input_tensor.permute((0, 2, 1))
        out_tensor = self.unpooling(out_tensor, pool_indices)
        out_tensor = out_tensor.permute((0, 2, 1))
        out_tensor = residual_matrix * out_tensor
        out_tensor = self.softplus(self.dense1(out_tensor))
        out_tensor = self.softplus(self.dense2(out_tensor))
        out_tensor = self.softplus(self.dense3(out_tensor))
        out_tensor = self.ReLU(self.dense4(out_tensor))
        out_tensor = self.inverseConv(out_tensor)
        return out_tensor


class ZDNN(nn.Module):
    def __init__(self, training_parameters):
        super(ZDNN, self).__init__()
        self.training_parameters = training_parameters
        self.dense_local = nn.Linear(in_features=128, out_features=128)
        self.dense = nn.Linear(in_features=64, out_features=64)
        self.softplus = nn.Softplus()

    def forward(self, input_tensor):
        out_tensor = self.dense_local(input_tensor)
        out_tensor = out_tensor.permute((0, 2, 1))
        out_tensor = self.dense(out_tensor)
        out_tensor = out_tensor.permute((0, 2, 1))
        return out_tensor


class AdaptiveFrontend(nn.Module):
    def __init__(self, training_parameters):
        super(AdaptiveFrontend, self).__init__()
        self.training_parameters = training_parameters
        self.conv1 = nn.Conv1d(self.training_parameters.frame_length, self.training_parameters.frame_length, kernel_size=128, dilation=1, stride=1, padding=127)
        self.abs = Abs()
        self.conv2 = nn.Conv1d(self.training_parameters.frame_length, self.training_parameters.frame_length, kernel_size=1)
        self.softplus = nn.Softplus()
        self.pool = PoolInDim(dimension=1, input_size=self.training_parameters.frame_length)

    def forward(self, input_tensor):
        residual_matrix = self.conv1(input_tensor)
        abs_x1 = self.abs(residual_matrix)
        x2 = self.softplus(self.conv2(abs_x1))
        poolx2, pool_indices = self.pool(x2)
        return poolx2, pool_indices, residual_matrix


if __name__ == "__main__":
    import argparse
    from torchsummary import summary
    params = argparse.Namespace()
    params.frame_length = 1024
    for i in range(5, 9):
        params.hidden_length = int(2**i)
        model = Model(params)
        summary(model, (params.frame_length, 1), device='cpu')
