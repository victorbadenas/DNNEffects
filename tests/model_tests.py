import sys
sys.path.append('./src/')
import unittest
import argparse
import torch
from model import Model



class ModelTests(unittest.TestCase):
    model_params = argparse.Namespace()
    model_params.frame_length = 1024
    model_params.hidden_length = 64

    def reset_model_parameters(self):
        self.model_params.frame_length = 1024
        self.model_params.hidden_length = 64

    def test_model_outoput_size(self):
        self.reset_model_parameters()
        for i in range(1, 10):
            self.model_params.frame_length = 2**i
            input_tensor_shape = (2, self.model_params.frame_length, 1)
            input_tensor = torch.randn(input_tensor_shape)
            model = Model(self.model_params)
            output_tensor = model(input_tensor)
            self.assertEqual(output_tensor.size(), input_tensor_shape)

    def test_hidden_layer_size(self):
        self.reset_model_parameters()
        for i in range(3, 10):
            self.model_params.hidden_length = 2**i
            input_tensor_shape = (2, self.model_params.frame_length, 1)
            input_tensor = torch.randn(input_tensor_shape)
            model = Model(self.model_params)
            output_tensor = model(input_tensor)
            self.assertEqual(output_tensor.size(), input_tensor_shape)

if __name__ == "__main__":
    unittest.main()
