import os
import sys
import torch
import logging

class Trainer():
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, *args, **kwargs):
        self.train_model(*args, **kwargs)

    def train_model(self):
        pass
