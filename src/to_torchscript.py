import torch
import json
from pathlib import Path
import argparse
from model import Model
from model_io import ModelIO


class ToLibtorch:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self.convert(*args, **kwargs)

    def convert(self, model_path):
        config_file = model_path.parent / (model_path.stem + ".json")
        saved_name_path = model_path.parent / ("traced_" + model_path.name)
        with open(config_file, 'r') as config_file_handler:
            model_parameters = argparse.Namespace()
            model_parameters.__dict__ = json.load(config_file_handler)['parameters']
        model = Model(model_parameters)
        model_io = ModelIO(None, model)
        model_io.load_model_checkpoint(model_path)
        model.to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(str(saved_name_path))
        print(f"model {str(model_path)} saved to {str(saved_name_path)}")


def parseArgsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=Path, help="path to model.pt file")
    return parser.parse_args()


if __name__ == "__main__":
    parameters = parseArgsFromCommandLine()
    ToLibtorch()(parameters.model)
