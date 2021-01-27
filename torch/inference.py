import argparse
import logging
import json
import torch
import sys
from pathlib import Path
from model import Model
import numpy as np
from dataset import AudioFile
import soundfile as sf


def parse_arguments_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument("-a", "--audio_file", type=Path, required=True)
    parser.add_argument("-r", "--reference", type=Path, default=None)
    parser.add_argument('-l', "--log_file", type=Path, default=None)
    parser.add_argument("-s", "--save", action='store_true', default=False)
    parser.add_argument("-o", "--out_folder", type=Path, default="./inference/")
    return parser.parse_args()


def set_logger(parameters):
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    parameters.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=parameters.log_file, level=logging.INFO, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)


def load_model_parameters(model_path):
    config_file_path = model_path.parent / (model_path.stem + ".json")
    model_config = argparse.Namespace()
    with open(config_file_path, 'r') as f:
        config_data = json.load(f)
        model_config.__dict__ = config_data['parameters']
    return model_config


class Inference:
    def __init__(self, inference_parameters, training_parameters):
        self.inference_parameters = inference_parameters
        self.training_parameters = training_parameters
        self.save = inference_parameters.save
        self.load_model()

    def load_model(self):
        self.model = Model(self.training_parameters)
        self.load_state_dict()

    def load_state_dict(self):
        with open(self.inference_parameters.model_path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        self.model.load_state_dict(state_dict)

    @staticmethod
    def save_inference_to_audio_file(audio_data, audio_file_path, sample_rate):
        audio_file_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_file_path, audio_data, samplerate=sample_rate)

    def inference_from_audio_file(self, audio_file_path):
        audio_file = AudioFile(audio_file_path, self.training_parameters.frame_length)
        audio_file.zero_pad()
        batch_audio = np.array([frame for frame in audio_file])
        batch_processed_audio = self.inference_from_array(batch_audio)
        processed_audio = np.concatenate(batch_processed_audio)
        if self.save:
            inference_file_path = self.inference_parameters.out_folder / (audio_file_path.stem + "_out.wav")
            self.save_inference_to_audio_file(processed_audio, inference_file_path, audio_file.sample_rate)
            return
        return processed_audio

    def inference_from_array(self, array: np.ndarray):
        tensor = torch.Tensor(array)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1)
        out_tensor = self.model(tensor)
        return out_tensor.squeeze().detach().cpu().numpy()

    def inference_from_audio_list(self, audio_list: list):
        if not self.save:
            return [self.inference_from_audio_file(audio_file) for audio_file in map(Path, audio_list)]
        else:
            for audio_file in map(Path, audio_list):
                self.inference_from_audio_file(audio_file)


if __name__ == "__main__":
    inference_parameters = parse_arguments_from_command_line()
    set_logger(inference_parameters)
    model_parameters = load_model_parameters(inference_parameters.model_path)
    inference = Inference(inference_parameters, model_parameters)
    inference.inference_from_audio_file(inference_parameters.audio_file)
