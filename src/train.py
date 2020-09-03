import os
import sys
import logging
import argparse
from pathlib import Path
from trainer import Trainer


def set_logger(parameters):
    level = logging.DEBUG if parameters.debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    parameters.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=parameters.log_file, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)

def main(parameters):
    set_logger(parameters)
    show_parameters(parameters)
    trainer = Trainer(parameters)
    trainer()

def parse_arguments_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--log_file", type=str, default="../log/train.log")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()

def convert_paths_to_pathlib(parameters):
    parameters.log_file = Path(parameters.log_file)
    parameters.checkpoint = Path(parameters.checkpoint) if parameters.checkpoint is not None else None

def show_parameters(parameters):
    logging.info("Training with parameters:")
    for label, value in parameters.__dict__.items():
        logging.info(f"{label}: {value}")

if __name__ == "__main__":
    parameters = parse_arguments_from_command_line()
    convert_paths_to_pathlib(parameters)
    main(parameters)