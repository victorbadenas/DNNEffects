import logging
import argparse
from pathlib import Path
import sys
from trainer import Trainer
import tensorflow as tf

tf.random.set_seed(0)

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
    parser.add_argument("-trainlst", "--train_lst", type=Path, required=True)
    parser.add_argument("-testlst", "--test_lst", type=Path, required=True)
    parser.add_argument("--log_file", type=Path, default="./log/train.log")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--frame_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def show_parameters(parameters):
    logging.info("Training with parameters:")
    for label, value in parameters.__dict__.items():
        logging.info(f"\t{label}: {value}")


if __name__ == "__main__":
    parameters = parse_arguments_from_command_line()
    main(parameters)