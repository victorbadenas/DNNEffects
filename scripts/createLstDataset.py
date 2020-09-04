import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
from xmlutils import XmlDictConfig, XmlListConfig
import xml.etree.ElementTree as ElementTree
import pandas as pd

class LstBuilderFromXML:
    def __init__(self, dataset_root, output_path, effect):
        self.dataset_root = Path(dataset_root)
        self.effect = effect
        self.xml_files = self.search_xml_files()
        self.output_path = Path(output_path)
        self.lst_data = []

    def search_xml_files(self):
        return self.dataset_root.glob(os.path.join("*", "Lists", self.effect, "*.xml"))

    def __call__(self):
        self.build()

    def build(self):
        for xml_file in self.xml_files:
            _, xml_list = self.get_xml_list(xml_file)
            logging.info(f"file has {len(xml_list)} audiofiles")
            subset_root = xml_file.parent.parent.parent
            for audiofile in xml_list:
                try:
                    self.appendAudioFileDataToLst(audiofile, subset_root)
                except Exception as e:
                    audiofile_path = Path(subset_root, "Samples", self.effect, f"{audiofile['fileID']}.wav")
                    logging.error(f"Exception {e} raised loading audiofile {audiofile_path}")
                    logging.error(traceback.format_exc())
        self.save_lst()

    @staticmethod
    def get_xml_list(xml_file):
        logging.info(xml_file)
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        xml_list = XmlListConfig(root)
        return xml_list[0], xml_list[1:]

    def appendAudioFileDataToLst(self, audiofile, subset_root):
        target_audio = Path(subset_root, "Samples", self.effect, f"{audiofile['fileID']}.wav")
        source_audio = self.find_noFx_audio(target_audio, subset_root)
        self.check_existance_of_files(target_audio, source_audio)
        audiofile_dict = self.build_audiofile_dict(audiofile, target_audio, source_audio)
        self.lst_data.append(audiofile_dict)

    def build_audiofile_dict(self, audiofile, target_audio, source_audio):
        audiofile['source'] = source_audio
        audiofile['target'] = target_audio
        return audiofile

    def find_noFx_audio(self, target_audio, subset_root):
        noFxFolder = subset_root / "Samples" / "noFx"
        target_filename = target_audio.stem
        source_filename = '-'.join(target_filename.split('-')[:2]) + "-*.wav"
        source_filenames = list(noFxFolder.glob(source_filename))
        if len(source_filenames) == 0:
            raise OSError("No source files found")
        if len(source_filenames) != 1:
            raise OSError("found multiple possible source files")
        return source_filenames[0]

    @staticmethod
    def check_existance_of_files(*args):
        for file_path in args:
            if not file_path.exists():
                raise OSError(f"file {file_path} does not exist")

    def save_lst(self):
        lst_dataframe = pd.DataFrame.from_dict(self.lst_data)
        self.output_path.mkdir(parents=True, exist_ok=True)
        lst_file_path = self.output_path / f"{self.effect}.lst"
        lst_dataframe.to_csv(lst_file_path, sep="\t")


def set_logger(log_file):
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)

def parse_arguments_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--root_database", type=Path, required=True)
    parser.add_argument("-o", "--output_folder", type=Path, default="./dataset/")
    parser.add_argument("-e", "--effect", type=str, required=True)
    parser.add_argument("--log_file", type=Path, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    parameters = parse_arguments_from_command_line()
    if parameters.log_file is None:
        parameters.log_file = f"./log/{parameters.effect}.log"
    set_logger(Path(parameters.log_file))
    lstbuilder = LstBuilderFromXML(parameters.root_database, parameters.output_folder, parameters.effect)
    lstbuilder()