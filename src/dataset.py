from email.mime import audio
import logging
from numpy.lib.utils import source
from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf
from utils import timer

class LstDataset(Dataset):
    def __init__(self, parameters):
        super(LstDataset, self).__init__()
        self.parameters = parameters
        self.lst_data = self.load_lst(parameters.lst)
        self.load_audio_data()
        self.compute_frame_list()

    @staticmethod
    def load_lst(lst_path):
        if not lst_path.exists():
            raise OSError("lst file not found")
        return pd.read_csv(lst_path, sep='\t')

    def load_audio_data(self):
        self.source_audio_files, load_time = self.load_audio_files(label='source', timer=timer)
        logging.info(f"source audio files have been loaded in {load_time:.2f}s")
        self.target_audio_files, load_time = self.load_audio_files(label='target', timer=timer)
        logging.info(f"target audio files have been loaded in {load_time:.2f}s")

    def compute_frame_list(self):
        self.compute_frame_list_from_audio_files(self.source_audio_files)

    def compute_frame_list_from_audio_files(self, audio_files_array):
        self.frames = []
        for audioIdx, audiofile in enumerate(audio_files_array):
            number_of_frames = len(audiofile)
            for frameIdx in range(number_of_frames):
                self.frames.append({"audio_idx":audioIdx, "frame_idx": frameIdx})
        logging.info(f"{len(self.frames)} have been computed")

    @timer(print_=False)
    def load_audio_files(self, label, timer=None):
        return [AudioFile(source_file, self.parameters.frame_length) for source_file in self.lst_data[label]]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        audio_idx, frame_idx = self.frames[idx]['audio_idx'], self.frames[idx]['frame_idx']
        source_frame = self.source_audio_files[audio_idx].get_frame(frame_idx)
        target_frame = self.target_audio_files[audio_idx].get_frame(frame_idx)
        return source_frame, target_frame


class AudioFile:
    def __init__(self, audiofile_path, frame_length):
        self.frame_counter = 0
        self.audiofile_path = audiofile_path
        self.frame_length = frame_length
        self.load_audio_data()

    def load_audio_data(self):
        self.audio_data, self.sample_rate = sf.read(self.audiofile_path, dtype='float32')
        self.audio_length = len(self.audio_data)

    def __len__(self):
        return int(self.audio_length / self.frame_length)

    def has_next_frame(self):
        return self.frame_counter <= len(self)

    def get_next_frame(self):
        start_index = self.frame_counter * self.frame_length
        end_index = start_index + self.frame_length
        current_frame = self.audio_data[start_index:end_index]
        self.frame_counter += 1
        return current_frame

    def get_frame(self, frame_idx):
        start_index = frame_idx * self.frame_length
        end_index = start_index + self.frame_length
        return self.audio_data[start_index:end_index]
