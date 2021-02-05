import numpy as np
import tensorflow.keras as keras
import pandas as pd
import logging
import math
import soundfile as sf
import numpy as np
from utils import timer


class DataGenerator(keras.utils.Sequence):
    def __init__(self, lst_path, batch_size=32, frame_length=256, shuffle=True):
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.lst_data = self.load_lst(lst_path)
        self.load_audio_data()
        self.compute_frame_list()
        self.shuffle = shuffle
        self.on_epoch_end()

    @staticmethod
    def load_lst(lst_path):
        if not lst_path.exists():
            raise OSError("lst file not found")
        df = pd.read_csv(lst_path, sep='\t')
        return df

    def load_audio_data(self):
        self.source_audio_files, load_time = self.load_audio_files_with_tags(label='source')
        logging.info(f"source audio files have been loaded in {load_time:.2f}s")
        self.target_audio_files, load_time = self.load_audio_files(label='target')
        logging.info(f"target audio files have been loaded in {load_time:.2f}s")

    def compute_frame_list(self):
        self.compute_frame_list_from_audio_files(self.source_audio_files)

    def compute_frame_list_from_audio_files(self, audio_files_array):
        self.frames = []
        silence_threshold = 1e-8
        for audioIdx, audiofile in enumerate(audio_files_array):
            for frameIdx, frame in enumerate(audiofile):
                if np.sum(abs(frame)**2)/len(frame) > silence_threshold:
                    self.frames.append({"audio_idx": audioIdx, "frame_idx": frameIdx})
        logging.info(f"{len(self.frames)} frames have been computed")

    @timer(print_=False)
    def load_audio_files_with_tags(self, label):
        fxsetting_max = self.lst_data['fxsetting'].max()
        audio_files = list()
        for index, source_file_metadata in self.lst_data.iterrows():
            audio_file = AudioFile(source_file_metadata[label], self.frame_length)
            audio_file.normalize()
            effect_params = dict(fxsetting=source_file_metadata['fxsetting']/fxsetting_max)
            audio_file.set_params(effect_params)
            audio_files.append(audio_file)
        return audio_files

    @timer(print_=False)
    def load_audio_files(self, label):
        audio_files = [AudioFile(source_file, self.frame_length) for source_file in self.lst_data[label]]
        for af in audio_files:
            af.normalize()
        return audio_files

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.frames) // self.batch_size

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.frames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        selected_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_frames_indexes = [self.frames[k] for k in selected_indexes]

        X, fxsetting, y = self.__data_generation(batch_frames_indexes)

        return {"frame": X, "fxsetting": fxsetting}, y

    def __data_generation(self, batch_frames_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.frame_length, 1))
        fxsetting = np.empty((self.batch_size, 1))
        y = np.empty((self.batch_size, self.frame_length, 1))

        for i, frame_index in enumerate(batch_frames_indexes):
            audio_idx, frame_idx = frame_index['audio_idx'], frame_index['frame_idx']
            X[i] = self.source_audio_files[audio_idx].get_frame(frame_idx)[:, None]
            fxsetting[i, 0] = self.source_audio_files[audio_idx].fxsetting
            y[i] = self.target_audio_files[audio_idx].get_frame(frame_idx)[:, None]

        return X, fxsetting, y


class AudioFile:
    def __init__(self, audiofile_path, frame_length):
        self.frame_counter = 0
        self.audiofile_path = audiofile_path
        self.frame_length = frame_length
        self.load_audio_data()

    def load_audio_data(self):
        self.audio_data, self.sample_rate = sf.read(self.audiofile_path)
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

    def zero_pad(self):
        remaining = self.audio_length % self.frame_length
        if remaining == 0:
            return
        zero_pad = np.zeros(self.frame_length - remaining)
        self.audio_data = np.concatenate((self.audio_data, zero_pad))
        self.audio_length = len(self.audio_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_counter >= len(self):
            self.frame_counter = 0
            raise StopIteration
        else:
            frame = self.get_frame(self.frame_counter)
            self.frame_counter += 1
            return frame

    def normalize(self):
        self.audio_data /= np.max(np.abs(self.audio_data))

    def set_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)


if __name__ == "__main__":
    from pathlib import Path
    test_dg = DataGenerator(Path('./dataset/Distortion/test.lst'))
    for item in test_dg:
        print(item[0].shape, item[1].shape)
