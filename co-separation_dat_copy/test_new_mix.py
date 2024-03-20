from email import utils
from os import path
import os.path
import librosa
from data.base_dataset import BaseDataset
import h5py
import random
from random import randrange
import glob
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
from utils.utils import *
from data.audioVisual_dataset import *
snr = 0
audio_path_1 = '/ext_data2/manhnh/MUSIC_dataset/reshape_11025/MUSIC_solo/top_detections_two_classes/aerJoJj--HU_5.wav'
audio_path_2 = '/ext_data2/manhnh/MUSIC_dataset/reshape_11025/MUSIC_solo/top_detections_two_classes/VlIcqDWmPkw_13.wav'
audio1, audio_rate1 = librosa.load(audio_path_1, sr=11025)
audio_segment1= sample_audio(audio1, 65535)
            # if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
audio_segment1 = augment_audio(audio_segment1, 11025)

audio2, audio_rate2 = librosa.load(audio_path_2, sr=11025)
audio_segment2= sample_audio(audio2, 65535)
            # if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
audio_segment2 = augment_audio(audio_segment2, 11025)

audio_mix = mix_2(audio_segment1, audio_segment2, 'new.wav',snr)
# audio_1, audio_rate_1 = librosa.load(audio_path_1, sr=11025)
# audio_segment_1= sample_audio(audio_1, 65535)
# audio_2, audio_rate_2 = librosa.load(audio_path_2, sr=11025)
# audio_segment_2= sample_audio(audio_2, 65535)

# audio_mix = mix_2(audio_segment_1, audio_segment_2,'new.wav', snr = 0)