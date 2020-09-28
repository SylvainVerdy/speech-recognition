"""
Developper: Sylvain Verdy
contact: sylvain.verdy.pro@gmail.com
"""
import hashlib
import os
import json
import re

import librosa as librosa
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import pandas as pd
import numpy as np
import time

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M


def load_data(path):
    """
    :param path: Data Path
    :param type_: If the data path is for train/test/val
    :return: List of data waves.
    """
    time_now = time.time()
    all_wave = []
    all_label = []
    labels = os.listdir(path)

    # find count of each label and plot bar graph
    no_of_recordings = []
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 8300
    top_db = 80
    for label in labels:
        count = 0
        waves = [f for f in os.listdir(path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            if count <= 4:
                samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr=16000)
                samples = librosa.resample(samples, sample_rate, 8000)
                if len(samples) == 8000:
                    spec = librosa.feature.melspectrogram(samples, sr=16000, n_fft=n_fft, hop_length=hop_length,
                                                          n_mels=n_mels, fmin=fmin, fmax=fmax)
                    spec_db = librosa.power_to_db(spec, top_db=top_db)
                    all_wave.append(spec_to_image(spec_db))
                    """
                    # show (128,16) shape to image spectrogram
                    plt.imshow(spec_to_image(spec_db), interpolation='nearest')
                    plt.title("show : " + label)
                    plt.show()
                    """
                    all_label.append(label)
                    count += 1
            else:
                break
    print(time.time() - time_now, "secondes")
    return all_wave, all_label


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def visualize_data(path):
    labels = os.listdir(path)

    # find count of each label and plot bar graph
    no_of_recordings = []
    for label in labels:
        waves = [f for f in os.listdir(path + '/' + label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))
    # plot
    plt.figure(figsize=(30, 5))
    index = np.arange(len(labels))
    plt.bar(index, no_of_recordings)
    plt.xlabel('Commands', fontsize=12)
    plt.ylabel('No of recordings', fontsize=12)
    plt.xticks(index, labels, fontsize=15, rotation=60)
    plt.title('No. of recordings for each command')
    plt.show()

    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


