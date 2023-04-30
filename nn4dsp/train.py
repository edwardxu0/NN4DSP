# When running this tutorial in Google Colab, install the required packages
# with the following.
# !pip install torchaudio

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


# @title Prepare data and utility functions. {display-mode: "form"}
# @markdown
# @markdown You do not need to look into this cell.
# @markdown Just execute once and you are good to go.

# -------------------------------------------------------------------------------
# Preparation of data and helper functions.
# -------------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
from IPython.display import Audio, display

_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)


def train():
    dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

    print(len(dataset))

    """
    for i in [1, 3, 5]:
        waveform, sample_rate, label = dataset[i]
        plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
        play_audio(waveform, sample_rate)
    """

    waveform, sample_rate, label = dataset[0]

    waveform = waveform.numpy()
    display(
        Audio(
            waveform[0],
            rate=sample_rate,
        ),
        autoplay=True,
    )

    print(waveform.shape)
