
import os
import time
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio, display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt




_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)


dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)
print(len(dataset))

waveform, sample_rate, label = dataset[0]
waveform = waveform.numpy()
display(Audio(waveform[0],rate=sample_rate,))


rng = np.random.default_rng()

fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = rng.normal(scale=np.sqrt(noise_power),
                   size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

#print(x, min(x), max(x))
plt.plot(x)
plt.show()
f, t, Zxx = signal.stft(x, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot(Zxx)
plt.show()