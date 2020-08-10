from preprocessor import visualize

import numpy as np
import matplotlib.pyplot as plt

import pickle
from scipy.io import wavfile

pkl_file = './test_data/label_whosecough_uwct-1-0-bm_0.pkl'
wav_file = './test_data/wav_whosecough_uwct-1-0-bm_0.wav'

with open(pkl_file,'rb') as pf:
    aug = pickle.load(pf).astype("uint8")

_, wav = wavfile.read(wav_file,'w') # sr=16000, wav = list[80000], max=0.71435547

visualize(wav,aug)
plt.show()