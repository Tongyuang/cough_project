# Dependencies:

# !pip install praat-textgrids
# !pip install praatio --upgrade
# !pip install pydub
# !pip install youtube_dl
# !pip install SoundFile
from __future__ import unicode_literals
import glob
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa

import youtube_dl
import soundfile as sf
import os
import pandas as pd
import librosa
import pickle
from collections import Counter
from pathlib import Path
import json
import sys
from config import config as c
from preprocess import process_samp, get_min_samps, combine_intervals
import numpy as np
import errno



# Change File Directories:
currentDir = "/Users/aadityasrivathsan/Documents/GitHub/cough_count"
wavFilesDir = "/Users/aadityasrivathsan/Downloads/cough_wavs/*.wav"
textgridfilesDir = "/Users/aadityasrivathsan/Downloads/cough_labels/*.txt"
pkl_path = "/Users/aadityasrivathsan/Documents/GitHub/cough_count/forced-coughs.pkl"

filename = pkl_path
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

sys.path.append(currentDir)

wavFiles = sorted(glob.glob(wavFilesDir))

textFiles = sorted(glob.glob(textgridfilesDir))

pickle_dict = {}
for l in c['label_types']:
    pickle_dict[l] = []

pickle_counter = {}
for obj in ["cnt", "time_sec", "skipped_cnt-too_short"]:
    pickle_counter[obj] = Counter()

for i in range(len(textFiles)):
  
  print("starting "+str(i))
  wavFile = wavFiles[i]
  textFile = textFiles[i]
  lines = open(textFile,'r').readlines()
  if len(lines) > 0:
    timeInSec = []
    time = []
    for j in range(len(lines)):
      words = lines[j].split('\t')
      time.append([float(words[0])*1000, float(words[1])*1000])
      timeInSec.append([float(words[0]), float(words[1])])

    for t in range(len(timeInSec)):
      # y, _ = sf.read(wavFiles[i], samplerate = c['sr'])
      y, _ = librosa.load(wavFiles[i], sr=c['sr'])

      # Turn multiple channels into 1
      if len(np.shape(y)) >1:
        y = np.mean(y, axis=1)
      
      name = wavFiles[i]
      name = name[50:]
      name_with_domain = 'audioset_{}'.format(name)
      sound_segs_per_file = 0
      cough_count = 0
      # intervals = librosa.effects.split(y, top_db=20)
      # intervals = combine_intervals(intervals)
      label = "cough"
      
      time = timeInSec[t]
      start_samps = time[0]
      end_samps = time[1]
      
      pickle_dict, pickle_counter, cough_count = process_samp(y, start_samps, end_samps,
                                                                label, pickle_dict, pickle_counter,
                                                                name_with_domain, name_with_domain, cough_count,
                                                                suff='audioset_{}'.format(name))

      sound_segs_per_file += 1
  print("done "+str(i))

filename = pkl_path
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


with open(pkl_path, 'wb') as pkl_file:
	pickle.dump([pickle_dict, pickle_counter], pkl_file)

print(pickle_dict.get("cough")[0])

