# Necessary Dependencies

# !pip install praat-textgrids
# !pip install praatio --upgrade
# !pip install pydub
# !pip install youtube_dl
# !pip install SoundFilep

# Imports:
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
from praatio import tgio
import numpy as np
import errno

# Change File Directories:
currentDir = "/Users/aadityasrivathsan/Documents/GitHub/cough-count/"
wavFilesDir = "/Users/aadityasrivathsan/Downloads/FluSense audio/*.wav"
textgridfilesDir = "/Users/aadityasrivathsan/Downloads/flusense_data/*.TextGrid"
pkl_path = "/Users/aadityasrivathsan/Documents/GitHub/cough-count/flusense-coughs.pkl"


wavFiles = sorted(glob.glob(wavFilesDir))
gridFiles = sorted(glob.glob(textgridfilesDir))

pickle_dict = {}
for l in c['label_types']:
    pickle_dict[l] = []

pickle_counter = {}
for obj in ["cnt", "time_sec", "skipped_cnt-too_short"]:
    pickle_counter[obj] = Counter()


for i in range(len(wavFiles)):
  print("starting "+str(i))
  y, _ = librosa.load(wavFiles[i],sr=c['sr'])

  # Turn multiple channels into 1
  if len(np.shape(y)) >1:
    y = np.mean(y, axis=1)
  
  
  grid = tgio.openTextgrid(gridFiles[i])
  a = grid.tierNameList
  wordTier = grid.tierDict[a[0]]
  listOfIntervals = wordTier.entryList
  timeInSec = []
  sublistLabel = []

  for j in range(len(listOfIntervals)):
    data = listOfIntervals[j]
    start = data[0]
    end = data[1]
    label = data[2]
    timeInSec.append([start, end])
    if (label == "sniffle"):
      label = "sniff"
    check = 0
    for l in c['label_types']:
      if (l == label):
        check = 1
    if check == 0:
      label = "sound"
    sublistLabel.append(label)

  for t in range(len(timeInSec)):
    name = wavFiles[i]
    name = name[50:]
    name_with_domain = 'audioset_{}'.format(name)
    sound_segs_per_file = 0
    cough_count = 0
    label = sublistLabel[t]
    if (label == "etc"):
      label = "sound"
    
    time = timeInSec[t]
    start_samps = time[0]
    end_samps = time[1]
    
    pickle_dict, pickle_counter, cough_count = process_samp(y, start_samps, end_samps,
                                                              label, pickle_dict, pickle_counter,
                                                              name_with_domain, name_with_domain, cough_count,
                                                              suff='audioset_{}'.format(name))
    sound_segs_per_file += 1
    
  print("done "+str(i)) 

print("completed")
filename = pkl_path
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise  
with open(pkl_path, 'wb') as pkl_file:
  pickle.dump([pickle_dict, pickle_counter], pkl_file)



