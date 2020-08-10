import numpy as np
import pandas as pd
import os
import soundfile as sf
import librosa
from pathlib import Path
from scipy.io import wavfile
import progressbar

folder= r'C:\Users\mattw12\Downloads\jotform_test_2'
segments = [f for f in os.listdir(folder) if not f.endswith('clipped')]

for seg in segments:
     seg_new = seg +'_clipped'
     path_seg = os.path.join(folder, seg)
     path_seg_new = os.path.join(folder, seg_new)

     if os.path.exists(path_seg_new):
          continue

     os.makedirs(path_seg_new, exist_ok=True)

     print("starting segment:", seg)

     subjects = os.listdir(path_seg)
     bar = progressbar.ProgressBar(maxval=len(subjects)).start()
     for i, subj in enumerate(subjects):

          path_subj = os.path.join(path_seg, subj)
          path_subj_new = os.path.join(path_seg_new, subj)

          for typ in os.listdir(path_subj):

               path_typ = os.path.join(path_subj, typ)
               path_typ_new = os.path.join(path_subj_new, typ)
               os.makedirs(path_typ_new, exist_ok=True)

               for f in os.listdir(path_typ):

                    path_f = os.path.join(path_typ, f)
                    path_f_new = os.path.join(path_typ_new, Path(f).stem + '_clipped.wav')

                    y, _ = librosa.load(path_f, sr=16000)
                    if len(np.shape(y)) > 1:
                         y = np.mean(y, axis=1)
                    if max(abs(y)) > 1:
                         np.clip(y, -1.0, 1.0, y)

                    y, _ = librosa.effects.trim(y, top_db=20)

                    wavfile.write(path_f_new, 16000, y)

          bar.update(i)

# audio_column_names = ['Single Cough 1',
#      'Single Cough 2', 'Double Cough 1', 'Double Cough 2',
#      'Muffled Single Cough 1', 'Muffled Single Cough 2',
#      'Muffled Double Cough 1', 'Muffled Double Cough 2',
#      'Copy the Cough 1', 'Copy the Cough 2',
#      'Copy the Cough 3', 'Copy the Cough 4',
#      'Copy the Cough 5', 'Copy the Cough 6',
#      'Copy the Cough 7', 'Copy the Cough 8',
#      'Copy the Cough 9', 'Copy the Cough 10',
#      'Copy the Cough 11', 'Copy the Cough 12']
#
# column_names = ['Submission Date', "Parent's email address (we'll email a copy of the consent form)",
#      "Please enter a name to identify this study participant (i.e. the participant's name). If this study participant completes the study again in the future, please make sure to use this same name again.",
#      'What is your age?',
#      'To which gender identity do you most identify?'] + audio_column_names
#
#
# file = r'C:\Users\mattw12\Downloads\Cough-Collection-Study.csv'
# df = pd.read_csv(file,index_col=3)
# df = df.loc[:,column_names]
#
# df_audio=df.loc[:,audio_column_names]
# df_audio_sorted = df_audio.apply(lambda x: sorted([f.split('/')[-1] if not(pd.isna(f)) else 'na' for f in x]), axis=1, result_type='expand')
# # print(df_audio_sorted.head())
# df = pd.concat([df,df_audio_sorted],axis=1)
# print(df.head())
# df.to_csv('df.csv')

# folder = r'C:\Users\mattw12\Downloads\jotform_cough_0_138p'
#
# for root, dirs, files in os.walk(folder, topdown=False):
#      for name in dirs:
#           print(name)