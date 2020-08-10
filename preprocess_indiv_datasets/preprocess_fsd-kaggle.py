import os
import pandas as pd
from collections import Counter
from pathlib import Path
import csv
import librosa
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle

from config import config as c
from preprocess import process_samp, get_min_samps, combine_intervals

folder_fsdkaggle = r'Z:\research\cough_count\data\raw\FSDKaggle'
folder_meta = os.path.join(folder_fsdkaggle,r'FSDKaggle2018.meta\FSDKaggle2018.meta')
folder = os.path.join(folder_fsdkaggle, r'FSDKaggle2018.audio_train')
folder_coughs = os.path.join(folder,'FSDKaggle2018.audio.train.coughs')

# def copy_files():
#     os.makedirs(folder_coughs, exist_ok=True)
#     path_meta = os.path.join(folder_meta, r'train_post_competition.csv')
#
#     df = pd.read_csv(path_meta)
#     df_cough = df[df.label == 'Cough']
#     cough_files = df_cough.fname.tolist()
#
#     for f in cough_files:
#         path_old = os.path.join(folder,f)
#         path_new = os.path.join(folder_coughs,f)
#         if os.path.exists(path_new):
#             raise ValueError("New file already exists")
#         copyfile(path_old,path_new)
#         print("copied", f)

def preprocess_coughs():

    path_meta = os.path.join(folder_meta, r'train_post_competition_cough_labeled.xlsx')
    df = pd.read_excel(path_meta)
    df_coughs = df[df.cough==1]

    print('Processing', len(df_coughs.index), 'cough files')

    cough_count=0
    for idx in range(len(df_coughs.index)):
        row = df_coughs.iloc[idx]

        # if row.fname not in ['73608822.wav']:#, '71f2b82b.wav', '69175165.wav']:
        #     continue

        f = Path(row.fname).stem+'_clipped.wav' if row.clipped else row.fname
        name_with_domain = 'FSDKaggle_' + Path(f).stem
        path = os.path.join(folder_coughs, f)
        if not(os.path.exists(path)):
            raise ValueError(path, 'doesnt exist')

        #Load audio
        # load the audio
        y, _ = librosa.load(path, sr=c['sr'])

        # Turn multiple channels into 1
        if len(np.shape(y)) > 1:
            y = np.mean(y, axis=1)

        # check_sr_less_than_16khz(y,name_with_domain)

        cough_per_file=0
        intervals = librosa.effects.split(y, top_db=20)
        intervals = combine_intervals(intervals)
        # print(name_with_domain, len(y))
        # print(intervals)
        for i, (start_samps,end_samps) in enumerate(intervals):
            if end_samps-start_samps < .1*c['sr']: #skip if less than 100 ms #HALF_MIN_SAMPS:
                continue

            pickle_dict, pickle_counter, cough_count = process_samp(y, start_samps/c['sr'], end_samps/c['sr'], 'cough',
                                                                    pickle_dict, pickle_counter,
                                                                    name_with_domain, name_with_domain, cough_count)

            cough_per_file+=1
        print('Finished',f,"Coughs in file:", cough_per_file)

    # Save pickle file and counter for this file
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump([pickle_dict, pickle_counter], pkl_file)

    print("Cough cnt:", pickle_counter["cnt"]["cough"], "| Time (s):", pickle_counter["time_sec"]["cough"])
    print("Finished preprocessing on:", pkl_name)

def preprocess_sounds():

    MIN_SAMPS, HALF_MIN_SAMPS, MIN_TIME, HALF_MIN_TIME = get_min_samps()
    folder_pkl = c['folder_pkl']

    path_meta = os.path.join(folder_meta, r'train_post_competition_cough_labeled.xlsx')
    df = pd.read_excel(path_meta)
    sounds_types = sorted(df.label.unique())
    sounds_types.remove("Cough")

    for sound in sounds_types:

        print("Start preprocessing", sound)

        df_sound = df[df.label==sound]

        pkl_name = 'FSDKaggle_sound_{}.pkl'.format(sound)
        pkl_path = os.path.join(folder_pkl, pkl_name)

        if os.path.exists(pkl_path):
            print("Pickle exists for:", pkl_name, "- skipping.")
            continue

        # Create pickle dictionary to store samples
        pickle_dict = {}
        for l in c['label_types']:
            pickle_dict[l] = []

        # Counter for this file
        pickle_counter = {}
        for obj in ["cnt", "time_sec", "skipped_cnt-too_short"]:
            pickle_counter[obj] = Counter()

        print('Processing', len(df_sound.index), sound, 'files')

        sound_count = 0
        for idx in range(len(df_sound.index)):
            row = df_sound.iloc[idx]

            # if row.fname not in ['e334ed2d.wav']:#, '71f2b82b.wav', '69175165.wav']:
            #     continue

            f = row.fname
            name_with_domain = 'FSDKaggle_' + Path(f).stem
            path = os.path.join(folder, f)
            if not (os.path.exists(path)):
                raise ValueError(path, 'doesnt exist')

            # Load audio
            # load the audio
            y, _ = librosa.load(path, sr=c['sr'])

            # Turn multiple channels into 1
            if len(np.shape(y)) > 1:
                y = np.mean(y, axis=1)

            # check_sr_less_than_16khz(y,name_with_domain)

            sound_segs_per_file = 0
            cough_count = 0
            intervals = librosa.effects.split(y, top_db=20)
            intervals = combine_intervals(intervals)
            # print(name_with_domain, len(y))
            # print(intervals)
            for i, (start_samps, end_samps) in enumerate(intervals):
                if end_samps - start_samps < .1*c['sr']: #skip if less than 100 ms #HALF_MIN_SAMPS:
                    continue

                pickle_dict, pickle_counter, cough_count = process_samp(y, start_samps / c['sr'], end_samps / c['sr'],
                                                                        'sound',
                                                                        pickle_dict, pickle_counter,
                                                                        name_with_domain, name_with_domain, cough_count, suff=sound)

                sound_segs_per_file += 1
            print('Finished', f, sound, "segs in file:", sound_segs_per_file)

        print("Sound cnt:", pickle_counter["cnt"]["sound"], "| Time (s):", pickle_counter["time_sec"]["sound"])
        print("Finished preprocessing on:", pkl_name)

        # Save pickle file and counter for this file
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump([pickle_dict, pickle_counter], pkl_file)


if __name__ == '__main__':

    # copy_files()
    preprocess_coughs()
    # add_coughs()
    # preprocess_sounds()
