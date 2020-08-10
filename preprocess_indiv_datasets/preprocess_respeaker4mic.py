import librosa
import os
import numpy as np
import pickle
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from config import config as c
from config import config_data as c_d
from preprocess import get_min_samps, get_samp, get_spec

mel_basis = librosa.filters.mel(sr=c['sr'], n_fft=c['n_fft'], n_mels=c['nmels'])

def get_framed_spec(y):

    S = get_spec(y)

    S_framed = None
    for i in range(S.shape[1] - 18):
        start = i
        end = i + 19
        S_framed_new = S[:, start:end].transpose()
        S_framed = np.dstack((S_framed, S_framed_new)) if S_framed is not None else np.expand_dims(S_framed_new, -1)
    return (np.expand_dims(np.transpose(S_framed, (2, 0, 1)), -1))

def create_pkl_file_individuals():

    folder_pkl = c['folder_pkl']

    folder_indiv = os.path.join(c['folder_raw'], r'respeaker_4mic\data\individuals')
    subjects = set([f.split('_')[-2] for f in os.listdir(folder_indiv) if f.endswith('.wav')])

    for subject in subjects:

        # if subject in ['noise']:#'matt', 'manuja']:
        #     continue

        print("Starting:", subject)

        path_pkl_dict = os.path.join(folder_pkl, 'respeaker_4mic_individual_{}.pkl'.format(subject))

        #Create empty one
        # if not(os.path.exists(path_pkl_dict)):
        #     with open(path_pkl_dict, 'wb') as file:
        #         pkl_dict = {}
        #         pkl_counter = {'cnt':{},'time_sec':{},'skipped_cnt-too_short':{}}
        #         pickle.dump([pkl_dict, pkl_counter], file)
        #
        # with open(path_pkl_dict, 'rb') as file:
        #     pkl_dict, pkl_counter = pickle.load(file)
        # Create pickle dictionary to store samples
        pkl_dict = {}
        for l in c['label_types']:
            pkl_dict[l] = []

        # Counter for this file
        pkl_counter = {}
        for obj in ["cnt", "time_sec", "skipped_cnt-too_short"]:
            pkl_counter[obj]=Counter()

        ## Silence, speech, laughter
        files = [f for f in os.listdir(folder_indiv) if f.endswith('.wav') and f.split('_')[-2]==subject]

        for typ in ['speech', 'laugh']:
            save_typ = 'silence'  #we're going to label all of the speech+laugh as silence so it shows up more often #'speech' if typ == 'laugh' else typ

            for f in files:
                if typ not in f:
                    continue

                path = os.path.join(folder_indiv, f)

                y, _ = librosa.load(path, sr=c['sr'])
                # Turn multiple channels into 1
                if len(np.shape(y)) > 1:
                    y = np.mean(y, axis=1)

                S = get_spec(y)

                # print(f.split('_')[-2], f.split('_')[-1].split('.')[0], save_typ, "shape:", S.shape)

                pkl_dict[save_typ] += [[S, Path(f).stem, 'respeaker_4mic', 'na']]

                pkl_counter["cnt"][save_typ] += 1
                pkl_counter["time_sec"][save_typ] += len(y)/c['sr']

                with open(path_pkl_dict, 'wb') as file:
                    pickle.dump([pkl_dict, pkl_counter],file)
                print("After", f.split('_')[-2], f.split('_')[-1].split('.')[0], pkl_counter)

        #COUGHS
        labels_path = os.path.join(folder_indiv, 'individuals_labels.txt')
        plot_path = 'plots'
        os.makedirs(plot_path, exist_ok=True)
        for typ in ['cough', 'noise']:
            pkl_dict[typ] = []

            for f in files:
                if typ not in f:
                    continue

                subject = Path(f).stem.split('_')[-2]
                subject = 'background_sounds' if subject == 'noise' else subject

                # print("Starting", f, subject)

                path = os.path.join(folder_indiv, f)

                y, _ = librosa.load(path, sr=c['sr'])
                # Turn multiple channels into 1
                if len(np.shape(y)) > 1:
                    y = np.mean(y, axis=1)

                cnt = 0
                with open(labels_path, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter='\t')

                    labels_start = False

                    for i,row in enumerate(reader):
                        if row[0] == "\\":
                            continue
                        start = float(row[0])
                        end = float(row[1])
                        label = row[2]

                        if label == subject:
                            labels_start = True
                            continue
                        elif not labels_start:
                            continue
                        elif labels_start and label not in ['c', 'noise']:
                            break

                        samp, start, end = get_samp(y, start, end, label, pkl_counter)
                        if len(samp)<1:
                            continue
                        plt.figure()
                        plt.plot(samp)
                        plt.savefig(os.path.join(plot_path,"{}_{}_{}".format(typ,subject, cnt)))
                        # plt.show()
                        plt.close()
                        S = get_spec(samp)

                        pkl_dict[typ] += [[S, Path(f).stem, 'respeaker_4mic', 'na']]
                        # print(typ, "spec shape:", S.shape)
                        cnt+=1

                        pkl_counter["cnt"][typ] += 1
                        pkl_counter["time_sec"][typ] += (end-start)

                    print("After", subject, typ, pkl_counter)

            # print(typ, "total num samples:", len(pkl_dict[typ]))

        # print("Num coughs:", len(pkl_dict['cough']), "Num noise", len(pkl_dict['noise']), "Num silence:", len(pkl_dict['silence']))
        with open(path_pkl_dict, 'wb') as file:
            pickle.dump([pkl_dict, pkl_counter],file)
        print("Final pkl_cnt for:", subject, pkl_counter)

def split_to_quarters(path):
    audio_files = [f for f in os.listdir(path) if f.endswith('algo.wav')]
    audio_files_len_quarters = len(audio_files) // 4

    audio_files_quarters = []
    for i in range(4):
        start = i * audio_files_len_quarters

        if i == 3:
            new_audio_files = audio_files[start:]
        else:
            new_audio_files = audio_files[start:start + audio_files_len_quarters]

        audio_files_quarters.append(new_audio_files)
    return(audio_files_quarters)

def create_pkl_file_other():

    pkl_dict_quarters = {}

    folder_pkl = c['folder_pkl']

    folder_data = os.path.join(c['folder_raw'], r'respeaker_4mic\data')

    folders = ['background_babble', 'conv3_ppl', 'mic_speak', 'sounds']
    # subjects = set([f.split('_')[-2] for f in os.listdir(folder_indiv) if f.endswith('.wav')])

    for rs_folder in folders:

        # if rs_folder in ['background_babble']:
        #     continue

        pkl_dict_quarters[rs_folder]={}

        print("Starting:", rs_folder)

        path_rs_folder = os.path.join(folder_data, rs_folder)
        audio_files_quarters = split_to_quarters(path_rs_folder)

        for i, audio_file_list in enumerate(audio_files_quarters):

            pkl_dict_quarters[rs_folder][i] = []

            pkl_name = 'respeaker_4mic_{}_{}'.format(rs_folder,i)
            path_pkl_dict = os.path.join(folder_pkl, pkl_name + '.pkl')

            pkl_dict = {}
            for l in c['label_types']:
                pkl_dict[l] = []

            # Counter for this file
            pkl_counter = {}
            for obj in ["cnt", "time_sec", "skipped_cnt-too_short"]:
                pkl_counter[obj]=Counter()

            ## Silence, speech, laughter
            typ = 'silence'  #we're going to label all of the speech+laugh as silence so it shows up more often #'speech' if typ == 'laugh' else typ

            for f in audio_file_list:

                pkl_dict_quarters[rs_folder][i] += [f]

                path = os.path.join(path_rs_folder, f)

                y, _ = librosa.load(path, sr=c['sr'])
                # Turn multiple channels into 1
                if len(np.shape(y)) > 1:
                    y = np.mean(y, axis=1)

                S = get_spec(y)

                pkl_dict[typ] += [[S, pkl_name, 'respeaker_4mic', 'na']]

                pkl_counter["cnt"][typ] += 1
                pkl_counter["time_sec"][typ] += len(y)/c['sr']

                with open(path_pkl_dict, 'wb') as file:
                    pickle.dump([pkl_dict, pkl_counter],file)
                print("After", f, pkl_counter)

            # print("Num coughs:", len(pkl_dict['cough']), "Num noise", len(pkl_dict['noise']), "Num silence:", len(pkl_dict['silence']))
            with open(path_pkl_dict, 'wb') as file:
                pickle.dump([pkl_dict, pkl_counter],file)
            print("Final pkl_cnt for:", rs_folder, i, pkl_counter)

    pkl_dict_path = os.path.join(c_d['folder_data'], 'files_CV_dict_rs4mic.pkl')
    if not(os.path.exists(pkl_dict_path)):
        with open(pkl_dict_path, 'wb') as handle:
            pickle.dump(pkl_dict_quarters, handle)
    # print(pkl_dict_quarters)

def add_coughs_to_file():

    folder_data = r'C:\Users\mattw12\Documents\Research\cough_count\data'
    with open(os.path.join(folder_data,'coughs_fsdkaggle_no_rs4mic.pkl'), 'rb') as file:
        coughs = pickle.load(file)

    ## Save Coughs
    rs_files = [f for f in os.listdir(c['folder_pkl']) if 'respeaker_4mic_individual' in f]
    for f in rs_files:
        with open(os.path.join(folder_save,f),'rb') as file:
            pkl_dict_new, pkl_cnt = pickle.load(file)
        samps = pkl_dict_new['cough']
        for i in range(len(samps)):
            samps[i][1] = Path(f).stem
        coughs += 3*samps

    with open(os.path.join(folder_data,'coughs_fsdkaggle_rs4mic.pkl'), 'wb') as file:
        pickle.dump(coughs, file)

def test_load():
    with open(r'C:\Users\mattw12\Documents\Research\cough_count\data\pkl\uwct_8_0_sp.pkl','rb') as file:
        pkl_dict = pickle.load(file)
    print(type(pkl_dict))
    print(len(pkl_dict))
    print(len(pkl_dict[0]),len(pkl_dict[1]))

if __name__ == '__main__':

    create_pkl_file_individuals()
    # create_pkl_file_other()
    # add_coughs_to_file()
    # test_load()