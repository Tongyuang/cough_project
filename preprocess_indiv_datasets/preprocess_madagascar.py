import os
import pandas as pd
from collections import Counter
from pathlib import Path
import librosa
import numpy as np
import pickle
import socket

from config import config_data as c_d
from config import config as c
from preprocess import process_samp, get_min_samps, combine_intervals, get_spec
from evaluate import load_trained_model
from shutil import copyfile

folder_data = os.path.join(c_d['folder_data'], 'madagascar') if socket.gethostname() == 'area51.cs.washington.edu' else r'Z:\research\cough_count\data\raw\madagascar'
folder = os.path.join(folder_data, 'ResApp Cough Data')
folder_npy = os.path.join(folder_data, 'npy')
os.makedirs(folder_npy, exist_ok=True)
file_list_path = os.path.join(folder_data, 'wav_fnames.pkl')

def get_file_list():
    file_list = []

    for root, dirs, files in os.walk(folder):#, topdown=False):
        for name in files:
            if name.endswith('.wav'):
                file_list.append(os.path.join(root, name))

    print("Finished file_list of length:", len(file_list))
    with open(file_list_path,'wb') as handle:
        pickle.dump(file_list, handle)

def get_audio_and_pad(path):

    MIN_SAMPS, HALF_MIN_SAMPS, MIN_TIME, HALF_MIN_TIME = get_min_samps()

    # start=time.time()
    y, _ = librosa.load(path, sr=c['sr'])

    # Turn multiple channels into 1
    if len(np.shape(y)) > 1:
        y = np.mean(y, axis=1)

    # Zero-pad Samp
    if len(y) < MIN_SAMPS:
        y = np.pad(y, int(np.ceil(((MIN_SAMPS - len(y)) / 2))))
        y = y[:MIN_SAMPS]  # in case went over by one samp or so

    return(y)

def get_framed_input(S):
    S_framed = None
    f = c['frames']
    HALF_FRAME = int(f/2)
    for k, j in enumerate(np.arange(0, S.shape[1]-HALF_FRAME, HALF_FRAME)):
        end = S.shape[1] if S.shape[1] - j < f else j + f
        start = end - f if S.shape[1] - j < f else j
        S_framed_new = np.expand_dims(S[:, start:end].transpose(),0)
        S_framed = np.concatenate((S_framed,S_framed_new)) if S_framed is not None else S_framed_new
    return(S_framed)


def preprocess():

    with open(file_list_path, 'rb') as handle:
        file_list = pickle.load(handle)
    print("Number of files:", len(file_list))

    for i, path in enumerate(file_list):

        npy_path = os.path.join(folder_npy, Path(os.path.basename(path)).stem + '.npy')
        if os.path.exists(npy_path):
            print(os.path.basename(path), 'exists - skipping')
            continue

        y = get_audio_and_pad(path)
        S = get_spec(y)
        S_framed = get_framed_input(S)
        if S_framed.shape[0]>1:
            print(os.path.basename(path), len(y), S.shape, S_framed.shape)

        np.save(npy_path, S_framed)

        if (i+1) % 100 == 0:
            print("Finished:", i+1, "of", len(file_list))

def add_folder_to_df():
    df = pd.read_csv(os.path.join(folder_data, 'cough_classification.csv'))
    df_new_path = os.path.join(folder_data, 'cough_classification_w_folder.csv')
    if os.path.exists(df_new_path):
        os.remove(df_new_path)
    columns = ['folder1', 'folder2','folder3','folder4','file','cough','times_played']
    df_new = pd.DataFrame([],columns=columns)
    idxs = list(range(len(df.index)))
    cnt=0
    old_cnt=0
    for root, dirs, files in os.walk(folder):
        idx_remove =[]
        for idx in idxs:
            row = df.iloc[idx]
            fname = row.file
            path_wav = Path(fname).stem + '.wav'
            if path_wav in files:
                cough = row.cough
                tp = row.times_played
                folder_location_split = root.split('\\')
                folder_1 = folder_location_split[-4]
                folder_2 = folder_location_split[-3]
                folder_3 = folder_location_split[-2]
                folder_4 = folder_location_split[-1]
                df_new = df_new.append(pd.DataFrame([[folder_1,folder_2,folder_3,folder_4, fname, cough, tp]],columns=columns),ignore_index=True)
                idx_remove.append(idx)
                cnt+=1

        for idx in idx_remove:
            idxs.remove(idx)

        if cnt-old_cnt>100:
            print("Finished", cnt, "of", len(df.index))
            old_cnt=cnt

        # if cnt>5:
        #     break

    df_new.to_csv(df_new_path,index=False)
    # print("Finished:", i+1, "of", len(df.index))

def create_predictions_template():


    columns = ['folder','file','cough','times_played','prediction','confidence']
    df_new = pd.DataFrame([],columns=columns)

    df_labels = pd.read_csv(os.path.join(folder_data, 'cough_classification_w_folder.csv'))

    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith('.wav'):
                cough = 'not_checked'
                times_played = 0
                if name in df_labels.file.unique():
                    row = df_labels[df_labels.file == name]
                    cough = row.cough.values[0]
                    times_played = row.times_played.values[0]
                prediction = cough if cough in ['yes','no'] else ''
                confidence = "na" if cough in ['yes','no'] else ''
                root_small = root[root.find('ResApp Cough Data')+18:]
                df_new = df_new.append(pd.DataFrame([[root_small, name, cough, times_played, prediction, confidence]],columns=columns),
                                       ignore_index=True)

                if (len(df_new.index)+1) % 1000 == 0:
                    print("Finished:", len(df_new.index)+1)
                    # break

        # if (len(df_new.index)+1) % 5 == 0:
        #     print("Finished:", len(df_new.index)+1)
        #     break


    df_new.to_csv(os.path.join(folder_data,'preds_template.csv'),index=False)


def preprocess_for_pkl():

    # Create pickle dictionary to store samples
    pickle_dict = {}
    for l in c['label_types']:
        pickle_dict[l] = []

    # Counter for this file
    pickle_counter = {}
    for obj in ["cnt", "time_sec", "skipped_cnt-too_short"]:
        pickle_counter[obj] = Counter()

    df = pd.read_csv(os.path.join(folder_data, 'cough_classification_w_folder.csv'))

    cough_count = 0
    for idx in range(len(df.index)):
        row = df.iloc[idx]
        fname = row.file
        cough = row.cough
        label = 'cough' if cough =='yes' else 'silence'
        if cough not in ['yes','no']:
            continue
        path = row.folder1 if row.folder1 !='ResApp Cough Data' else None
        path = os.path.join(path, row.folder2) if path is not None else row.folder2
        path = os.path.join(folder, path, row.folder3, row.folder4, fname)
        y = get_audio_and_pad(path)
        pickle_dict, pickle_counter, cough_count = process_samp(y, 0, len(y)/c['sr'], label, pickle_dict, pickle_counter, Path(fname).stem, Path(fname).stem,
                     cough_count)

    with open(os.path.join(c['folder_pkl'], 'madagascar_labeled' + '.pkl'), 'wb') as pkl_file:
        pickle.dump([pickle_dict, pickle_counter], pkl_file)
    print("Time (s):", pickle_counter["time_sec"])
    print("Skipped because too short:", pickle_counter["skipped_cnt-too_short"])

def add_coughs_to_file():

    with open(os.path.join(c_d['folder_data'],'coughs.pkl'), 'rb') as file:
        coughs = pickle.load(file)

    ## Save Coughs
    with open(os.path.join(c['folder_pkl'], 'madagascar_labeled' + '.pkl'), 'rb') as pkl_file:
        pickle_dict, pickle_counter = pickle.load(pkl_file)

    coughs += pickle_dict['cough']

    with open(os.path.join(c_d['folder_data'],'coughs_madagascar.pkl'), 'wb') as file:
        pickle.dump(coughs, file)

def predict():

    vggish=True
    df = pd.read_csv(os.path.join(folder_data, 'preds_template.csv'))

    model, model_name = load_trained_model(vggish=vggish, tflite=False)
    model_name = model_name.split('.')[1]
    df_results_path = os.path.join(folder_data, 'predictions_{}.csv'.format(model_name))

    if os.path.exists(df_results_path):
        os.remove(df_results_path)

    for i in range(len(df.index)):

        if (i+1) % 100 == 0:
            print("Finished:", i+1, "of", len(df.index))

        if (i+1) % 1000 == 0:
            df.to_csv(os.path.join(folder_data, 'predictions_{}_{}.csv'.format(model_name,i)), index=False)


        row = df.loc[i]
        if row.cough in ['yes', 'no']:
            continue

        fname = Path(row.file).stem
        path = os.path.join(folder_npy, fname +'.npy')

        if not(os.path.exists(path)):
            print("Npy doesn't exist - skpping:", fname)
            continue

        S_framed = np.load(path)
        result = np.mean(model.predict(np.expand_dims(S_framed, -1)))

        df.at[i, 'prediction'] = 'yes' if result >=.5 else 'no'
        df.at[i, 'confidence'] = result

    df.to_csv(df_results_path, index=False)

def test():

    vggish=True

    df = pd.read_csv(os.path.join(folder_data, 'cough_classification_w_folder.csv'))
    model, model_name = load_trained_model(vggish=vggish, tflite=False)

    df_results_path = os.path.join(folder_data, 'results.csv')
    if os.path.exists(df_results_path):
        os.remove(df_results_path)

    columns = ['cough_real', 'cough_pred', 'confidence','correct', 'TP-true_pos','TN-true-neg','FP-false_pos', 'FN-missed_cough', 'folder']
    df_results = pd.DataFrame([], columns=columns)

    results_arr = np.array([])

    for i in range(len(df.index)):
        row = df.iloc[i]

        if row.cough not in ['yes','no']:
            continue

        fname = row.file

        f1 = row.folder1
        f2 = row.folder2
        f3 = row.folder3
        f4 = row.folder4
        folder_location = '/'.join([f1,f2,f3,f4])

        y_real = 1 if row.cough == 'yes' else 0

        path = os.path.join(folder_npy, Path(fname).stem +'.npy')

        if not(os.path.exists(path)):
            print("Npy doesn't exist - skpping:", fname)
            continue

        S_framed = np.load(path)
        result = np.mean(model.predict(np.expand_dims(S_framed, -1)))

        y_pred = 1 if result >=.5 else 0
        if y_pred == y_real:
            correct = 1
            TP = 1 if y_pred ==1 else 0
            TN= 0 if y_pred ==1 else 1
            FP = 0
            FN = 0
        else:
            correct = 0
            TP=0
            TN=0
            FP = 1 if y_pred ==1 else 0
            FN = 0 if y_pred ==1 else 1

        results_arr = np.append(results_arr, correct)
        df_results = df_results.append(pd.DataFrame([[y_real, y_pred, result, correct, TP,TN,FP,FN,folder_location]],columns=columns,index=[fname]))

        if (i+1)%100 ==0:
            print("Completed", i+1, "of", len(df.index), '| Current accuracy: {:.2f}%'.format(100*sum(results_arr)/len(results_arr)))
        # if i > 10:
        #     break
    print('Accuracy: {:.2f}%'.format(100*sum(results_arr)/len(results_arr)))
    df_results.to_csv(df_results_path)

# def make_predictions():
#
#     columns = ['fname', ]

if __name__ == '__main__':
    # get_file_list()
    # preprocess()
    # add_folder_to_df()
    # preprocess_for_pkl()
    # add_coughs_to_file()
    # create_predictions_template()
    # predict()
    if socket.gethostname() == 'area51.cs.washington.edu':
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    test()


# def get_clips():
#
#     df = pd.read_csv(os.path.join(folder_data, 'cough_classification_w_folder.csv'))
#     folder_mada_test = os.path.join(folder_npy, 'folder_mada_test')
#     os.makedirs(folder_mada_test, exist_ok=True)
#
#     for i in range(len(df.index)):
#         row = df.iloc[i]
#         fname = row.file
#         if row.cough not in ['yes', 'no']:
#             continue
#
#         path = os.path.join(folder_npy, Path(fname).stem + '.npy')
#
#         if not (os.path.exists(path)):
#             print("Npy doesn't exist - skpping:", fname)
#             continue
#
#         path_new = os.path.join(folder_mada_test, Path(fname).stem + '.npy')
#         copyfile(path, path_new)
