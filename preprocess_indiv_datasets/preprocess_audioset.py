
from __future__ import unicode_literals
import youtube_dl
import soundfile as sf
import os
import pandas as pd
import librosa
import pickle
from collections import Counter
from pathlib import Path
import json
import numpy as np
from praatio import tgio

from config import config as c
from preprocess import combine_intervals

folder_audioset = os.path.join(c['folder_raw'],'audioset')
folder_wavs = os.path.join(folder_audioset, 'wavs')
bal = 'unbalanced'
ydl_opts = {
    'quiet': True,
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', str(c['sr']),
        '-ac', '1']
}

json_path = os.path.join(folder_audioset, 'ontology.json')
df_ont = pd.read_json(json_path).set_index('name')

def n2id(name):
    return(df_ont.loc[name, 'id'])
def id2name(id):
    return(df_ont[df_ont['id']==id].iloc[0].name)
# Gets the ids for the classes that have been chosen
def get_all_ids(chosen_classes):
    chosen_ids = []

    # check the chosen classes. if they have child ids, add those to the list
    # if not, add that id to the list of chosen ids
    for cc in chosen_classes:
        row = df_ont.loc[cc]
        childs = row['child_ids']
        chosen_ids.append(row.id)
        # if not(childs):
        #     chosen_ids.append(row.id)
        # else:
        if childs:
            #add these classes to list so we can get their ids
            chosen_classes += [id2name(f) for f in childs]
    return(chosen_ids)

# Gets a dataframe with the youtube IDs in a good format
def get_df_YTID():

    path_df_ytid = os.path.join(folder_audioset,'df_ytid.csv')
    if os.path.exists(path_df_ytid):
        df_ytid = pd.read_csv(path_df_ytid)
        print("Loaded df_ytid")
    else:
        youtube_id_path = os.path.join(folder_audioset, '{}_train_segments.csv'.format(bal))
        columns = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']
        df_ytid = pd.read_csv(youtube_id_path, skiprows=3, sep=' ', header=None, names=columns)
        df_ytid_strip = df_ytid[['YTID', 'start_seconds', 'end_seconds']]

        df_ytid[['YTID', 'start_seconds', 'end_seconds']] = df_ytid_strip.apply(lambda x: x.str.strip(','))
        df_ytid[['start_seconds', 'end_seconds']] = df_ytid[['start_seconds', 'end_seconds']].apply(pd.to_numeric)
        df_ytid = df_ytid.set_index('YTID')
        df_ytid.to_csv(os.path.join(folder_audioset, 'df_ytid.csv'))

        print("Created df_ytid and saved to csv")

    return(df_ytid)

def get_YTIDs_from_chosen_ids(df_ytid, chosen_ids, typ):

    suff = '_music' if typ == 'music' else ''
    pickle_path = os.path.join(folder_audioset, '{}_ytids_and_ytids_dict{}.pkl'.format(bal, suff))
    print("PICKLE PATH:", pickle_path)

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            ytids, ytids_by_id_dict = pickle.load(handle)
        print("Loaded ytids from file")
    else:
        ytids = []
        ytids_by_id_dict = {}
        for c_i in chosen_ids:
            ytids_by_id_dict[c_i] = []
        for i in range(len(df_ytid.index)):
            row = df_ytid.iloc[i]
            labels = row.positive_labels.split(',')
            for lab in labels:
                if lab in chosen_ids:
                    ytids.append(row.name)
                    ytids_by_id_dict[lab].append(row.name)
                    break
            if (i+1) % 5000 == 0:
                print("Finished {:.2f}% -".format(100*(i+1)/len(df_ytid.index)), (i+1), "ytids of", len(df_ytid.index),)
        with open(pickle_path, 'wb') as handle:
            pickle.dump([ytids, ytids_by_id_dict], handle)
        print("Saved ytids to file")

    print("\nCounts per type:")
    ytids_by_id_dict_return = {}
    for k, v in ytids_by_id_dict.items():

        if len(v)>0:
            ytids_by_id_dict_return[k]=v
            print("Added", id2name(k), len(v))
        else:
            print("Skipping", id2name(k), len(v))
    print("")
    return(ytids, ytids_by_id_dict_return)

def download_yt_clip_and_save_wav(ytid, df_ytid):

    saved_wav_path = os.path.join(folder_wavs, '{}.wav'.format(ytid))
    if os.path.exists(saved_wav_path):
        print(ytid, "already saved - skipping")
        return

    ydl_opts['outtmpl'] ='{}.wav'.format(ytid)
    link = 'https://www.youtube.com/watch?v={}'.format(ytid)
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info("{}".format(link))
            filename = ydl.prepare_filename(result)
    except:
        print("Couldn't download file", ytid, '- skipping')
        return
    data, _ = librosa.load(filename, sr=c['sr'])
    start = int(df_ytid.loc[ytid].start_seconds * c['sr'])
    end = int(df_ytid.loc[ytid].end_seconds * c['sr'])
    data_clipped = data[start:end]
    sf.write(saved_wav_path, data_clipped, samplerate=c['sr'])
    print('Saved','{}.wav'.format(ytid))
    os.remove(filename)

def download_and_save_all(ytids, df_ytid):
    for i, ytid in enumerate(ytids):
        download_yt_clip_and_save_wav(ytid, df_ytid)
        if (i + 1) % 10 == 0:
            print("\nFinished {:.2f}% -".format(100 * (i + 1) / len(ytids)), i + 1, "yt_audios of", len(ytids), "\n")

def convert_df_ytid_pkls_to_csv():

    pickle_path_1 = os.path.join(folder_audioset, '{}_ytids_and_ytids_dict.pkl'.format(bal))
    pickle_path_2 = os.path.join(folder_audioset, '{}_ytids_and_ytids_dict_music.pkl'.format(bal))
    pickle_path_3 = os.path.join(folder_audioset, '{}_ytids_and_ytids_dict_coughs.pkl'.format(bal))

    columns = ['label', 'fname']
    to_save = None
    for path in [pickle_path_1, pickle_path_2, pickle_path_3]:
        with open(path, 'rb') as handle:
            ytids, ytids_by_id_dict = pickle.load(handle)

        for id, ytid_list in ytids_by_id_dict.items():
            names = [id2name(id)] * len(ytid_list)
            ytid_list = [y+'.wav' for y in ytid_list]
            to_add = np.array([names,ytid_list]).T
            to_save = np.vstack((to_save,to_add)) if to_save is not None else to_add

    pd.DataFrame(to_save,columns=columns).to_csv(os.path.join(folder_audioset,'audioset_files.csv'),index=False)

def save_to_pkl(ytids_by_id_dict):


    for id, ytid_list in ytids_by_id_dict.items():

        name = id2name(id)
        pkl_name = 'audioset_{}.pkl'.format(name)
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

        print('Processing', len(ytid_list), name, 'files')

        for ytid in ytid_list:
            try:
                y, _ = sf.read(os.path.join(folder_wavs, ytid + '.wav'))
            except:
                print("Couldn't read wav for ytid", ytid)
                continue

            name_with_domain = 'audioset_{}_{}'.format(name, ytid)
            sound_segs_per_file = 0
            cough_count = 0
            intervals = librosa.effects.split(y, top_db=20)
            intervals = combine_intervals(intervals)
            for i, (start_samps, end_samps) in enumerate(intervals):
                if end_samps - start_samps < .1 * c['sr']:
                    continue

                pickle_dict, pickle_counter, cough_count = process_samp(y, start_samps / c['sr'], end_samps / c['sr'],
                                                                        'sound', pickle_dict, pickle_counter,
                                                                        name_with_domain, name_with_domain, cough_count,
                                                                        suff='audioset_{}'.format(name))

                sound_segs_per_file += 1
            print('Finished', ytid, name, "segs in file:", sound_segs_per_file)

        print("sound cnt:", pickle_counter["cnt"]["sound"], "| Time (s):", pickle_counter["time_sec"]["sound"])
        print("Finished preprocessing on:", pkl_name)

        # Save pickle file and counter for this file
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump([pickle_dict, pickle_counter], pkl_file)


def flusense_textgrid_to_txt():

    folder = os.path.join(c['folder_raw'],'flusense')
    folder_labels = os.path.join(folder, 'labels_textgrid')

    labs = [f for f in os.listdir(folder_labels) if f.endswith('.TextGrid')]

    labs_all = []
    for lab in labs:

        name_no_ext = Path(lab).stem
        path = os.path.join(folder_labels, lab)
        path_new = os.path.join(folder, name_no_ext+'.label.txt')

        # if os.path.exists(path_new):
        #     continue

        tg = tgio.openTextgrid(path)
        t_name = tg.tierNameList[0]
        entryList = tg.tierDict[t_name].entryList

        with open(path_new, 'w') as handle:
            for entry in entryList:
                lab_cur = entry.label
                if lab_cur in c['label_convert'].keys():
                    lab_cur = c['label_convert'][lab_cur]

                labs_all += [lab_cur]
                handle.write('{}\t{}\t{}\n'.format(entry.start, entry.end, lab_cur))

    print(set(labs_all))


if __name__ == '__main__':

    # typ = 'sounds' #'sounds' #'music
    # assert(typ in ['music','sounds'])
    # chosen_classes_music = ['Music genre']
    # chosen_classes_sounds = ['Sneeze', 'Laughter', 'Baby cry, infant cry', 'Dishes, pots, and pans', 'Cutlery, silverware']
    # # chosen_classes_coughs = ['Cough', 'Throat clearing']
    #
    # chosen_classes = chosen_classes_music if typ=='music' else chosen_classes_sounds
    # # chosen_classes = chosen_classes_coughs
    #
    # df_ytid = get_df_YTID()
    # chosen_ids = get_all_ids(chosen_classes)
    #
    # ytids, ytids_by_id_dict = get_YTIDs_from_chosen_ids(df_ytid, chosen_ids, typ)


    # download_and_save_all(ytids,df_ytid)
    # save_to_pkl(ytids_by_id_dict)
    # convert_df_ytid_pkls_to_csv()
    # df = pd.read_csv(r'Z:\research\cough_count\data\raw\audioset\audioset_files.csv')
    flusense_textgrid_to_txt()
