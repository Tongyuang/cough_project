import os
import pandas as pd
from collections import Counter
from pathlib import Path
import csv
import librosa
import numpy as np
import soundfile as sf

from config import config as c

label_file_header = ['start', 'end', 'label']
cnt_types = ['cnt','time_sec','skipped_cnt-too_short']

folder_wc = c['folder_raw']

def get_name_user_chall(file):
    name_no_ext = Path(file).stem
    name_no_ext = '_'.join(name_no_ext.split('_')[0:3])
    _, user, challenge = name_no_ext.split('_')
    return(name_no_ext,user,challenge)

# Sorts all of the audio files into a dictionary by user, challenge, and device
def get_device_files(file_labels,file_dict):

    name_no_ext, user, challenge = get_name_user_chall(file_labels)

    if user not in file_dict.keys():
        file_dict[user] = {}
    file_dict[user][challenge] = {}

    files_user_challenge = [f for f in os.listdir(folder_wc) if f.startswith(name_no_ext) and f.endswith('.wav')]

    for f in files_user_challenge:
        dev = Path(f).stem.split('_')[-1]
        if dev not in file_dict[user][challenge].keys():
            file_dict[user][challenge][dev] = []
        file_dict[user][challenge][dev] += [f]
    return(name_no_ext,user,challenge)

# We only have one label file per user/challenge. Each one has labels for the 5 devices in it. We need to split that out
# so there's one file per device, then process the labels to make sure they're compatible with preprocess.py
def create_dev_label_files(COUGH_VOL=False):

    files_labels = [f for f in os.listdir(folder_wc) if f.endswith('.txt') and 'uwct' in f]

    for file_labels in files_labels:

        name_no_ext, user, challenge = get_name_user_chall(file_labels)
        if 'uwct_1_0' not in name_no_ext:
            continue

        dict_df = {}

        # The label file has labels for each device. We'll first split these up into individual
        # dataframes for each device
        with open(os.path.join(folder_wc,file_labels), newline='') as csvfile:

            reader = csv.reader(csvfile, delimiter='\t')

            dev = None
            df = pd.DataFrame([],columns=label_file_header)
            first_row = True # see below for explanation of this and offset
            offset = 0
            for row in reader:
                start = row[0]
                end = row[1]
                label = row[2]

                # If start of new device, create new dataframe (and save old one)
                if "label_track" in label:
                    if dev is not None:
                        raise ValueError('Found a new device before reach end of file (eof) for previous device')

                    # Get device name for this dataframe
                    if name_no_ext == 'uwct_1_1':
                        dev = label.split('_')[-1]
                    else:
                        dev = label.split('_')[0]
                elif label.strip() in ['eof']:
                    print(name_no_ext, dev, "offset:", offset)
                    # Save this row to dataframe
                    df = df.append(pd.DataFrame([[float(start)-offset,float(end)-offset,label]],columns=label_file_header), ignore_index=True)
                    # Store the whole dataframe
                    dict_df[dev] = df
                    # Start new dataframe and reset variables
                    df = pd.DataFrame([],columns=label_file_header)
                    dev = None
                    first_row = True
                    offset = 0
                else:
                    # For files that start with nothing at the beginning (happens because we want to align the files
                    # across devices in audcatity so the coughs occur at the same time), audacity doesn't add that part
                    # into the wav file. As a result, only if the first label is "none", we need to add in an offset
                    # to all of the rest of the labels.
                    if first_row and label == 'none':
                        offset = float(end)
                    else:
                        # just save the row
                        df = df.append(pd.DataFrame([[float(start)-offset,float(end)-offset,label]],columns=label_file_header), ignore_index=True)
                    first_row = False

        # Next, we're going to extract the voluntary coughs and episodes from the dataframe. We do this in 2 parts
        # with normal coughs coming second because we need to check if the cough is inside of a larger cough episode.
        # If it is, we don't want to add both the cough and the episode because they would be repeats, so just add
        # the episode and skip the cough.
        for dev, df in dict_df.items():
            print("Starting:", name_no_ext, "- device:", dev)

            # Setup the counter
            cnt_dict = {}
            for cnt_type in cnt_types:
                cnt_dict[cnt_type] = Counter()

            # Walk through each line in df, add all voluntary coughs and episodes (we'll deal with coughs next)
            df_new = pd.DataFrame([], columns=label_file_header)
            for i in range(len(df.index)):
                row = df.iloc[i]
                start = row.start
                end = row.end
                label = row.label.strip()

                if label in ['vc','vci','vc?','vci?','vciw','v']:
                    label = 'cough_vol' if COUGH_VOL else 'cough'
                    idx = "cough_voluntary"
                elif 'bgsound' in label:
                    label = 'cough_bgsound'
                    idx = "cough_bgsound"
                elif label in ['ep','ep?']:
                    label = 'cough'
                    idx = "cough"
                elif label in ["none"]:
                    label = 'none'
                    idx = "none"
                elif label in ['eof']:
                    label = 'eof'
                    idx = 'eof'
                elif label in ['c','c?','cc','?','']:
                    continue
                else:
                    print("Label not found:", label)
                    continue

                cnt_dict[cnt_types[0]][idx] += 1
                cnt_dict[cnt_types[1]][idx] += (end - start)
                df_new = df_new.append(pd.DataFrame([[start, end, label]], columns=label_file_header),
                                       ignore_index=True)

            # Add coughs not inside an episode
            # For most label files, all events labeled "c" in the label file are inside a label of "ep". However,
            # for some of the label files, they aren't. As a result, we want to walk through all labels of "c" and check
            # that they are inside of a label of "ep". If not, we want to add them to the end of the dataframe. We'll
            # sort it by start time at the end.
            cnt_added = 0
            df_new_old = df_new.copy()
            for i in range(len(df.index)):
                row = df.iloc[i]
                start = row.start
                end = row.end
                label = row.label.strip()

                in_episode = False
                if label in ['c', 'c?', 'cc']:
                    for j in range(len(df_new_old.index)):
                        row_new = df_new_old.iloc[j]
                        start_new = row_new.start
                        end_new = row_new.end

                        #check if inside an episode
                        if start >= start_new and end <= end_new:
                            # print("Cough inside episode. Cough:",start, end, "Episode:",start_new,end_new)
                            in_episode = True
                            break
                    if not in_episode:
                        cnt_added += 1
                        # print("Adding cough", start, end)
                        df_new = df_new.append(pd.DataFrame([[start, end, "cough"]], columns=label_file_header),ignore_index=True)
                        cnt_dict[cnt_types[0]]["cough"] += 1
                        cnt_dict[cnt_types[1]]["cough"] += (end - start)

            # We added the regular coughs (not episodes) to end, so sort by start time for adding silence in the next part
            df_new = df_new.sort_values(by=['start'])

            print(dev, "- # Labels to Start:",len(df.index))
            print(dev, "- # Labels Not Including Coughs in Episodes:",len(df_new.index))

            # The whosecough datafiles don't have non-coughs labeled, so we add the label 'silence' in between label
            # events. It might not actually be silence (i.e. might actually be speech), but
            # we don't have it labeled so just bucket everything into "silence".
            end_last = 0
            df_new_old = df_new.copy()
            df_new = pd.DataFrame([], columns=label_file_header)
            for i in range(len(df_new_old.index)):
                row = df_new_old.iloc[i]
                start = row.start
                end = row.end
                label = row.label.strip()

                # If there's a sufficient gap, add a silence label
                if (start - end_last) > .001:
                    df_new = df_new.append(pd.DataFrame([[end_last, start, 'silence']], columns=label_file_header),
                                           ignore_index=True)
                    cnt_dict[cnt_types[0]]['silence'] += 1
                    cnt_dict[cnt_types[1]]['silence'] += (start - end_last)

                # none corresponds times when the audio cut out, so don't add that in
                if label not in ["none"]:
                    df_new = df_new.append(pd.DataFrame([[start, end, label]], columns=label_file_header),
                                       ignore_index=True)
                end_last = end

            # eof marks the end of the file. "start" is the actually time end of the file
            if df_new.iloc[-1].label is not 'eof':
                raise ValueError('The last row in this dataframe is not "eof" meaning there was some sort of problem.')
            #clip out the "eof" row
            df_new = df_new.iloc[:-1]

            print(dev, "- # Labels With Silence Added:", len(df_new.index))
            print(cnt_dict)
            cough_vol_suff = '_vc' if COUGH_VOL else ''
            path_new = os.path.join(folder_wc,'{}_{}{}.label.txt2'.format(name_no_ext,dev,cough_vol_suff))
            df_new.to_csv(path_new,sep='\t', index=False,header=False)

def convert_wc_speech_to_single_file():

    folder_speech = r'Z:\research\cough_count\data\raw\whosecough\speech'
    files = [f for f in os.listdir(folder_speech) if f.endswith('.wav') and f.startswith('speech')]

    subjects = sorted(list(set([f.split('_')[3] for f in files])))

    for subject in subjects:
        files_subject = [f for f in files if 'all_{}'.format(subject) in f]
        challenges = sorted(list(set([f.split('_')[4] for f in files_subject])))
        for challenge in challenges:
            devices = sorted(list(set([f.split('_')[5] for f in files_subject])))
            for dev in devices:
                name = 'uwct_{}_{}_{}_speech'.format(subject, challenge, dev)
                path_wav = os.path.join(c['folder_raw'],'{}.wav'.format(name))
                path_label = os.path.join(c['folder_raw'],'{}.label.txt2'.format(name))

                if os.path.exists(path_wav):
                    print(path_wav, "already exist, skipping")
                    continue

                files_new = [f for f in files if 'all_{}_{}_{}'.format(subject, challenge, dev) in f]

                y = np.array([])
                for f in files_new:
                    path = os.path.join(folder_speech, f)

                    # load the audio
                    y_new, _ = librosa.load(path, sr=c['sr'])

                    # Turn multiple channels into 1
                    if len(np.shape(y_new)) > 1:
                        y_new = np.mean(y_new, axis=1)

                    y = np.append(y,y_new)

                if len(y)<1:
                    print("not enough samples, skipping", name)
                    continue

                sf.write(path_wav, y, samplerate=c['sr'])
                with open(path_label,'w') as handle:
                    handle.write('{}\t{}\t{}'.format(0,(len(y)-1)/c['sr'],'speech'))

                print("Finished", name)


if __name__ == '__main__':
    COUGH_VOL = False
    create_dev_label_files(COUGH_VOL=COUGH_VOL)
    # convert_wc_speech_to_single_file()