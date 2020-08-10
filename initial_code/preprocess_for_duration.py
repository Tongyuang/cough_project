import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import audioread
import random
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import progressbar


import config
from config import config as c

audio_endings = ('.flac', '.wav')
label_file_header = ['start', 'end', 'label']

audioset_csv_columns = ['silence', 'vomit', 'snore', 'etc', 'gasp', 'hiccup', 'wheeze', 'cough', 'sneeze', 'burp',
                         'breathe', 'speech', 'throat-clearing', 'sniffle']

def get_domain(name, by_device=False, label = False):
    if '-cp' in name or '-n' in name:
        d = 'coughsense'
    elif name.startswith('CF'):
        d = 'pediatric'
    elif 'TDRS' in name:
        d = 'southafrica'
    elif 'uwct' in name:
        d = 'whosecough'
        if by_device:
            dev = name.split('_')[-1]
            d = d + '_{}'.format(dev)
    elif 'respeaker_4mic' in name or '4mic_array' in name:
        d = 'respeaker4mic'
    elif 'FSDKaggle' in name:
        d = 'FSDKaggle'
    elif 'audioset' in name:
        d = 'audioset'
    elif 'flusense' in name:
        d = 'flusense'
    elif 'jotform' in name:
        d = 'jotform'
    elif 'demand' in name:
        d = 'demand'
    elif 'madagascar' in name or '--2018' in name:
        d = 'madagascar'
    else:
        d = 'unknown'
    if label:
        if by_device:
            return(c['domains_devices'].index(d))
        else:
            return(c['domains'].index(d))
    else:
        return(d)

def get_raw_folder(dom):
    if dom == 'audioset':
        return(os.path.join(c['folder_raw'], r'audioset\audioset_csv_coughs'))
    elif dom == 'flusense':
        return(os.path.join(c['folder_raw'], r'flusense'))
    else:
        return(c['folder_raw'])


def get_name(name):

    name_no_ext = Path(name).stem
    # Whosecough data has weird naming
    if 'uwct' in name_no_ext:
        if 'speech' not in name_no_ext:
            name_split = name_no_ext.split('_')
            name_no_ext = '_'.join(name_split[:3] + [name_split[5]])
    elif 'TDRS' in name_no_ext:
        pass
    elif get_domain(name) in ['flusense', 'audioset']:
        name = '_'.join(name.split('_')[1:])
        name_no_ext = '_'.join(name_no_ext.split('_')[1:])
    else:
        name_no_ext = '_'.join(name_no_ext.split('_')[:4])

    name_no_ext_no_us = name_no_ext.replace('_','-')

    return(name, name_no_ext, name_no_ext_no_us)

def get_labels(dom, name_no_ext):

    # Label file path. TDRS data has weird label file name
    if dom == 'southafrica':
        label_name = name_no_ext + '_new.label.txt2'
    elif dom in ['coughsense','audioset','flusense']:
        label_name = name_no_ext + '.label.txt'
    else:
        label_name = name_no_ext + '.label.txt2'

    path_label = os.path.join(get_raw_folder(dom), label_name)

    # Try to load the label file
    try:
        df = pd.read_csv(path_label, sep=r'\t', header=None, engine='python')
        df.columns = label_file_header
    except FileNotFoundError:
        print('ERROR - File:', label_name, 'could not be found')
        return([], False)
    except pd.errors.EmptyDataError:
        print("No labels, skipping")
        return([], False)
    return(df, True)

def convert_labels_to_df_to_time_array(df, y_len):

    labels_arr = np.zeros((int(y_len/c['duration_samps'])))

    df_cough = df[df.label=='cough']
    for i in df_cough.index.values:
        start_idx = int(np.ceil(df_cough.loc[i].start * c['sr'] / c['duration_samps']))
        end_idx = int(np.floor(df_cough.loc[i].end * c['sr'] / c['duration_samps']))

        if end_idx > len(labels_arr):
            break
        labels_arr[start_idx:end_idx+1] = 1

    return(labels_arr)

def get_wav(path, dom, path_wavfile, name_no_ext):

    try:
        # Verify the sample rate of the file is at least greater than the project samplerate
        file_sr = sf.info(path).samplerate
        if file_sr < c['sr']:
            print("ERROR - Samplerate for", name_no_ext, "of", file_sr, "hz is less than program samplerate of",
                  c['sr'],
                  "hz. File not imported")
            return ([], False)

        # load the audio
        y, _ = librosa.load(path, sr=c['sr'])

        # Turn multiple channels into 1
        if len(np.shape(y)) > 1:
            y = np.mean(y, axis=1)

        # clip to max of 1/-1 to match normal wavs
        np.clip(y, -1.0, 1.0, y)

        # only save the long files
        if dom not in ['audioset', 'FSDKaggle', 'madagascar', 'respeaker4mic', 'flusense', 'jotform', 'demand']:
            # Save to numpy
            wavfile.write(path_wavfile, c['sr'], y)
            print("Saved as wavfile.")

    except (audioread.exceptions.NoBackendError, RuntimeError):
        print("ERROR - Could not read audio for:", name_no_ext, "- skipping file")
        return ([], False)

    return(y,True)

def get_audio(name, name_no_ext, dom):

    path = os.path.join(get_raw_folder(dom), name)

    # Load from wavfile if exists
    path_wavfile = os.path.join(c['folder_wavfile'], name_no_ext + '_{}khz_wavfile.wav'.format(int(c['sr'] / 1000)))

    # path_wavfile = path_wavfile.replace('.wav','_trunc.wav')

    if os.path.exists(path_wavfile):
        _, y = wavfile.read(path_wavfile)

        # clip to max of 1/-1 to match normal wavs (some of the old saved numpys didn't do this)
        if max(abs(y))>1:
            np.clip(y,-1.0,1.0, y)
            os.remove(path_wavfile)
            wavfile.write(path_wavfile, c['sr'], y)

        print("Loaded from wavfile.")

        # y = y[:10*60*c['sr']]
        # path_wavfile = path_wavfile.replace('.wav','_trunc.wav')
        # wavfile.write(path_wavfile,c['sr'],y)
        # raise

    # If not, load the actual audio and save to numpy (for next time)
    else:
        y, result = get_wav(path, dom, path_wavfile, name_no_ext)
        if not result:
            return([], False)

    y = y[:int(np.floor(len(y) / c['duration_samps']) * c['duration_samps'])]  # make number of samples divisible by min duration between labels

    print("Finished loading audio.")
    return(y, True)

def save_wav(dom, name_no_ext_no_us, start_samps, end_samps, y, labels_arr, folder, cnt):

    wav_name = 'wav_{}_{}_{}.wav'.format(dom, name_no_ext_no_us, cnt)
    label_name = 'label_{}_{}_{}.pkl'.format(dom, name_no_ext_no_us, cnt)

    # Get samp and labels
    samp = y[start_samps:end_samps]
    wavfile.write(os.path.join(folder, wav_name), c['sr'], samp)

    if labels_arr is not None:
        labels = labels_arr[int(start_samps / c['duration_samps']):int(end_samps / c['duration_samps'])]

        with open(os.path.join(folder, label_name), 'wb') as handle:
            pickle.dump(labels, handle)

    return(Path(wav_name).stem)

def save_all_wavs(df, labels_arr, dom, name_no_ext_no_us, y):

    wc_spch = True if dom in ['whosecough'] and 'speech' in name_no_ext_no_us else False
    hop_samps = int(config.MAX_SAMPS/2)

    df_files_indiv = pd.DataFrame([], columns=all_files_csv_columns)

    folder_dom = os.path.join(c['folder_dur'], dom)
    os.makedirs(folder_dom, exist_ok=True)

    # Walk through full file getting samples of size "window". If they have a cough or rejected, just skip the segment
    # since we have enoug segments, dont need those. also skip if has just silence
    cnt = 0

    if dom is not 'flusense':
        bar = progressbar.ProgressBar(maxval=int(len(y)/hop_samps)).start()
    for i, start_samps in enumerate(np.arange(0, len(y)-config.MAX_SAMPS+1, hop_samps)):
        if dom is not 'flusense':
            bar.update(i)
        end_samps = start_samps + config.MAX_SAMPS
        start_t = start_samps/c['sr']
        end_t = end_samps/c['sr']

        df_labels = df[np.logical_not(np.logical_or(df.end <= start_t, df.start >= end_t))]

        # if len(df_labels.index.values) < 1 or df_labels.label.str.contains('cough').any() or df_labels.label.str.contains('rejected').any():
        # Throw it out if don't have labels (happens for coughsense 18-n) or there's a 'rejected' in it
        if dom == 'audioset' and len(df_labels.index.values) < 1:
            df_labels = df_labels.append(pd.DataFrame([[start_t, end_t, 'unknown']], columns=label_file_header),ignore_index=True)

        if len(df_labels.index.values) < 1 or df_labels.label.str.contains('rejected').any():
            continue

        # Clip the first and last label to the size of this frame
        df_labels.at[df_labels.index.values[0], 'start'] = max(df_labels.loc[df_labels.index.values[0], 'start'], start_t)
        df_labels.at[df_labels.index.values[-1], 'end'] = min(df_labels.loc[df_labels.index.values[-1], 'end'], end_t)

        labels_cnt_arr = np.zeros(len(all_labels))

        # Store how much of each type of label exists in this frame
        for idx in df_labels.index.values:
            row = df_labels.loc[idx]
            if row.label not in all_labels:
                print("Label not found:",row.label,"| file:", name_no_ext_no_us)
                row.label = 'unknown'
            labels_idx = all_labels.index(row.label)
            labels_cnt_arr[labels_idx] += row.end-row.start

        # Skip if there's just silence/noise/speech - we have more than enough of the samples already
        if dom in ['whosecough'] and 'speech' in name_no_ext_no_us: #whosecough doesn't have these labeled, and need the whosecough_speech files to go through
            pass
        else:
            if len(df_labels.index.values) == 1 and df_labels.iloc[0].label in ['silence', 'noise', 'speech']:
                continue

        name = save_wav(dom, name_no_ext_no_us, start_samps, end_samps, y, labels_arr, folder_dom, cnt)
        df_files_indiv = df_files_indiv.append(pd.DataFrame([[dom, name_no_ext_no_us, name] + list(labels_cnt_arr)],
                                                            columns=all_files_csv_columns),
                                               ignore_index=True)
        cnt+=1

    # # Now get coughs
    # df_coughs = df[df.label=='cough']
    # cough_intervals = None
    # for idx in df_coughs.index.values:
    #     int_new = np.array([[int(df_coughs.loc[idx,'start']*c['sr']), int(df_coughs.loc[idx,'end']*c['sr'])]])
    #     cough_intervals = np.vstack((cough_intervals,int_new)) if cough_intervals is not None else int_new
    #
    # min_size = int(config.MAX_SAMPS*3/2) if dom is not 'flusense' else config.MAX_SAMPS
    # cough_intervals_comb = combine_intervals(cough_intervals, comb_size=config.MAX_SAMPS, comb_derate=4,
    #                                          remove_short_sec=False, min_size = min_size, len_y=len(y))
    #
    # labels_cnt_arr = np.zeros(len(all_labels))
    # labels_cnt_arr[all_labels.index('cough')]=1
    # for i, (start_samps, end_samps) in enumerate(cough_intervals_comb):
    #     name = save_wav(dom, name_no_ext_no_us, start_samps, end_samps, 'cough', y, labels_arr, folder_cough, i)
    #     df_files_indiv = df_files_indiv.append(pd.DataFrame([[dom, 'cough', name_no_ext_no_us, name] + list(labels_cnt_arr)], columns=all_files_csv_columns),
    #                                ignore_index=True)

    return(df_files_indiv)

# This function walks through each audio file, clips out chunks based on the label file, converts them to a
# a mel-spectrogram, then saves them to a pickle file. It also saves the loaded audio file to numpy since these load
# much much faster.
def preprocess_from_wavs():

    audio_files = sorted([f for f in os.listdir(c['folder_raw']) if f.endswith(audio_endings) and 'khz' not in f and '4mic' not in f])
    audio_files_flusense = ['flusense_'+f for f in os.listdir(os.path.join(c['folder_raw'],'flusense')) if f.endswith(audio_endings)]
    audio_files_audioset_csv_coughs = ['audioset_' + f for f in os.listdir(os.path.join(c['folder_raw'], r'audioset\audioset_csv_coughs')) if f.endswith(audio_endings)]
    audio_files += audio_files_flusense + audio_files_audioset_csv_coughs

    print("Num audio files:", len(audio_files))

    for name in audio_files:

        dom = get_domain(name)
        name, name_no_ext, name_no_ext_no_us = get_name(name)

        if dom in ['respeaker4mic']:
            continue

        if not os.path.exists(c['files_csv']):
            df_files = pd.DataFrame([], columns=all_files_csv_columns)
        else:
            df_files = pd.read_csv(c['files_csv'])

        if (df_files['name'] == name_no_ext.replace('_','-')).any():
            print("File already preprocessed, skipping -", name_no_ext_no_us)
            continue

        print("Starting preprocessing on:", name)

        df, result = get_labels(dom, name_no_ext)
        if not result:
            continue

        y, result = get_audio(name, name_no_ext, dom)

        if not result:
            continue

        labels_arr = convert_labels_to_df_to_time_array(df, len(y))

        df_files_indiv = save_all_wavs(df, labels_arr, dom, name_no_ext_no_us, y)

        df_files = df_files.append(df_files_indiv)
        df_files.to_csv(c['files_csv'], index=False)

        print("Finished preprocessing on:", name)

def combine_intervals(intervals, comb_size=config.MAX_SAMPS, comb_derate=2, remove_short_sec=False, min_size=False,
                      len_y=None, max_size=-1):

    if min_size:
        assert(len_y is not None)

    HALF_SAMPS = int(np.ceil(min_size / 2))

    # Combine intervals
    intervals_new = np.array([intervals[0]])
    prev_end = intervals[0][1]

    if len(intervals) > 1:
        for i, (start_samps, end_samps) in enumerate(intervals[1:]):

            if max_size>0 and (end_samps - intervals_new[-1][0]) > max_size : # don't combine if the new samp would be too long
                intervals_new = np.vstack((intervals_new, [start_samps, end_samps]))
            elif start_samps-prev_end < (comb_size/comb_derate):
                intervals_new[-1][1] = end_samps
            else:
                intervals_new = np.vstack((intervals_new,[start_samps,end_samps]))
            prev_end = end_samps

    intervals = intervals_new.copy()
    intervals_new = None

    if min_size:
        # if the sample is shorter than the min_size, make the interval the whole thing and return
        if len_y <= min_size:
            return(np.array([[0,len_y-1]]))

    # Now remove short intervals and/or make each interval the min size
    if remove_short_sec or min_size:
        for i, (start_samps, end_samps) in enumerate(intervals):

            # get rid of really short samples
            if remove_short_sec and end_samps - start_samps < remove_short_sec * c['sr']:
                continue

            # make at least min size
            if min_size and end_samps - start_samps < min_size:

                # if interval start would extend past start, start at 0
                if start_samps - HALF_SAMPS < 0:
                    start_samps = 0
                    end_samps = min_size

                # if interval end would extend past end, end at the end of the samp
                elif end_samps + HALF_SAMPS >= len_y:
                    end_samps = len_y-1
                    start_samps = end_samps-min_size

                #otherwise, just make an interval centered on the original event
                else:
                    start_samps = int(np.mean([start_samps,end_samps]))-HALF_SAMPS
                    end_samps = start_samps+min_size

            intervals_new = np.vstack((intervals_new,[start_samps,end_samps])) if intervals_new is not None else np.array([[start_samps,end_samps]])
    else:
        intervals_new = intervals.copy()

    intervals_new = np.array([[]]) if intervals_new is None else intervals_new
    return(intervals_new)

def preprocess_from_csv():

    dom_dict_list = [{'dom':'FSDKaggle', 'meta':'train_post_competition_cough_labeled.xlsx', 'data':'FSDKaggle2018.audio_train'},
                     {'dom':'audioset', 'meta':'audioset_files.csv', 'data':'wavs'}]

    if not os.path.exists(c['files_csv']):
        df_files = pd.DataFrame([], columns=all_files_csv_columns)
    else:
        df_files = pd.read_csv(c['files_csv'])

    df_files_indiv = pd.DataFrame([], columns=all_files_csv_columns)
    for dom_dict in dom_dict_list:

        dom = dom_dict['dom']
        folder_dom = os.path.join(c['folder_dur'], dom)
        os.makedirs(folder_dom, exist_ok=True)

        # if dom == 'FSDKaggle':
        #     continue

        folder = os.path.join(c['folder_raw'], dom)
        path_meta = os.path.join(folder, dom_dict['meta'])
        folder_data = os.path.join(folder, dom_dict['data'])
        folder_data_coughs = os.path.join(folder_data, 'FSDKaggle2018.audio.train.coughs') if dom == 'FSDKaggle' else folder_data

        df = pd.read_excel(path_meta) if dom == 'FSDKaggle' else pd.read_csv(path_meta)
        labels = sorted(df.label.unique())

        labels_arr = np.zeros(int(config.MAX_SAMPS/c['duration_samps']), dtype=int) #zeros because these will all have no coughs in them

        for label in labels:

            # if dom =='audioset' and label.lower() =='cough':
            #     continue
            if label.lower() =='cough': #we're not doing coughs because we don't have labels for the frame-level coughs
                continue

            name_sound = '{}_{}'.format(dom, label)

            if (df_files['name'] == name_sound).any():
                print("File already preprocessed, skipping -", name_sound)
                continue

            label_name_fixed = label.replace('_', '-').replace(' ','-').lower()
            label_name_fixed = 'instrument' if label_name_fixed in sounds_dict['instrument'] else label_name_fixed
            label_name_fixed = 'music' if label_name_fixed in sounds_dict['music'] else label_name_fixed

            labels_cnt_arr = np.zeros(len(all_labels))
            labels_cnt_arr[all_labels.index(label_name_fixed)] = 5

            df_label = df[df.label==label]

            if dom == 'FSDKaggle' and label == 'Cough':
                # we listened to these and many are not actually coughs. this gets them by their manual label
                df_label = df_label[df_label.cough == 1]

            bar = progressbar.ProgressBar(maxval=len(df_label.index)).start()
            for i in range(len(df_label.index)):
                cnt = 0
                bar.update(i)

                row = df_label.iloc[i]

                folder = folder_data if label != 'Cough' else folder_data_coughs
                f = Path(row.fname).stem+'_clipped.wav' if (dom == 'FSDKaggle' and label == 'Cough' and row.clipped) else row.fname

                path = os.path.join(folder, f)
                if not(os.path.exists(path)):
                    if dom == 'audioset':
                        print("Couldn't find wav -", f)
                        continue
                    else:
                        raise ValueError(path, 'doesnt exist')

                f = f.replace('_clipped.wav', '.wav')
                name_no_ext_no_us = Path(f).stem.replace('_','-').replace(' ','-')

                y, result = get_wav(path, dom, [], f)
                if not result:
                    print("Couldn't get wav", f)
                    continue

                intervals = librosa.effects.split(y, top_db=20)
                intervals_comb = combine_intervals(intervals, comb_size=int(config.MAX_SAMPS*3/4), comb_derate=1,
                                                   remove_short_sec=False, min_size = config.MAX_SAMPS/c['sr'],
                                                   len_y=len(y), max_size=int(config.MAX_SAMPS*3/2))

                for start_samps, end_samps in intervals_comb:

                    # get rid of really short samples
                    # if end_samps - start_samps < .15*c['sr']:
                    #     continue

                    name = save_wav(dom, name_no_ext_no_us, start_samps, end_samps, y, labels_arr, folder_dom, cnt)

                    df_files_indiv = df_files_indiv.append(pd.DataFrame([[dom, name_no_ext_no_us, name]
                                                                         + list(labels_cnt_arr)],
                                                                        columns=all_files_csv_columns),
                                                           ignore_index=True)
                    cnt+=1

            df_files = df_files.append(df_files_indiv)
            df_files.to_csv(c['files_csv'], index=False)

            print("Finished preprocessing on:", name_sound)

def preprocess_jotform():

    dom = 'jotform'
    folder_jotform = os.path.join(c['folder_raw'], 'jotform')
    jotform_email_to_id_csv = os.path.join(folder_jotform, 'jotform_email2id.csv')
    csv_columns = ['name']

    if not os.path.exists(c['files_csv']):
        df_files = pd.DataFrame([], columns=all_files_csv_columns)
    else:
        df_files = pd.read_csv(c['files_csv'])

    folder_dom = os.path.join(c['folder_dur'], dom)
    os.makedirs(folder_dom, exist_ok=True)

    if not os.path.exists(jotform_email_to_id_csv):
        df_id = pd.DataFrame([],columns=csv_columns)
    else:
        df_id = pd.read_csv(jotform_email_to_id_csv, index_col=0)

    user_dict = {}

    for root, dirs, _ in os.walk(folder_jotform):
        root_bname = os.path.basename(root)
        if root_bname == 'jotform' or root_bname.split('_')[-1] in ['cough', 'speech', 'laugh', 'clear']:
            continue
        user_dict[root_bname] = dirs

    df_files_indiv = pd.DataFrame([], columns=all_files_csv_columns)

    bar = progressbar.ProgressBar(maxval=len(user_dict.keys())).start()
    cnt_u = 0
    for u, dirs in user_dict.items():
        cnt_u+=1

        # add this file to the id spreadsheet
        if u not in df_id.name.values:
            df_id = df_id.append(pd.DataFrame([[u]], columns=csv_columns), ignore_index=True)
            df_id.to_csv(jotform_email_to_id_csv)

        # get this patient id
        id = df_id.index[df_id['name'] == u].tolist()[0]
        name = 'jotform-' + str(id)

        # check if we've already preprocessed these files
        if (df_files['name'] == name).any():
            print("File already preprocessed, skipping -", name)
            continue

        cnt=0
        for d in dirs:

            label = d.split('_')[-1]

            # convert form to what the others have used
            label = 'throat-clearing' if 'clear' in label else label
            label = 'laughter' if label == 'laugh' else label

            labels_cnt_arr = np.zeros(len(all_labels))
            labels_cnt_arr[all_labels.index(label)] = 5

            path_dir = os.path.join(folder_jotform, u, d)
            for f in os.listdir(path_dir):

                path = os.path.join(path_dir, f)

                y, result = get_wav(path, 'jotform', [], f)
                if not result:
                    print("Couldn't get wav", f)
                    continue

                y, _ = librosa.effects.trim(y, top_db=30, frame_length=160, hop_length=80)
                y = y[:int(np.floor(len(y)/c['duration_samps']) * c['duration_samps'])]
                labels_arr = np.zeros(int(len(y)/c['duration_samps']))

                if label == 'cough':

                    intervals = librosa.effects.split(y, top_db=40, frame_length=160,hop_length=80)

                    for (start_samps, end_samps) in intervals:
                        start_idx = int(np.ceil(start_samps / c['duration_samps']))
                        end_idx = int(np.floor(end_samps / c['duration_samps']))

                        labels_arr[start_idx:end_idx+1] = 1

                wavfile.write(path, c['sr'], y)

                name_new = save_wav(dom, name, 0, len(y), y, labels_arr, folder_dom, cnt)
                df_files_indiv = df_files_indiv.append(pd.DataFrame([[dom, name, name_new]
                                                                     + list(labels_cnt_arr)],
                                                                    columns=all_files_csv_columns),
                                                       ignore_index=True)

                cnt += 1

        df_files = df_files.append(df_files_indiv)
        df_files.to_csv(c['files_csv'], index=False)
        bar.update(cnt_u)

        # print("Finished preprocessing:", name, cntr)

def preprocess_demand():
    folder_demand_raw = r'Z:\research\cough_count\data\raw\demand'
    folder_aug = os.path.join(c['folder_aug'], 'demand')
    os.makedirs(folder_aug, exist_ok=True)

    cnt = 0
    for root, dirs, wavs in os.walk(folder_demand_raw):
        bname = os.path.basename(root)
        if not(bname.endswith('16k')):
            continue

        path_wav = os.path.join(root, dirs[0], 'ch01.wav')

        y, result = get_wav(path_wav, 'demand', [], bname)
        if not result:
            print("Get can't wav for:", bname)

        for start_samps in np.arange(0,len(y)-config.MAX_SAMPS, int(config.MAX_SAMPS/2)):
            end_samps = start_samps + config.MAX_SAMPS
            name_new = save_wav('demand', bname.replace('_', '-'), start_samps, end_samps, y, None, folder_aug, cnt)
            cnt+=1

        print("Finished:", bname)

def preprocess_from_folders():

    preprocess_jotform()


def preprocess_wrapper(wavs=False,csv=False, folders=False):

    print("CURRENT SAMPLE RATE:", c['sr'])
    print("Audio Files Folder:", c['folder_raw'])

    os.makedirs(c['folder_wav'], exist_ok=True)
    if wavs:
        preprocess_from_wavs()
    if csv:
        preprocess_from_csv()
    if folders:
        preprocess_from_folders()

def get_sound_categories():

    df = pd.read_excel(os.path.join(os.path.dirname(c['folder_meta']), 'sound_categories.xlsx'))
    sounds_dict = {}
    for i in range(len(df.index)):
        row = df.iloc[i, :]
        sound = row.sound
        type = row.type
        if type not in sounds_dict.keys():
            sounds_dict[type] = []
        sounds_dict[type] += [sound]

    all_labels = list(df.sound)
    return(sounds_dict,all_labels)

if __name__ == '__main__':

    sounds_dict, all_labels = get_sound_categories()
    all_files_csv_columns = ['domain', 'name', 'name-long'] + all_labels

    wavs = False
    csv = False
    folders = True

    # preprocess_wrapper(wavs=wavs,csv=csv,folders=folders) #coughsense, pediatric, southafrica, whosecough
    preprocess_demand()

