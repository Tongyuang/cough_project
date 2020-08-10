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


import config
from config import config as c

audio_endings = ('.flac', '.wav')
label_file_header = ['start', 'end', 'label']

all_files_csv_columns = ['domain','type','name'] + c['label_types']
audioset_csv_columns = ['silence', 'vomit', 'snore', 'etc', 'gasp', 'hiccup', 'wheeze', 'cough', 'sneeze', 'burp',
                         'breathe', 'speech', 'throat-clearing', 'sniffle']
MAX_SAMPS_PER_WAV = 10 * c['sr']

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

    # Load from npy if exists
    path_wavfile = os.path.join(c['folder_wavfile'], name_no_ext + '_{}khz_wavfile.wav'.format(int(c['sr'] / 1000)))
    if os.path.exists(path_wavfile):
        _, y = wavfile.read(path_wavfile)

        # clip to max of 1/-1 to match normal wavs (some of the old saved numpys didn't do this)
        if max(abs(y))>1:
            np.clip(y,-1.0,1.0, y)
            os.remove(path_wavfile)
            wavfile.write(path_wavfile, c['sr'], y)

        print("Loaded from wavfile.")
    # If not, load the actual audio and save to numpy (for next time)
    else:
        y, result = get_wav(path, dom, path_wavfile, name_no_ext)
        if not result:
            return([], False)

    print("Finished loading audio.")
    return(y, True)

def save_wav(y, start_samp, end_samp, label, dom, name_no_ext_no_us, num_saved=0, idx =0, length=MAX_SAMPS_PER_WAV,
             one_samp=False, min_size=False):

    samp_len = end_samp-start_samp

    # if the sample is longer than max length (i.e. 10 seconds)
    if samp_len > length:

        # if just want one sample
        if one_samp:
            start = random.randint(start_samp, end_samp - length)
            labels = [start]
        # if want all samples
        else:
            labels = np.arange(start_samp, end_samp, length)
    else:
        labels = [start_samp]

    dom_folder = os.path.join(c['folder_wav'], label, dom)
    os.makedirs(dom_folder, exist_ok=True)

    for start in labels:
        if min_size and min(end_samp, start + length) - start < min_size:
            print("skipped short sample: {:.2f}".format((min(end_samp, start + length) - start)/c['sr']))
            continue
        samp = y[start:min(end_samp, start + length)]
        samp_name = '{}_{}_{}_{}.wav'.format(dom, name_no_ext_no_us, label, idx)

        samp_path = os.path.join(dom_folder, samp_name)
        wavfile.write(samp_path, c['sr'], samp)
        idx += 1
        num_saved += 1

    return(idx, num_saved)

def save_wav_wrapper(dom, name_no_ext_no_us, cntr, y, row, length = MAX_SAMPS_PER_WAV, one_samp=False):

    num_saved = 0
    start_samp = int(row.start * c['sr'])
    end_samp = int(row.end * c['sr'])
    label = row.label

    #removes out the 'speech" from the name
    name_no_ext_no_us = name_no_ext_no_us[:-1] if ('uwct' in name_no_ext_no_us and 'speech' in name_no_ext_no_us) else name_no_ext_no_us

    if dom != 'flusense' and label not in c['label_types']:
        if 'bgsound' in label:
            print('skipping a cough with background sound')
        else:
            print("could not find label -", label)
        return(cntr, num_saved)

    idx, num_saved = save_wav(y, start_samp, end_samp, label, dom, name_no_ext_no_us, num_saved=num_saved,
                              idx=cntr[label], length=length, one_samp=one_samp)
    cntr[label] = idx
    return (cntr,num_saved)

def save_all_wavs(df, dom, name_no_ext_no_us, y):

    cntr = Counter()

    df_to_process = df.copy()
    df_to_process = df_to_process[df_to_process['label'] != 'rejected']

    if not('uwct' in name_no_ext_no_us and 'speech' in name_no_ext_no_us) and dom not in ['flusense', 'audioset']:

        for typ in ['silence', 'noise', 'speech', 'breath', 'other']:

            df_typ = df[df['label'] == typ]


            num_sec = [float(df_typ.iloc[i].end) - float(df_typ.iloc[i].start) for i in range(len(df_typ.index))]

            # Only run the below code if we have too much data
            if sum(num_sec) < 240 and len(num_sec)<240:
                continue

            num_samps_above_half = (np.array(num_sec) > .5).sum()
            num_to_save = 240

            if num_samps_above_half > 180:

                # if have enough samps, reduce short ones probability to 0
                num_sec = [0 if samp*c['sr'] < .5*config.MAX_SAMPS else samp for samp in num_sec]

            if num_samps_above_half < 240:

                # if don't have very many samples, store the short ones now
                for i in range(len(df_typ.index)):

                    if num_sec[i] > 0 and num_sec[i] <= config.MAX_SAMPS*2/c['sr']:
                        row = df_typ.iloc[i]
                        cntr, num_saved = save_wav_wrapper(dom, name_no_ext_no_us, cntr, y, row, length=int(config.MAX_SAMPS*2))
                        num_sec[i] = 0
                        num_to_save -= num_saved

                    if num_to_save < 80:
                        break

            # Give a minimum probability to all samps so all the probability doesn't go to 1 really long sample
            num_sec_max_derate = (1/num_to_save)*max(num_sec)
            num_sec = np.array([max(num_sec_max_derate,samp) if samp>0 else samp for samp in num_sec])

            # normalize so they become probabilities
            num_sec = num_sec/sum(num_sec)

            # pick which samps to use based on probs
            choices = np.random.multinomial(num_to_save,num_sec)

            # print([[i,num, '{:.2f} s'.format(df_typ.iloc[i].end-df_typ.iloc[i].start)] for i,num in enumerate(choices) if num>1])
            # get the samps
            for i, num in enumerate(choices):
                while (num >0):
                    row = df_typ.iloc[i]
                    # print(i, '{:.2f} s'.format(row.end-row.start))
                    cntr, num_saved = save_wav_wrapper(dom, name_no_ext_no_us, cntr, y, row, length=config.MAX_SAMPS, one_samp=True)
                    num-=1

            #remove these from processing the rest of the samples
            df_to_process = df_to_process[df_to_process['label'] != typ]

    for i in range(len(df_to_process.index)):
        row = df_to_process.iloc[i]
        cntr, num_saved = save_wav_wrapper(dom, name_no_ext_no_us, cntr, y, row)

    print(cntr)
    return(cntr)

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

        typ = 'all'

        if dom in ['respeaker4mic']:
            continue

        if not os.path.exists(c['files_csv']):
            df_files = pd.DataFrame([], columns=all_files_csv_columns)
        else:
            df_files = pd.read_csv(c['files_csv'])

        if (df_files['name'] == name_no_ext).any():
            print("File already preprocessed, skipping -", name_no_ext_no_us)
            continue

        print("Starting preprocessing on:", name)

        df, result = get_labels(dom, name_no_ext)
        if not result:
            continue

        y, result = get_audio(name, name_no_ext, dom)

        if not result:
            continue

        cntr = save_all_wavs(df, dom, name_no_ext_no_us, y)
        cntr_list = [cntr[label] if label in cntr.keys() else 0 for label in c['label_types']]

        # Everything that isn't in one of the standard categories gets thrown into "sound"
        cntr_list[c['label_types'].index('sound')] += sum([cnt for label, cnt in cntr.items() if label not in c['label_types']])

        df_files = df_files.append(pd.DataFrame([[dom, typ, name_no_ext]+cntr_list],columns=all_files_csv_columns),ignore_index=True)
        df_files.to_csv(c['files_csv'], index=False)

        print("Finished preprocessing on:", name)

def combine_intervals(intervals, comb_size=config.MAX_SAMPS, comb_derate=2, remove_short_sec=False, min_size=False, len_y=None):

    if min_size:
        assert(len_y is not None)

    HALF_SAMPS = int(np.ceil(min_size / 2))

    # Combine intervals
    intervals_new = np.array([intervals[0]])
    prev_end = intervals[0][1]
    for i, (start_samps, end_samps) in enumerate(intervals[1:]):
        if start_samps-prev_end < (comb_size/comb_derate):
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


    for dom_dict in dom_dict_list:

        dom = dom_dict['dom']

        # if dom == 'FSDKaggle':
        #     continue

        folder = os.path.join(c['folder_raw'], dom)
        path_meta = os.path.join(folder, dom_dict['meta'])
        folder_data = os.path.join(folder, dom_dict['data'])
        folder_data_coughs = os.path.join(folder_data, 'FSDKaggle2018.audio.train.coughs') if dom == 'FSDKaggle' else folder_data

        df = pd.read_excel(path_meta) if dom == 'FSDKaggle' else pd.read_csv(path_meta)
        labels = sorted(df.label.unique())

        for label in labels:

            if dom =='audioset' and label.lower() =='cough':
                continue

            name_sound = '{}_{}'.format(dom, label)
            df_files = pd.read_csv(c['files_csv'])
            if (df_files['name'] == name_sound).any():
                print("File already preprocessed, skipping -", name_sound)
                continue

            label_name_fixed = label.replace('_', '-').replace(' ','-').lower()

            df_label = df[df.label==label]

            if dom == 'FSDKaggle' and label == 'Cough':
                # we listened to these and many are not actually coughs. this gets them by their manual label
                df_label = df_label[df_label.cough == 1]

            idx_all = 0
            for i in range(len(df_label.index)):

                idx=0
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
                intervals = combine_intervals(intervals)

                for start_samps, end_samps in intervals:

                    # get rid of really short samples
                    if end_samps - start_samps < .15*c['sr']:
                        continue
                    idx, _ = save_wav(y, start_samps, end_samps, label_name_fixed, dom, name_no_ext_no_us, idx=idx)

                idx_all+=idx
                if i%10 == 0:
                    print(label, "\t| Files finished:", i, "of",len(df_label.index), "\t| Num samps:", idx_all)

            cntr_list = [0] * len(c['label_types'])
            cntr_list_lab = 'cough' if label == 'Cough' else 'sound'
            cntr_list[c['label_types'].index(cntr_list_lab)] = idx_all

            df_files = df_files.append(pd.DataFrame([[dom, label, name_sound]+cntr_list],columns=all_files_csv_columns),ignore_index=True)
            df_files.to_csv(c['files_csv'], index=False)

            print("Finished preprocessing on:", name_sound)

def preprocess_jotform():

    dom = 'jotform'
    folder_jotform = os.path.join(c['folder_raw'], 'jotform')
    jotform_email_to_id_csv = os.path.join(folder_jotform, 'jotform_email2id.csv')
    csv_columns = ['name']
    df_files = pd.read_csv(c['files_csv'])

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

    for u, dirs in user_dict.items():

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

        cntr = Counter()

        for d in dirs:

            label = d.split('_')[-1]

            # convert form to what the others have used
            label = 'throat-clearing' if label == 'clear' else label
            label = 'laughter' if label == 'laugh' else label

            idx = 0

            path_dir = os.path.join(folder_jotform, u, d)
            for f in os.listdir(path_dir):

                path = os.path.join(path_dir, f)

                y, result = get_wav(path, 'jotform', [], f)
                if not result:
                    print("Couldn't get wav", f)
                    continue

                y, _ = librosa.effects.trim(y, top_db=20)

                idx, num_saved = save_wav(y, 0, len(y), label, dom, name, idx=idx)

            cntr[label]=idx

        cntr_list = [cntr[label] if label in cntr.keys() else 0 for label in c['label_types']]
        # Everything that isn't in one of the standard categories gets thrown into "sound"
        cntr_list[c['label_types'].index('sound')] += sum([cnt for label, cnt in cntr.items() if label not in c['label_types']])

        df_files = df_files.append(pd.DataFrame([[dom, 'all', name] + cntr_list], columns=all_files_csv_columns),ignore_index=True)
        df_files.to_csv(c['files_csv'], index=False)

        print("Finished preprocessing:", name, cntr)

def preprocess_demand():
    folder_demand_raw = r'Z:\research\cough_count\data\raw\demand'

    for root, dirs, wavs in os.walk(folder_demand_raw):
        bname = os.path.basename(root)
        if not(bname.endswith('16k')):
            continue

        path_wav = os.path.join(root, dirs[0], 'ch01.wav')

        y, result = get_wav(path_wav, 'demand', [], bname)
        if not result:
            print("Get can't wav for:", bname)

        idx = 0

        save_wav(y, 0, len(y)-1, 'aug', 'demand', bname.replace('_', '-'), idx=idx, min_size=config.MAX_SAMPS)
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



if __name__ == '__main__':

    wavs = True
    csv = True
    folders = True

    # preprocess_wrapper(wavs=wavs,csv=csv,folders=folders) #coughsense, pediatric, southafrica, whosecough
    # preprocess_demand()

