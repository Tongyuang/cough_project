import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from config import config as c
import os
import pickle
from pathlib import Path
import tensorflow as tf
import time
import pandas as pd
import progressbar
from scipy.io import wavfile
from shutil import copyfile

from tensorflow.keras.models import load_model
from model_mobilenet import relu6
from tensorflow.keras.layers import DepthwiseConv2D
from model_samp_cnn import AudioVarianceScaling
from model_vggish import vggish_model
from cross_validation import get_file_list, get_typ_and_domain_probabilities_dicts
from metrics import F1Score
from vggish import mel_features
from vggish import vggish_params as vp

import config
from config import config as c
from config import config_data as c_d
from preprocess import get_domain
from wav_preprocess import mean_std_normalize
import os

label_file_header = ['start', 'end', 'label']
df_results_columns = ['dataset', 'tr_te','correct', 'FP', 'FN', 'sens', 'recall','spec', 'fa/hour', 'hours']
df_results_idx = 'fname'

folder_l_and_c = os.path.join(os.path.dirname(c_d['folder_data']), 'logs_and_checkpoints')
folder_label_predictions = os.path.join(c_d['folder_data'],'label_preds')
os.makedirs(folder_label_predictions, exist_ok=True)

NUM_MINS=5
COMBINE_COUGHS_SEC = .3
HALF = int(config.MAX_SAMPS / 2)
thresh=.5

model_idx = [0,0]

def setup_tflite_intepreter(interpreter, batch_size):

    # Set up the tflite model
    model_idx[0] = interpreter.get_input_details()[0]['index']
    model_idx[1] = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']

    # Adjust the input shape to take the batch size
    input_shape_full = [batch_size] + list(input_shape[1:])

    interpreter.resize_tensor_input(model_idx[0], input_shape_full)
    interpreter.allocate_tensors()

    return(interpreter)

def get_model_tflite(model_folder_name, model_name, batch_size, tflite=True):

    folder_model = os.path.join(os.path.dirname(c_d['folder_data']), 'logs_and_checkpoints', model_folder_name)
    path_model_h5 = os.path.join(folder_model, '{}.h5'.format(model_name))

    if not tflite:
        model = load_model(path_model_h5, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
                                           'AudioVarianceScaling':AudioVarianceScaling, 'F1Score':F1Score})
        return(model)
    else:
        path_model_tflite = os.path.join(folder_model, '{}.tflite'.format(model_name))
        if not os.path.exists(path_model_tflite):
            model = load_model(path_model_h5, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
                                               'AudioVarianceScaling':AudioVarianceScaling, 'F1Score':F1Score})
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            open(path_model_tflite, "wb").write(tflite_model)

        interpreter = tf.lite.Interpreter(model_path=path_model_tflite)

        interpreter = setup_tflite_intepreter(interpreter, batch_size=batch_size)

        return(interpreter)

def get_wav_batched(y, max_samps=config.MAX_SAMPS):

    HALF = int(max_samps / 2)
    wb = []
    for start in np.arange(0,len(y),HALF):
        new_samp = y[start:start+max_samps]
        if len(new_samp) == max_samps:
            wb += [new_samp]
    return(np.array(wb))

def get_wav_full_file_simple(f, max_samps=config.MAX_SAMPS):

    path = os.path.join(c['folder_wavfile'],f)
    _, y = wavfile.read(path)

    wb = get_wav_batched(y, max_samps=max_samps)

    return(y, wb)

def get_wav_full_file(f, truncate=False):

    trunc_suff = '_trunc' if truncate else ''

    name = Path(f).stem
    path_wavfile = os.path.join(c['folder_wavfile'], name + trunc_suff + '_{}khz_wavfile.wav'.format(int(c['sr']/1000)))

    # First load audio from numpy if exists, otherwise read wav and save to numpy
    if os.path.exists(path_wavfile):
        _,y = wavfile.read(path_wavfile)
        print("Loaded", name, "audio from wavfile.")
    else:
        y, _ = librosa.load(os.path.join(c['folder_raw'], Path(f).stem + '.flac'), sr=c['sr'])
        if len(np.shape(y)) > 1:
            y = np.mean(y, axis=1)
        if truncate:
            y = y[:16000 * 60 * NUM_MINS]
        # clip to max of 1/-1 to match normal wavs
        if max(abs(y)) > 1:
            np.clip(y, -1.0, 1.0, y)

        wavfile.write(path_wavfile, c['wr'], y)

    wb = get_wav_batched(y)

    return(y, wb)

def get_model_predictions(batch_size, model, wb, tflite):

    # print("Starting model predictions")

    confidences = []
    # rc = random_crop(config.MAX_SAMPS)
    msn = mean_std_normalize()
    # for samp in wb:
    #     print(samp.shape)
    #     break
    # wb = [msn(samp) for samp in wb]
    batches_all = np.arange(0, int(batch_size*np.ceil(len(wb)/batch_size)), batch_size)

    bar = progressbar.ProgressBar(maxval=len(batches_all)).start()
    for i, start in enumerate(batches_all):

        batch = np.array(wb[start:start+batch_size])

        # if last batch doesn't have enough samples, just add some zero padded samples to batch
        pad_size=0
        if batch.shape[0] < batch_size:
            pad_size = batch_size - batch.shape[0]
            batch = np.pad(batch,((0, pad_size), (0, 0)), 'constant')

        #get predictions
        if tflite:
            model.set_tensor(model_idx[0], batch.astype(np.float32))
            model.invoke()
            new_confidences = model.get_tensor(model_idx[1]).squeeze()
        else:
            new_confidences = model.predict(batch).squeeze()

        #take off padding
        if pad_size>0:
            new_confidences = new_confidences[:batch_size-pad_size]

        confidences += list(new_confidences)
        bar.update(i)
    print("\n")
    #print results
    # for i, classif in enumerate(classifications):
    #     if classif:
    #         print("{0:.2f} s - {1}".format(i*.48, classif))
    # print("\nFinished model predictions")

    return(confidences)


def get_label_file(f):

    name_no_ext = Path(f).stem
    name_no_ext = name_no_ext.replace('_16khz_wavfile', '')
    name_no_ext = name_no_ext.replace('_8khz_wavfile', '')
    path_label_rej = None
    df_rej = None

    if 'TDRS' in f:
       label_name = name_no_ext + '_new.label.txt2'
    elif get_domain(name_no_ext) == 'coughsense':
        label_name = name_no_ext + '.label.txt'
    else:
        label_name = name_no_ext + '.label.txt2'
    path_label = os.path.join(c['folder_raw'], label_name)

    if get_domain(name_no_ext) == 'coughsense':
        # print("loaded rejected")
        path_label_rej = os.path.join(c['folder_raw'], name_no_ext + '.label.txt')

    try:
        with open(path_label) as file:
            df = pd.read_csv(file, sep=r'\t', header=None, engine='python')
            df.columns = label_file_header
    except FileNotFoundError:
        print('ERROR - File:', label_name, 'could not be found')
        return (pd.DataFrame([]), pd.DataFrame([]))

    if path_label_rej != None:
        try:
            with open(path_label_rej) as file:
                df_rej = pd.read_csv(file, sep=r'\t', header=None, engine='python')
                df_rej.columns = label_file_header
        except FileNotFoundError:
            print('ERROR - Rejcted File:', label_name, 'could not be found')

    if isinstance(df_rej, pd.DataFrame):
        if 'rejected' not in df_rej['label'].values:
            df_rej = None
        else:
            df_rej = df_rej.loc[df_rej['label'] == 'rejected']

    return(df, df_rej)

def create_plot_labels(t_frames, t_frame_len, cough_segs, rejected_segs):
    labels = []
    for frame_start in t_frames:
        frame_end = frame_start + t_frame_len
        exit = False

        for seg in cough_segs:
            cough_start = seg[0]
            cough_end = seg[1]
            if ((frame_start >= cough_start and frame_end <= cough_end) or # whole frame is inside cough
                (frame_start <= cough_start and frame_end >= cough_end) or # whole cough is inside frame
                (cough_start <= frame_start and cough_end >= (frame_start + .3)) or # make sure have at least .3 s of cough in frame
                (cough_end >= frame_end and frame_end >= (cough_start + .3))): # make sure have at least .3 s of cough in frame

                labels+=[1]
                exit=True
                break

        if exit:
            continue

        for seg in rejected_segs:
            rej_start = seg[0]
            rej_end = seg[1]
            if ((frame_start >= rej_start and frame_end <= rej_end) or # whole frame is inside cough
                (frame_start <= rej_start and frame_end >= rej_end) or # whole cough is inside frame
                (rej_start <= frame_start and rej_end >= (frame_start + .3)) or # make sure have at least .3 s of cough in frame
                (rej_end >= frame_end and frame_end >= (rej_start + .3))): # make sure have at least .3 s of cough in frame
                labels+=[2]
                exit=True
                break

        if exit:
            continue
        labels+=[0]

    # print(['{:.2f}'.format(t) for t in t_frames])
    # for t, cough in zip(t_frames, labels):
    #     if cough:
    #         print(t, t+t_frame_len, cough)
    # print(cough_segs)
    return(np.array(labels))

def get_labels(f, t_frames, t_frame_len, truncate):

    df, df_rej = get_label_file(f)
    if len(df.index) < 1:
        return (False)#, False, False)
    cough_segs = []
    rejected_segs = []
    # cough_segs_comb = []
    for i in range(len(df.index)):
        row = df.iloc[i]
        if truncate and row.end >= NUM_MINS * 60:
            continue
        if 'cough' in row.label:
            cough_segs += [[row.start, row.end]]

            # # Combine cough labels within 300 ms of each other
            # if len(cough_segs_comb)>0 and (row.start - cough_segs_comb[-1][1]) < COMBINE_COUGHS_SEC:
            #     cough_segs_comb[-1][1] = row.end
            # else:
            #     cough_segs_comb += [[row.start, row.end]]

    if df_rej is not None:
        for i in range(len(df_rej.index)):
            row = df_rej.iloc[i]
            if truncate and row.end >= NUM_MINS * 60:
                continue
            rejected_segs += [[row.start, row.end]]

    labels = create_plot_labels(t_frames, t_frame_len, cough_segs, rejected_segs)
    # labels_comb = create_plot_labels(t_frames, cough_segs_comb)

    # print(np.argwhere(np.array(labels)>0))
    return(labels) #, labels_comb, cough_segs_comb)

def create_preds_label_file(preds_path, coughs, t_frames, t_frame_len):
    # preds_segs_comb = []

    with open(preds_path, 'w') as handle:
        in_cough = False
        start = 0
        end = 0
        for i, samp in enumerate(coughs):
            if samp:
                end = t_frames[i] + t_frame_len
                if in_cough:
                    continue
                else:
                    start = t_frames[i]
                    in_cough = True
            else:
                if in_cough:
                    handle.write("{}\t{}\tcough\n".format(start, end))
                    # if len(preds_segs_comb)>0 and (start -preds_segs_comb[-1][1]) < COMBINE_COUGHS_SEC:
                    #     preds_segs_comb[-1][1] = end
                    # else:
                    #     preds_segs_comb += [[start,end]]
                    in_cough = False
    print("Wrote predictions to", preds_path)
    return()#preds_segs_comb)

def get_results(f, df_path, label_tr_te, classifications, labels, t_frames):

    prev_result = 0  # 0 = nothing, 1 = tp, 2 = fp
    true_positive = 0
    false_positive = 0
    for i, classif in enumerate(classifications):
        if classif:
            if np.any(np.array(labels[i - 2:i + 3]) == 1):
                if prev_result == 1:
                    continue
                else:
                    true_positive += 1
                    prev_result = 1

            elif np.any(labels[i - 2:i + 3] == 2):
                prev_result = 0
                continue
            else:
                if prev_result == 2:
                    continue
                else:
                    false_positive += 1
                    prev_result = 2
        else:
            prev_result = 0

    prev_result = 0  # 0 = nothing, 1 = fn
    false_negative = 0
    for i, label in enumerate(labels):
        if label == 1:
            if np.all(classifications[i - 2:i + 3] == 0):  # false negative (missed cough)

                if prev_result == 1:
                    continue
                else:
                    false_negative += 1
                    prev_result = 1

            else:
                prev_result = 0
        else:
            prev_result = 0

    sens = true_positive / max(1, true_positive + false_negative)
    recall = true_positive / max(1, true_positive + false_positive)
    specificity = (1 - (false_negative / (len(classifications))))
    hours = t_frames[-1] / 60 / 60
    FA_per_hour = false_positive / hours
    print(f, "True_postive:", true_positive, "| # False Positives:", false_positive, "| False Negatives:", false_negative)
    print(f,
          "Sens: {:.2f}%, Recall: {:.2f}%, Spec: {:.2f}%, FA/h: {:.2f}, Hours:{:.2f}".format(100 * sens, 100 * recall, 100 * specificity, FA_per_hour,
                                                                            hours))
    # print("{}|{}|{}|{}|{:.2f}|{:.2f}|{:.2f}".format(f, correct, false_positives, false_negatives, 100*sens, 100*specificity,
    #                                                 FA_per_hour))

    df = pd.read_csv(df_path, index_col=0)
    df.index.name = df_results_idx
    df_row = [[get_domain(f), label_tr_te, true_positive, false_positive, false_negative, sens, recall, specificity, FA_per_hour, hours]]
    df.append(pd.DataFrame(df_row, columns=df_results_columns, index=[Path(f).stem])).to_csv(df_path)

def test_predict_full_file(model_folder_name, model_name, batch_size=32, tflite=False, truncate=False):

    CV = model_folder_name.split('_')[1]
    assert (CV.split('-')[0] == 'CV')
    CV = int(CV.split('-')[1])

    files_te = []
    df_files = pd.read_excel(c['files_cv_split'],index_col=0)
    for dom in ['coughsense']:#,'whosecough']:
        df_dom = df_files[df_files['domain']==dom]
        files_te += list(df_dom[df_dom['CV']==CV].index)
        files_te = [f for f in files_te if 'speech' not in f] # removes the whosecough "speech" samples

    model = get_model_tflite(model_folder_name, model_name, batch_size, tflite=tflite)

    model_name = '_'.join([model_folder_name,model_name])
    preds_folder = os.path.join(folder_label_predictions, model_name)
    os.makedirs(preds_folder, exist_ok=True)

    trunc_suff = '_trunc' if truncate else ''

    # Get results dataframe
    df_path = os.path.join(preds_folder, 'df_results{}.csv'.format(trunc_suff))
    # Save an empty data frame if none exists
    if not(os.path.exists(df_path)):
        pd.DataFrame([],columns=df_results_columns).to_csv(df_path)

    for f in files_te:

        preds_name = Path(f).stem + '{}_preds.label.txt'.format(trunc_suff)
        preds_path = os.path.join(preds_folder, preds_name)

        if not(truncate) and os.path.exists(preds_path):
            print("Predictions path already exists for", f, "- skipping")
            continue

        conf_path = os.path.join(preds_folder, Path(f).stem + '{}_conf.pkl'.format(trunc_suff))
        if os.path.exists(conf_path):
            with open(conf_path, 'rb') as handle:
                confidences = pickle.load(handle)
            print("Loaded confidences/classifications from pickle")
        else:

            y, wb = get_wav_full_file(f, truncate=truncate)

            confidences = get_model_predictions(batch_size, model, wb, tflite)

            with open(conf_path,'wb') as handle:
                pickle.dump(confidences, handle)
            # print("Saved confidences/classifications to pickle")

        classifications = np.array([1 if f > thresh else 0 for f in confidences])

        t_frames = np.arange(0,int(len(confidences)*HALF), HALF)
        t_frames = librosa.samples_to_time(t_frames,sr=c['sr'])
        t_frame_len = config.MAX_SAMPS/c['sr']

        # Get the ground truth labels
        # labels, labels_comb, cough_segs_comb = get_labels(f, t_frames, t_frame_len, truncate)
        labels = get_labels(f, t_frames, t_frame_len, truncate)
        if labels is False:
            print("Couldn't find label file for:", f, "skipping")
            continue

        # for t, classif in zip(t_frames, classifications):
        #     if classif:
        #         print(t, t+t_frame_len, classif)

        # Create audacity predictions label file
        create_preds_label_file(preds_path, classifications, t_frames, t_frame_len)

        # labels_preds_comb = create_plot_labels(t_frames, preds_segs_comb)

        get_results(f, df_path, 'test', classifications, labels, t_frames)

def test_predict_full_file_wrapper():

    truncate = False
    tflite = False

    # models = [('mobilenet_CV-0_no-name_0','model.750-0.33'),
    #           ('mobilenet-simple_CV-0_no-name_0', 'model.750-0.56'),
    #           ('conv-model_CV-0_no-name_3','model.1300-0.51'),
    #           ('conv-model_CV-0_new-llf-pretrain-reverb-no-msn_1', 'model.500-0.24')]
    # idx = 3

    for model_folder_name in ['sample-cnn_CV-0_8khz-res2-7L-64F-coughsense-reverb_0']:

        # for model_id in ['1000-0.27']:
        #     model_name = 'model.{}'.format(model_id)
        for model_name in [f for f in os.listdir(os.path.join(folder_l_and_c, model_folder_name)) if f.endswith('.h5')]:
            model_name = Path(model_name).stem
            test_predict_full_file(model_folder_name, model_name, tflite=tflite, truncate=truncate)

def get_eval_files_ondevice(model_type, CV, sr, max_samps):

    X = []
    y = []
    HALF = int(max_samps / 2)

    _, file_list = get_file_list(CV=CV, TEST=True)

    print("Loading Full Files")

    files = sorted(list(set([f.split('_')[1]+'_{}khz_wavfile.wav'.format(int(c['sr']/1000)) for f in file_list['cough']['coughsense']])))
    print(files)

    for f in files:
        _, wb = get_wav_full_file_simple(f, max_samps=max_samps)

        t_frames = np.arange(0, int(len(wb) * HALF), HALF)
        t_frames = librosa.samples_to_time(t_frames, sr=sr)
        t_frame_len = max_samps / sr

        labels = get_labels(f, t_frames, t_frame_len, truncate=False)

        # remove all the samples on the sides of each cough. We don't have a reliable way of confirming
        # whether the model should classify it as a cough so better just to throw it out
        idxs = np.arange(0,len(labels))
        idxs_new = []
        idxs_rej = []
        for i in idxs:
            #remove all rejected labels
            if labels[i] == 2:
                idxs_rej += [i]
                continue
            #remove all non-coughs that are 1 away from a cough or a rejected
            elif labels[i] == 0 and ((i-1>=0 and labels[i-1]in [1]) or (i+1<=len(labels)-1 and labels[i+1] in [1])):
                idxs_rej += [i]
                continue
            idxs_new += [i]

        wb = wb[idxs_new]
        labels = labels[idxs_new]

        X += list(wb)
        y += list(labels)

    len_bs = int(np.floor(len(X)/c['batch_size'])*c['batch_size'])

    X = np.array(X[:len_bs])
    y = np.array(y[:len_bs])

    print("Finished loading full files")

    if model_type == 'sample-cnn':
        X = np.expand_dims(X, -1)

    return(X,y)


def create_eval_files(file_list, CV, max_samps, sr, vggish=False):

    vggish_suff = '_vggish' if vggish else ''
    HALF = int(max_samps / 2)

    files = sorted(list(set([f.split('_')[1] + '_{}khz_wavfile.wav'.format(int(sr / 1000)) for f in file_list['cough']['coughsense']])))

    cnt = 0
    for f in files:
        y, wb = get_wav_full_file_simple(f, max_samps)

        t_frames = np.arange(0, int(len(wb) * HALF), HALF)
        t_frames = librosa.samples_to_time(t_frames, sr=sr)
        t_frame_len = max_samps / sr

        labels = get_labels(f, t_frames, t_frame_len, truncate=False)

        # remove all the samples on the sides of each cough. We don't have a reliable way of confirming
        # whether the model should classify it as a cough so better just to throw it out
        idxs = np.arange(0, len(labels))
        idxs_new = []
        idxs_rej = []
        for i in idxs:
            # remove all rejected labels
            if labels[i] == 2:
                idxs_rej += [i]
                continue
            # remove all non-coughs that are 1 away from a cough or a rejected
            elif labels[i] == 0 and (
                    (i - 1 >= 0 and labels[i - 1] in [1]) or (i + 1 <= len(labels) - 1 and labels[i + 1] in [1])):
                idxs_rej += [i]
                continue
            idxs_new += [i]

        wb = wb[idxs_new]
        if vggish:
            wb_new = []
            for wav in wb:
                samp = mel_features.log_mel_spectrogram(wav, audio_sample_rate=c['sr'], log_offset=vp.LOG_OFFSET,
                                                        window_length_secs=vp.STFT_WINDOW_LENGTH_SECONDS,
                                                        hop_length_secs=vp.STFT_HOP_LENGTH_SECONDS,
                                                        num_mel_bins=vp.NUM_MEL_BINS, lower_edge_hertz=vp.MEL_MIN_HZ,
                                                        upper_edge_hertz=vp.MEL_MAX_HZ)
                wb_new.append(samp)
            wb = np.expand_dims(np.array(wb_new), axis=-1)

        labels = labels[idxs_new]

        folder_cv = os.path.join(c_d['folder_data'], 'evaluate_pkl', '{}khz'.format(int(sr/1000)), str(max_samps),'CV{}{}'.format(CV, vggish_suff))
        os.makedirs(folder_cv, exist_ok=True)
        print(folder_cv)

        batch_size = 32
        bar = progressbar.ProgressBar(maxval=int(np.floor(len(labels)/batch_size))).start()
        for start in np.arange(0,len(labels)-batch_size,batch_size):
            path_pkl = os.path.join(folder_cv, 'eval_{}khz_cv{}_{}.pkl'.format(int(sr / 1000), CV, cnt))
            end = start + batch_size
            wavs = wb[start:end]
            labs = labels[start:end]
            with open(path_pkl, 'wb') as handle:
                pickle.dump([wavs, labs], handle)
            bar.update(int(start/batch_size))
            cnt += 1

        # for i, (w, l) in enumerate(zip(wb,labels)):
        #     path_wav = os.path.join(folder_cv,'eval_{}khz_cv{}_label-{}_{}.wav'.format(int(sr/1000), CV,l,cnt))
        #     wavfile.write(path_wav, sr, w)
        #     bar.update(i)
        #     cnt+=1

        print("Finished", f)

class DataGeneratorFullFile(tf.keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, model_type, CV, sr=c['sr'], max_samps=config.MAX_SAMPS):

        'Initialization'
        vggish_suff = '_vggish' if  model_type == 'vggish' else ''
        self.model_type = model_type
        self.idx = 0
        sr_str = '{}khz'.format(int(sr / 1000))
        self.folder_cv = os.path.join(c_d['folder_data'], 'evaluate_pkl', sr_str, str(max_samps), 'CV{}{}'.format(CV, vggish_suff))
        files = np.array([[int(Path(f).stem.split('_')[-1]), f]for f in os.listdir(self.folder_cv)], dtype=object)
        self.files = files[np.argsort(files[:, 0])][:,1]

        # # remove the extras that can't fill full batch
        # self.len_bs = int(np.floor(len(self.files)/c['batch_size']))
        # self.files = self.files[:int(self.len_bs*c['batch_size'])]
        #
        # # get the labels
        # self.labels = [int(Path(f).stem.split('_')[3].split('-')[-1]) for f in self.files]

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return self.len_bs
        return len(self.files)

    def tflite_converter_get_item(self,batch_size):

        f = self.files[np.random.choice(range(self.__len__()))]
        path = os.path.join(self.folder_cv, f)
        with open(path, 'rb') as handle:
            X, y = pickle.load(handle)

        idxs = np.random.permutation(range(len(X)))[:batch_size]
        X = X[idxs]

        if self.model_type == 'sample-cnn':
            X = np.expand_dims(X, -1)

        return X

    def __getitem__(self, index):


        f = self.files[self.idx]
        path = os.path.join(self.folder_cv, f)
        with open(path, 'rb') as handle:
            X, y = pickle.load(handle)

        # files = self.files[self.idx:self.idx+c['batch_size']]
        # X = []
        # for f in files:
        #     try:
        #        _, wav = wavfile.read(os.path.join(self.folder_cv, f))
        #     except:
        #         print("Couldn't get file:", f)
        #         continue
        #     X.append(wav)
        # y = self.labels[self.idx:self.idx+c['batch_size']]
        #
        # while (len(X)<c['batch_size']):
        #     X.append(X[-1])
        #     y.append(y[-1])
        #
        # X = np.array(X)
        # y = np.array(y)

        if self.model_type == 'sample-cnn':
            X = np.expand_dims(X, -1)

        self.idx += 1
        # self.idx += c['batch_size']

        return X, y

def test_predict_from_model_evaluate(predict=False):

    c_d['folder_data'] = r'C:\Users\mattw12\Documents\Research\cough_count\data'
    model_folder = r'sample-cnn_CV-0_8khz-res2-7L-64F-coughsense-reverb_0'
    epoch = 700
    sr=8000
    model_path = os.path.join(folder_l_and_c, model_folder,'model.{}.h5'.format(epoch))
    folder_save = os.path.join(r'C:\Users\mattw12\Documents\Research\cough_count\results\evaluate',model_folder)
    folder_fp = os.path.join(folder_save,'incorrect_coughs')
    folder_fn = os.path.join(folder_save, 'missed_coughs')
    os.makedirs(folder_fp, exist_ok=True)
    os.makedirs(folder_fn, exist_ok=True)

    dg = DataGeneratorFullFile('sample-cnn', 0)
    model = load_model(model_path, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
                                    'AudioVarianceScaling': AudioVarianceScaling, 'F1Score': F1Score})
    bar = progressbar.ProgressBar(maxval=len(dg)-1).start()
    cnt_fn = 0
    cnt_fp = 0
    for j in range(len(dg)-1):
        X, y = dg.__getitem__(0)
        res = model.predict(X).squeeze()
        res_cls = np.array([1 if r > .5 else 0 for r in res])
        idx_wrong = np.squeeze(np.argwhere(res_cls != y),-1)
        for i in idx_wrong:
            folder_save = folder_fn if y[i] else folder_fp
            file_save = 'cough_false_negative_{}.wav'.format(cnt_fn) if y[i] else'cough_false_positive_{}.wav'.format(cnt_fp)
            new_path = os.path.join(folder_save, file_save)
            wavfile.write(new_path, sr, X[i])

            if y[i]:
                cnt_fn +=1
            else:
                cnt_fp+=1

        bar.update(j)

    raise




    model_folder_name = 'conv-model_CV-0_pretrain_reverb_bs1_0' #'mobilenet-simple_CV-0_pretrain_reverb_bs-none_0' #'sample-cnn_CV-0_basic_7L_16F_1'
    sr = 8000

    model_type = model_folder_name.split('_')[0]
    CV = int(model_folder_name.split('_')[1].split('-')[-1])
    max_samps = 13122 if model_type == 'sample-cnn' else 15648
    max_samps = int(max_samps*sr/16000)
    model_folder = os.path.join(folder_l_and_c, model_folder_name)

    columns = ['name', 'f1', 'recall','precision']
    df = pd.DataFrame([], columns=columns)

    epochs = []
    f1s = []

    files = np.array([[int(f.split('.')[1]), f] for f in os.listdir(os.path.join(folder_l_and_c, model_folder_name)) if f.endswith('.h5')], dtype=object)
    files = files[np.argsort(files[:,0])][::-1]
    # # ds = fixed_file_eval_ds(CV=0)

    if predict:
        dg = DataGeneratorFullFile(model_type, CV)
    else:
        X,y = get_eval_files_ondevice(model_type, CV, sr, max_samps)

    for i, model_name in enumerate(files[:,1]):

        epoch = int(model_name.split('.')[1])
        path_model_h5 = os.path.join(model_folder, model_name)
        if model_folder_name.split('_')[0] == 'vggish':
            model = vggish_model()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', F1Score()])
            model.load_weights(path_model_h5)
        else:
            model = load_model(path_model_h5, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
                                                   'AudioVarianceScaling':AudioVarianceScaling, 'F1Score':F1Score})
        if predict:
            X,y =dg.__getitem__(0)

        res = model.evaluate(x=X, y=y)
        prec = res[2]
        recall = res[3]
        f1 = 2*prec*recall/(prec+recall)

        epochs.append(epoch)
        f1s.append(f1*100)
        print(model_name+"\t|", "F1: {:.2f}%".format(f1*100))

        df = df.append(pd.DataFrame([[epoch, f1, recall, prec]], columns=columns), ignore_index=True)

    df_path = os.path.join(os.path.dirname(c_d['folder_data']),'results','eval_results_'+model_folder_name+'.csv')
    if os.path.exists(df_path):
        os.remove(df_path)
    df.to_csv(df_path,index=False)

    plt.figure()
    plt.plot(epochs, f1s)
    plt.xlabel('Epoch')
    plt.ylabel('F1 (%)')
    plt.show()

if __name__ == '__main__':

    from train import F1Score

    dg = DataGeneratorFullFile('vggish',0)
    X,y = dg.__getitem__(0)
    print(X.shape, y.shape)
    raise
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #use cpu in case using gpu to train
    # test_predict_full_file_wrapper()
    # check_coughs_file()
    # test_predict_from_model_evaluate()

    # model=vggish_model()
    # model.load_weights(r'Z:\research\cough_count\logs_and_checkpoints\vggish_CV-0_no-name_1_test_3\model.001.h5')
    # model.summary()

    # checkpoint_path = os.path.join(os.path.dirname(c_d['folder_data']), 'vggish_model.ckpt')
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)

    CV=0
    files_tr, files_te = get_file_list(CV=CV, TEST=True)
    for sr, max_samps in [(16000, 15648), (8000, int(13122/2)), (16000, 13122), (8000, int(15648/2))]:
        create_eval_files(files_te, CV=CV, sr=sr, max_samps=max_samps, vggish=True)


###################
###################

# class DataGeneratorFullFile(tf.keras.utils.Sequence):
#
#     'Generates data for Keras'
#     def __init__(self, model_type, file_list, CV):
#
#         'Initialization'
#         self.model_type = model_type
#         self.idx = 0
#         self.batch_size = 32
#         self.setup(file_list, CV)
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.y)/self.batch_size))
#
#     def setup(self, file_list, CV):
#
#         # folder_wb = os.path.join(c_d['folder_data'], 'full_file_wavs_batched')
#         # path = os.path.join(folder_wb, 'CV{}.pkl'.format(CV))
#         # if os.path.exists(path):
#         #     with open(path, 'rb') as handle:
#         #         self.X, self.y = pickle.load(handle)
#         # else:
#         self.X = []
#         self.y = []
#         print("Loading Full Files")
#
#         files = sorted(list(set([f.split('_')[1]+'_{}khz_wavfile.wav'.format(int(c['sr']/1000)) for f in file_list['cough']['coughsense']])))
#         for f in files:
#             y, wb = get_wav_full_file_simple(f)
#
#             t_frames = np.arange(0, int(len(wb) * HALF), HALF)
#             t_frames = librosa.samples_to_time(t_frames, sr=c['sr'])
#             t_frame_len = config.MAX_SAMPS / c['sr']
#
#             labels = get_labels(f, t_frames, t_frame_len, truncate=False)
#
#             # remove all the samples on the sides of each cough. We don't have a reliable way of confirming
#             # whether the model should classify it as a cough so better just to throw it out
#             idxs = np.arange(0,len(labels))
#             idxs_new = []
#             idxs_rej = []
#             for i in idxs:
#                 #remove all rejected labels
#                 if labels[i] == 2:
#                     idxs_rej += [i]
#                     continue
#                 #remove all non-coughs that are 1 away from a cough or a rejected
#                 elif labels[i] == 0 and ((i-1>=0 and labels[i-1]in [1]) or (i+1<=len(labels)-1 and labels[i+1] in [1])):
#                     idxs_rej += [i]
#                     continue
#                 idxs_new += [i]
#
#             # folder = r'Z:\research\cough_count\pics'
#             # import matplotlib.pyplot as plt
#             # for i, cough in enumerate(wb[idxs_rej]):
#             #     fig = plt.figure()
#             #     plt.plot(cough)
#             #     plt.ylim([-1, 1])
#             #     wavfile.write(os.path.join(folder, 'cough_{}.wav'.format(i)), c['sr'], cough)
#             #     plt.savefig(os.path.join(folder, 'cough_{}.png'.format(i)))
#             #     # plt.show()
#             #     plt.close(fig)
#
#             wb = wb[idxs_new]
#             labels = labels[idxs_new]
#
#             self.X += list(wb)
#             self.y += list(labels)
#
#         len_bs = int(np.floor(len(self.X)/c['batch_size'])*c['batch_size'])
#
#         self.X = np.array(self.X[:len_bs])
#         self.y = np.array(self.y[:len_bs])
#         # with open(path, 'wb') as handle:
#         #     pickle.dump([self.X, self.y], handle)
#
#         print("Finished loading full files")
#
#         if self.model_type == 'sample-cnn':
#             self.X = np.expand_dims(self.X, -1)
#
#     def __getitem__(self, index):
#
#         start = int(self.idx*self.batch_size)
#         end = int((self.idx+1) * self.batch_size)
#
#         return self.X[start:end], self.y[start:end]

#My attempt at using at dataset for the evaluate files
#def get_dataset(CV):
#     print("started getting files")
#     folder_cv = os.path.join(c_d['folder_data'], 'evaluate', '{}khz'.format(int(c['sr'] / 1000)),'CV{}'.format(CV))
#     files = [os.path.join(folder_cv, f) for f in os.listdir(folder_cv) if f.endswith('.wav')]
#     labels = [int(Path(f).stem.split('_')[4].split('-')[-1]) for f in os.listdir(folder_cv) if f.endswith('.wav')]
#     ds1 = tf.data.Dataset.from_tensor_slices(files)
#     ds2 = tf.data.Dataset.from_tensor_slices(labels)
#     # files_cv = os.path.join(c_d['folder_data'], 'evaluate', '{}khz'.format(int(c['sr'] / 1000)),'CV{}'.format(CV),'*.wav')
#     # ds = tf.data.Dataset.list_files(files_cv)
#     print("got files")
#     return tf.data.Dataset.zip((ds1,ds2))
#
# def load_audio(file_path, label):
#     file_path = file_path.numpy()
#     print(file_path.numpy())
#     # bytes.decode(file_path)
#     _, audio = wavfile.read(file_path)
#     # label = int(Path(file_path).stem.split('_')[4].split('-')[-1])
#     return audio, label
#
# def set_shapes(audio, label):
#     print(len(audio))
#     print(type(audio[0]))
#     audio.set_shape((config.MAX_SAMPS,))
#     label.set_shape([])
#     return audio, label
#
# def fixed_file_eval_ds(CV, batch_size=64):
#     ds = get_dataset(CV)
#     # Randomly shuffle (file_path, label) dataset
#     #ds = ds.shuffle(buffer_size=shuffle_buffer_size)
#     ds = ds.cache()
#     # Load and decode audio from file paths
#     ds = ds.map(lambda audio, label: set_shapes(tf.py_function(load_audio, [audio, label], [tf.string, tf.int32])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     # Repeat dataset forever
#     ds = ds.repeat()
#     # Prepare batches
#     ds = ds.batch(batch_size)
#     # Prefetch
#     ds = ds.prefetch(buffer_size= tf.data.experimental.AUTOTUNE)
#     return ds