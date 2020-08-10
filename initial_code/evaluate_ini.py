import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from config import config as c
import os
import pickle
from pathlib import Path
import tensorflow as tf

from tensorflow.python.keras.models import load_model, model_from_json
# from tensorflow.keras.models import load_model, model_from_json

import config
from config import config as c
from config import config_data as c_d
from preprocess import get_domain
from data_generator import DataGenerator

def load_trained_model(vggish=False,tflite=False,CV=2,batch_size=32,skip_frames=1):
    global model_idx

    if tflite:
        if config.VGGISH:
            if CV == 0:
                model_name = '2020.04.26_22-19-04_vggish_CV0.330-0.09.tflite'
            elif CV == 1:
                model_name = '2020.04.26_22-16-56_vggish_CV1.290-0.09.tflite'
            elif CV==2:
                model_name = 'model.220_vggish_cv2_pt4thconv.tflite' if vggish else 'model_vggish_cv2.1757-0.21.tflite'
            else:
                raise
        else:
            model_name = "model_conv.133-0.06.tflite"  # "model_conv_rs4mic.144-0.09.tflite" #"model_conv_rs4mic.144-0.09.tflite" #"model_conv.133-0.06.tflite"
        load_path = os.path.join('pretrained_models', model_name)


        print("MODEL NAME:", model_name)
        # Set up the tflite model
        interpreter = tf.lite.Interpreter(model_path=load_path)
        interpreter = setup_tflite_intepreter(interpreter, batch_size)

        return (interpreter, model_name)

    else:
        if config.VGGISH:
            model_name = 'model.220_vggish_cv2_pt4thconv.hdf5' if vggish else 'model_vggish_cv2.1757-0.21.hdf5' #'model.220_vggish_cv2_pt4thconv.hdf5'
        else:
            model_name = 'model_conv.133-0.06.hdf5' #'model_conv_rs4mic.144-0.09.hdf5' #
        load_path = os.path.join('pretrained_models', model_name)
        model = load_model(load_path)
        return(model, model_name)


def get_framed_input(S, as_list=False):
    S_framed = [] if as_list else None
    for k, j in enumerate(np.arange(0, S.shape[1], 1)):
        end = S.shape[1] if S.shape[1] - j < 19 else j + 19
        start = end - 19 if S.shape[1] - j < 19 else j
        S_framed_new = S[:, start:end].transpose()
        if as_list:
            S_framed += [S_framed_new]
        else:
            S_framed = np.dstack((S_framed,S_framed_new)) if S_framed is not None else np.expand_dims(S_framed_new,-1)
            S_framed = np.transpose(S_framed,(2,0,1))
    return(S_framed)

def get_framed_input_wrapper(samps, name_no_ext):

    eval_folder =os.path.join(c_d['folder_data'], 'evaluate')
    os.makedirs(eval_folder,exist_ok=True)
    path = os.path.join(eval_folder, 'Sframed_{}.pkl'.format(name_no_ext))

    if os.path.exists(path):
        with open(path, 'rb') as handle:
            S_framed_all = pickle.load(handle)
        # print("Loaded framed features from pickle")
    else:
        # print("Starting framing")
        S_framed_all = []
        for samp in samps:
            S = samp[0]
            # if S.shape[1] > 32 * 100:
            #     S = S[:, :32 * 100 + 5]
            #     print("clipping")
            S_framed_all += get_framed_input(S, as_list = True)
            # S_framed_all = np.concatenate((S_framed_all, S_framed),
            #                               axis=0) if S_framed_all is not None else S_framed
            # if S_framed_all.shape[0] > (32 * 100):
            #     print("Too long, breaking")
            #     break
            # print(S_framed_all.shape)

        S_framed_all = np.expand_dims(S_framed_all, -1)
        # print(len(S_framed_all))

        # np.save(path, S_framed_all)
        with open(path, 'wb') as handle:
            pickle.dump(S_framed_all, handle)
        # print("Done framing and saved to pickle")

    return(S_framed_all)


def test_predict():

    tflite = False
    batch_size = 32
    CV=2
    OUTPUT_DOMAINS = True

    model = load_trained_model(tflite=tflite, batch_size=batch_size)

    files_speech_tr, files_speech_te, file_dict_silence_tr, file_dict_silence_te, files_noise_tr, files_noise_te,\
        files_sound_tr, files_sound_te = get_files_sil_noise(CV=CV)

    tr_generator = DataGenerator(TRAIN=True, CV=CV, file_dict_silence=file_dict_silence_tr, files_noise=files_noise_tr,
                                  files_speech=files_speech_tr, files_sound=files_sound_tr,
                                  frames=c['frames'], nmels=c['nmels'], n_classes=len(c['domains_devices']),
                                 OUTPUT_DOMAINS=OUTPUT_DOMAINS, BY_DEVICE=True)

    val_generator = DataGenerator(TRAIN=False, CV=CV, file_dict_silence=file_dict_silence_te, files_noise=files_noise_te,
                                   files_speech=files_speech_te, files_sound=files_sound_te,
                                   frames=c['frames'], nmels=c['nmels'], n_classes=len(c['domains_devices']),
                                  OUTPUT_DOMAINS=OUTPUT_DOMAINS, BY_DEVICE=True)

    doms = c['domains_devices']
    doms_len = len(doms)
    results_dict = {}
    blank_dict = {'acc':np.zeros(doms_len), 'cnt':np.zeros(doms_len), 'missed':{}}
    types = ['cough', 'sn_oth_br', 'speech', 'noise', 'sound', 'silence']
    # for r in range(2):
    #     X, y = tr_generator.__getitem__(0)
    X, y = tr_generator.__getitem__(0)
    print(X.shape, len(y))
    print(y.reshape(-1,3))
    raise
    # domains = np.argmax(y,axis=1)
    # domains_names = [c['domains_devices'][f] for f in domains]
    # print(domains_names)
    results = model.predict(X)
    print(['{:.2f}'.format(f*100) for f in results.squeeze()])

    types = ['cough', 'sn_oth_br', 'speech', 'noise', 'human+other', 'audioset_music', 'laughs_audioset', 'laughs_fsd',
             'sneezes', 'loud_sounds', 'silence']
    types_dict = {'cough':10, 'sn_oth_br':2, 'speech':4, 'noise':1, 'human+other':3, 'audioset_music':1,'laughs_audioset':1,
             'laughs_fsd':1,'sneezes':1,'loud_sounds':3, 'silence':5}
    for typ in types:
        results_dict[typ] = blank_dict.copy()

    start = 0
    for typ in types:
        end = config.num_per_batch_tr
        results_dict['typ']

def test_predict_FSDKaggle():

    tflite = True
    batch_size = 10 if tflite else 32

    folder = c['folder_pkl']
    files = sorted([f for f in os.listdir(folder) if ('FSDKaggle' in f) and f.endswith('.pkl') ])  #

    model = load_trained_model(tflite=tflite, batch_size=batch_size)

    for cnt, f in enumerate(files):

        # if cnt<3:
        #     print("skipping", f)
        #     continue

        name_no_ext = Path(f).stem
        sound = '_'.join(name_no_ext.split('_')[1:]) if 'coughs' in name_no_ext else '_'.join(name_no_ext.split('_')[2:])
        # print("Starting:", sound)

        with open(os.path.join(folder, f), 'rb') as pkl_file:
            pkl_dict, pkl_counter = pickle.load(pkl_file)

        typ = 'cough' if 'coughs' in name_no_ext else 'sound'

        S_framed_all = get_framed_input_wrapper(pkl_dict[typ], name_no_ext)

        label = 1 if typ == 'cough' else 0

        results_arr = []
        for i in np.arange(0, len(S_framed_all), batch_size):
            if len(S_framed_all) - i < batch_size:
                continue
            X = S_framed_all[i:i + batch_size]

            y = np.array([label] * X.shape[0])
            if tflite:
                model.set_tensor(model_idx[0], X.astype(np.float32))
                model.invoke()
                result = model.get_tensor(model_idx[1]).squeeze()
                acc = np.mean(np.array([1 if r > .5 else 0 for r in result]))
                result = [0, acc] if typ == 'cough' else [0, 1 - acc]
            else:
                result = model.test_on_batch(X, y)
            # print(result)

            results_arr += [result]

        results = np.mean(np.array(results_arr), axis=0)
        print(sound, 'loss: {:.4f}, acc: {:.2f}%'.format(results[0], results[1] * 100))
        # if cnt > 3:
        #     break


def test_predict_rs4mic():

    tflite = True
    batch_size = 10 if tflite else 32

    folder = c['folder_pkl']
    files = [f for f in os.listdir(folder) if ('respeaker' in f) and f.endswith('.pkl')] #'respeaker' in f or

    for f in files:

        subject = f.split('_')[-1].split('.')[0]
        # print("Starting:", subject)

        with open(os.path.join(folder,f), 'rb') as pkl_file:
            pkl_dict, pkl_counter = pickle.load(pkl_file)

        model = load_trained_model(tflite=tflite, batch_size=batch_size)

        for typ in c['label_types']:
            if len(pkl_dict[typ])<1:
                continue
            S_framed_all = None
            for samp in pkl_dict[typ]:
                S = samp[0]
                if S.shape[1] > 32*100:
                    S = S[:,:32*100+5]
                S_framed = get_framed_input(S)
                S_framed_all = np.concatenate((S_framed_all, S_framed), axis=0) if S_framed_all is not None else S_framed
                if S_framed_all.shape[0]>(32*100):
                    break
                # print(S_framed_all.shape)

            S_framed_all = np.expand_dims(S_framed_all,-1)

            label = 1 if typ =='cough' else 0

            results_arr = []
            for i in np.arange(0,S_framed_all.shape[0], batch_size):
                if S_framed_all.shape[0]-i < batch_size:
                    continue
                X = S_framed_all[i:i+batch_size]
                y = np.array([label] * X.shape[0])
                if tflite:
                    model.set_tensor(model_idx[0], X.astype(np.float32))
                    model.invoke()
                    result = model.get_tensor(model_idx[1]).squeeze()
                    acc = np.mean(np.array([1 if r > .5 else 0 for r in result]))
                    result = [0, acc] if typ == 'cough' else [0,1-acc]
                else:
                    result = model.test_on_batch(X,y)

                results_arr += [result]

            results = np.mean(np.array(results_arr), axis=0)
            print(subject, typ, 'loss: {:.4f}, acc: {:.2f}%'.format(results[0],results[1]*100))

def test_predict_4_mic_array():

    DOM_ADAPT = False

    path_pkl_dict = os.path.join(c_d['folder_data'], '4mic_array_test_pkl_dict.pkl')

    with open(path_pkl_dict, 'rb') as pkl_file:
        pkl_dict, pkl_counter = pickle.load(pkl_file)

    model = load_trained_model()

    batch_size = 32
    for typ in ['silence', 'speech', 'laughter','cough']:

        spec_framed = pkl_dict[typ][0][0]
        # print("starting", typ, "spec_size:",spec_framed.shape)

        label = 1 if typ =='cough' else 0

        results_arr = []
        for i in np.arange(0,spec_framed.shape[0],batch_size):
            X = spec_framed[i:i+batch_size]
            y = np.array([label] * X.shape[0])
            result = model.test_on_batch(X,y)

            result = list(np.array(result)[np.array([1,3])]) if DOM_ADAPT else result
            results_arr += [result]

        results = np.mean(np.array(results_arr), axis=0)
        print(typ, 'loss: {:.4f}, acc: {:.2f}%'.format(results[0],results[1]*100))

def test_predict_4_mic_array_while_training_wrapper():

    if False:
        DOM_ADPAT=False
        model_name = 'model_resnet.159-0.09.hdf5'  # 'model_conv.133-0.06.hdf5'
        load_path = os.path.join('pretrained_models', model_name)
    else:
        e = 128
        DOM_ADPAT=True
        model_folder = 'da_2/grad_rev_1'
        model_name = 'model_g.e{}.hdf5'.format(e)
        load_path = os.path.join('checkpoints', model_folder, model_name)

    model= load_model(load_path, {'GradReverse':GradReverse(),'categorical_focal_loss_fixed':categorical_focal_loss(),
                                  'binary_focal_loss_fixed':binary_focal_loss()})
    by_device = True if 'd8' in model_folder else False

    metrics = test_predict_4_mic_array_while_training(model,by_device=by_device, DOM_ADAPT=DOM_ADPAT)

    loss_sil, acc_sil = metrics[0]
    loss_sp, acc_sp = metrics[1]
    loss_laugh, acc_laugh = metrics[2]
    loss_cough, acc_cough = metrics[3]

    print("TEST - 4 Mic Array\tsil_acc: {:.2f}%, sp_acc: {:.2f}%, laugh_acc: {:.2f}%, cough_acc: {:.2f}%".format(acc_sil*100, acc_sp*100, acc_laugh*100, acc_cough*100))

    # tf.summary.scalar('12) test_4mic_sil_acc', acc_sil, step=e)
    # tf.summary.scalar('13) test_4mic_sp_acc', acc_sp, step=e)
    # tf.summary.scalar('14) test_4mic_laugh_acc', acc_laugh, step=e)
    # tf.summary.scalar('15) test_4mic_cough_acc', acc_cough, step=e)

def test_predict_4_mic_array_while_training(model, by_device, DOM_ADAPT=True):

    num_domains = len(c['domains_devices']) if by_device else len(c['domains'])

    path_pkl_dict = os.path.join(c_d['folder_data'], '4mic_array_test_pkl_dict.pkl')

    with open(path_pkl_dict, 'rb') as pkl_file:
        pkl_dict, pkl_counter = pickle.load(pkl_file)

    batch_size = 32
    metrics = []
    for typ in ['silence', 'speech', 'laughter','cough']:

        spec_framed = pkl_dict[typ][0][0]
        label = 1 if typ == 'cough' else 0

        results_arr = []
        for i in np.arange(0, spec_framed.shape[0], batch_size):
            X = spec_framed[i:i + batch_size]
            y = np.array([label] * X.shape[0])
            if DOM_ADAPT:
                y2 = tf.keras.utils.to_categorical(y, num_classes=num_domains)  # one-hot encode
                y = [y, y2]
            result = model.test_on_batch(X, y)

            result = list(np.array(result)[np.array([1, 3])]) if DOM_ADAPT else result
            results_arr += [result]

        results = np.mean(np.array(results_arr), axis=0)
        # print(typ, 'loss: {:.4f}, acc: {:.2f}%'.format(results[0], results[1] * 100))

        metrics += [results]

    return (metrics)

def test_predict_multiple():

    with open(os.path.join('data_test', 'coughs_tr.pkl'), 'rb') as pkl_file:
        coughs = pickle.load(pkl_file)

    model,domains = load_trained_model()

    samps = coughs

    file_dict_dom = {}
    X_stacked = None

    results = []
    meta = []

    for samp in samps:
        S = samp[0]
        name = samp[1]
        annotation = samp[3]
        domain = get_domain(name)

        # predict in 19 frame chunks
        seg = 0
        for k, j in enumerate(np.arange(0, S.shape[1], 19)):
            end = S.shape[1] if S.shape[1] - j < 19 else j + 19
            start = end - 19 if S.shape[1] - j < 19 else j
            S_test = S[:, start:end].transpose()
            S_test = np.expand_dims(np.expand_dims(S_test,-1),0)
            res = model.predict(S_test)
            results += [res, name, domain, annotation, seg]
            seg+=1
            print(res)
            print(name, res[0][0][0], np.argmax(res[1]))
            break
        break

def create_predict_pkl(file, model):

    with open(os.path.join(c_d['folder_data'], file), 'rb') as pkl_file:
        samps = pickle.load(pkl_file)

    samp_dict = {}
    for d in c['domains_devices']:
        samp_dict[d] = []
    for samp in samps:
        samp = list(samp)
        d = get_domain(samp[1],by_device=True)
        samp[2] = d
        S = get_framed_input(samp[0])
        res = model.predict(np.expand_dims(S,-1)).squeeze()
        res_mean = np.mean(res)
        samp +=[res_mean]
        samp_dict[d]+=[samp]
        # print(samp[1], res_mean)

    file_no_ext = Path(file).stem
    with open(os.path.join(c_d['folder_data'], '{}_results.pkl'.format(file_no_ext)), 'wb') as pkl_file:
        pickle.dump(samp_dict,pkl_file)

def check_test_predict_coughs(file):

    with open(os.path.join(c_d['folder_data'], file), 'rb') as pkl_file:
        results_dict = pickle.load(pkl_file)
    typ = Path(file).stem.split('_')[0]
    print(typ)

    for d,v in results_dict.items():
        if len(v)<1:
            continue
        result = np.array([])

        for samp in v:
            result = np.append(result, samp[4])
        print(d,"# samps:",len(result),"sigmoid confidence:",'{0:.3f}'.format(np.mean(result)))

def test_predict_all_typ():

    TRAIN = True
    suff = 'tr' if TRAIN else 'te'

    time_string = 'resnet1'  # 'conv0'
    model_name = 'model.079-0.42.hdf5'  # 'model.106-0.41.hdf5'
    load_path = os.path.join('checkpoints', time_string, model_name)
    model = load_model(load_path)

    create_predict_pkl('coughs_{}.pkl'.format(suff),model)
    check_test_predict_coughs('coughs_{}_results.pkl'.format(suff))

    create_predict_pkl('sniff_other_breath_{}.pkl'.format(suff),model)
    check_test_predict_coughs('sniff_other_breath_{}_results.pkl'.format(suff))

    create_predict_pkl('speech_{}.pkl'.format(suff),model)
    check_test_predict_coughs('speech_{}_results.pkl'.format(suff))

def get_results(model, S_framed, batch_size, tflite):

    confidences_all = np.array([])
    for i in np.arange(0, S_framed.shape[0], batch_size):
        if S_framed.shape[0] - i < batch_size:
            continue
        X = S_framed[i:i + batch_size]
        if tflite:
            model.set_tensor(model_idx[0], X.astype(np.float32))
            model.invoke()
            confidences = model.get_tensor(model_idx[1]).squeeze()
        else:
            confidences = model.predict(X)

        confidences_all = np.append(confidences_all, confidences)
    return(confidences_all)

    # results = np.mean(np.array(results_arr), axis=0)
    # print(subject, typ, 'loss: {:.4f}, acc: {:.2f}%'.format(results[0], results[1] * 100))

def check_spec_indiv():

    tflite = True
    batch_size = 10 if tflite else 32

    model = load_trained_model(tflite=tflite, batch_size=batch_size)

    # path = r'C:\Users\mattw12\Documents\Research\cough_count\data_test\features_farshid_speech.npy'
    # S_rpi = np.load(path).T
    #
    # plt.figure()
    # librosa.display.specshow(librosa.power_to_db(S_rpi.T + 10, ref=np.max), sr=c['sr'],
    #                          hop_length=int(c['hop'] * c['sr']), x_axis='time', y_axis='mel')
    #
    folder = r'Z:\research\cough_count\data\raw\respeaker_4mic'
    file = '4mic_array_Richard_cough.wav'
    path = os.path.join(folder,file)
    npy_path = os.path.join(folder, Path(file).stem +'_Sframed.npy')
    if os.path.exists(npy_path):
        S_framed_cpu = np.load(npy_path)
        print("LOADED S_framed FROM NPY")
    else:
        # if
        y,_ = librosa.load(path,16000)

        S_cpu = librosa.core.stft(y=y, n_fft=c['n_fft'], win_length=int(c['window'] * c['sr']),
                              hop_length=int(c['hop'] * c['sr']))
        S_cpu = np.abs(S_cpu) ** 2
        mel_basis = librosa.filters.mel(sr=c['sr'], n_fft=c['n_fft'], n_mels=c['nmels'])
        S_cpu = np.log10(np.dot(mel_basis, S_cpu) + 1e-6)
        #
        # plt.figure()
        # librosa.display.specshow(librosa.power_to_db(S_cpu + 10, ref=np.max), sr=c['sr'],
        #                          hop_length=int(c['hop'] * c['sr']), x_axis='time', y_axis='mel')
        # # plt.show()
        #
        # print(S_rpi.shape, S_cpu.shape)
        #
        # S_framed_rpi = np.expand_dims(get_framed_input(S_rpi), -1)
        S_framed_cpu = np.expand_dims(get_framed_input(S_cpu), -1)
        np.save(npy_path, S_framed_cpu)
    #
    # confidences_rpi = get_results(model, S_framed_rpi, batch_size, tflite)
    confidences_cpu = get_results(model, S_framed_cpu, batch_size, tflite)
    confidences_all_rpi = confidences_cpu
    #
    # for i, (c_r, c_c) in enumerate(zip(confidences_rpi,confidences_cpu)):
    #     if c_r >.5 or c_c>.5:
    #         print(i, '{:.2f}% {:.2f}%'.format(c_r*100,c_c*100))

    # path = r'C:\Users\mattw12\Documents\Research\cough_count\data_test\big_x.npy'
    # S_batches_rpi = np.load(path)
    # confidences_all_rpi = np.array([])
    # for batch in S_batches_rpi:
    #     confidences = get_results(model, batch, batch_size, tflite)
    #     confidences_all_rpi = np.append(confidences_all_rpi, confidences)

    # path = r'C:\Users\mattw12\Documents\Research\cough_count\data_test\confidences.npy'
    # confidences_all_rpi = np.load(path)

    med_len = 21
    coughs = 0
    cough_cntdown = 0
    for i in range(len(confidences_all_rpi)-med_len):
        conf = confidences_all_rpi[i:i+med_len]
        classification = np.median([1 if f>.75 else 0 for f in conf])
        if classification>0:
            if cough_cntdown==0:
                coughs+=1
            cough_cntdown = 30
        cough_cntdown-=1
        cough_cntdown = max(cough_cntdown,0)
        conf_print = [float('{:.2f}'.format(f*100)) for f in conf]
        print(i, int(classification), coughs, conf_print)

    print("coughs detected:", coughs)
    acc = np.mean([1 if f <.5 else 0 for f in confidences_all_rpi])
    print("Accuray: {:.2f}%".format(acc*100))

    # for i, c_r in enumerate(confidences_all_rpi):
    #     # if i>10:
    #     #     break
    #     # print(c_r)
    #     if c_r > .5:
    #         print(i, '{:.2f}%'.format(c_r * 100))


def check_specs(file):

    path_plots = os.path.join('plots',Path(file).stem)
    os.makedirs(path_plots,exist_ok=True)

    with open(os.path.join(c_d['folder_data'], file), 'rb') as pkl_file:
        samps = pickle.load(pkl_file)

    cnt = 0
    cnt_figs = 0
    n_rows = 2
    n_colmuns = 3
    for samp in samps:
        d = get_domain(samp[1],by_device=True)
        if not 'whosecough' in d:
            continue
        S = samp[0]
        name = samp[1]

        idx = cnt % (n_rows * n_colmuns)
        if idx ==0:
            fig, ax = plt.subplots(n_rows,n_colmuns,figsize=(12,8))
        row = 0 if idx <n_colmuns else 1
        column = idx % n_colmuns
        ax_cur = ax[row][column]
        librosa.display.specshow(librosa.power_to_db(S + 10, ref=np.max), sr=c['sr'],
                                 hop_length=int(c['hop'] * c['sr']), x_axis='time', y_axis='mel',ax=ax_cur)
        ax_cur.set_title(name)
        plt.tight_layout()

        # plt.colorbar(format='%+2.0f dB')
        # plt.title(title)

        if idx == 5:
            plt.savefig(os.path.join(path_plots,'specs_{}.png'.format(cnt_figs)))
            plt.close()
            print("Plot finished, included file:", name)
            cnt_figs+=1
        cnt+=1

def test_predict_domains():

    DOM_ADAPT=True
    SAMPLE_WEIGHTS = True
    BY_DEVICE = True#False
    MODEL_D = True
    model, domains = load_trained_model(DOM_ADAPT)

    CV = 0

    files_speech_tr, files_speech_te, file_dict_silence_tr, file_dict_silence_te, files_noise_tr, files_noise_te,\
    files_sound_tr, files_sound_te = get_files_sil_noise(CV)

    tr_generator = DataGenerator(TRAIN=True, CV=CV, file_dict_silence=file_dict_silence_tr, files_noise=files_noise_tr,
                                 files_sound = files_sound_tr, frames=c['frames'], nmels=c['nmels'], n_classes=2,
                                 SAMPLE_WEIGHTS=SAMPLE_WEIGHTS, by_device=BY_DEVICE)
    val_generator = DataGenerator(TRAIN=False, CV=CV, file_dict_silence=file_dict_silence_te, files_noise=files_noise_te,
                                  files_sound = files_sound_te, frames=c['frames'], nmels=c['nmels'], n_classes=2,
                                  SAMPLE_WEIGHTS=SAMPLE_WEIGHTS,by_device=BY_DEVICE)

    for i in range(5):
        X,y,sw = tr_generator.__getitem__(0)
        res = model.predict(X)
        res = res if MODEL_D else res[1]

        y2  = np.argmax(res.squeeze(),1)
        print(np.argmax(y[1].squeeze(),1))
        print(y2)
        hist = np.histogram(y2,bins = [-.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
        print(list(hist[0]))
        # for i in range(len(hist[1])-2):
        #     print(i, hist[0][i])

def convert_npy_to_wav():

    mic_type = 'rs'
    folder = r'C:\Users\mattw12\Documents\Research\cough_count\data_test\rs_vs_omni'
    folder_full = os.path.join(folder,mic_type)
    files = [f for f in os.listdir(folder_full) if f.endswith('.npy')]

    for f in files:
        path = os.path.join(folder_full,f)
        path_save = os.path.join(folder_full, Path(f).stem + '.wav')
        a = np.load(path)
        librosa.output.write_wav(path_save,a,sr=16000)


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #use cpu in case using gpu to train

    test_predict()
    # test_predict_all_typ()
    # check_specs('coughs.pkl')

    # test_predict_multiple()
    # test_predict_FSDKaggle()
    # test_predict_4_mic_array()
    # test_predict_4_mic_array_while_training_wrapper()
    # test_predict_domains()
    # test_predict_rs4mic()
    # test_predict_4_mic_array()
    # check_spec_indiv()
    # convert_npy_to_wav()
