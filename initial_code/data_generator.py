import numpy as np
import tensorflow as tf
import os
from scipy.io import wavfile
import librosa

import config
from config import config as c
from cross_validation import get_file_list, get_typ_and_domain_probabilities_dicts
import wav_preprocess as wp
from evaluate_full_file import DataGeneratorFullFile
from vggish import mel_features
from vggish import vggish_params as vp

# from here - https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
class DataGenerator(tf.keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, model_type, TRAIN, domains, subtypes_dict, percentages_dict, file_list):

        'Initialization'
        self.model_type = model_type
        self.TRAIN=TRAIN
        self.sps = config.samps_per_subtype
        self.domains = domains
        self.subtypes_dict = subtypes_dict
        self.percentages_dict = percentages_dict
        self.file_list = file_list
        self.preprocess_setup()
        self.random_seed_idx = 0

    def preprocess_setup(self):
        if self.TRAIN:
            funcs = [wp.random_crop(config.MAX_SAMPS),#, add_extra=True),
                     wp.random_gain(),
                     wp.add_noise(1),
                     wp.reverb()]#,
                     #wp.mean_std_normalize()]
        else:
            funcs = [wp.random_crop(config.MAX_SAMPS)]#,
                     # wp.mean_std_normalize()]
        self.preprocess_funcs = funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)
        return sound

    def __len__(self):
        'Denotes the number of batches per epoch'

        return 10000#int(np.floor(self.n_coughs/self.num_per_batch['cough']))



    def __getitem__(self, index):

        X = []
        y = []

        # This sets the random seed during validation so we can the same batch each time through for validation
        if not self.TRAIN:
            np.random.seed(self.random_seed_idx)
            self.random_seed_idx +=1

        # Walk through each subtype (i.e. hard sounds, respiratory sounds, speech, etc).
        # The number is hard coded in config.py
        for subtyp, n_sub in self.sps.items():

            label = 1 if subtyp == 'cough' else 0

            # The types are all of the sound types within that subtype. For example, in "hard sounds", it has snare drum,
            # gunshot, etc. The pcts are how much to weight each of those sounds since some have way more samples than
            # others.
            typs = self.subtypes_dict[subtyp]['types']
            pcts = self.subtypes_dict[subtyp]['pcts']

            # this will give an array back saying how many of each type to do based on the percentages.
            # i.e. [0, 2, 3,...] means do 0 snare samples, 2 gunshot, etc. It should add up to the "n_sub" number.
            n_per_subtyp = np.random.multinomial(n_sub, pcts)

            # Walk through each typ and find out how many from each domain to use. For example, under the cough type,
            # we have "coughsense", "pediatric", "southafrica", etc. Because some have way more samples than others,
            # we again use the percentages to figure out how many of each domain to do. For example, [0, 2, 3,..] means
            # do 0 from coughsense, 2 from pediatric, etc. Because not every type has samples from every domain, there
            # will be some types with only a few domains, or even just 1.
            for typ, n_typ in zip(typs, n_per_subtyp):

                pcts_typ = self.percentages_dict[typ]['pcts']
                doms_typ = self.percentages_dict[typ]['doms']

                # if just one domain, all samples come from that domain
                if len(pcts_typ) == 1:
                    n_per_typ = [n_typ]
                else:
                    n_per_typ = np.random.multinomial(n_typ, pcts_typ)

                # walk through the domains and get the right number of samples
                for dom, n_dom in zip(doms_typ, n_per_typ):

                    if n_dom < 1:
                        continue

                    folder = os.path.join(c['folder_wav'], typ, dom)
                    files = self.file_list[typ][dom]

                    if (len(files) < n_dom):
                        print(subtyp, typ, dom, len(files), doms_typ)

                    # get the files for this type and this domain based on the cross-validation set
                    files_chosen = np.random.choice(files, n_dom)
                    for f in files_chosen:
                        path = os.path.join(folder, f)
                        try:
                            _, samp = wavfile.read(path)
                        except:
                            print("Couldn't get file:", path) #typ, dom, f)
                            continue
                        samp = self.preprocess(samp)
                        if self.model_type == 'vggish':
                            samp = mel_features.log_mel_spectrogram(samp, audio_sample_rate=c['sr'], log_offset=vp.LOG_OFFSET,
                                                                 window_length_secs=vp.STFT_WINDOW_LENGTH_SECONDS, hop_length_secs=vp.STFT_HOP_LENGTH_SECONDS,
                                                                 num_mel_bins=vp.NUM_MEL_BINS, lower_edge_hertz=vp.MEL_MIN_HZ,
                                                                 upper_edge_hertz=vp.MEL_MAX_HZ)
                            samp = np.expand_dims(samp, axis=-1)

                        X.append(samp)
                        y.append(label)
                        # print("\nsamp_"+str(len(y)),typ, dom, f)
                        # if len(y)<2 and self.TRAIN:
                        #     print("\nsamp_"+str(len(y)),typ, dom, f)
                        # wavfile.write(r'C:\Users\mattw12\Documents\Research\cough_count\test_data\aug\samp_{}.wav'.format(len(y)), c['sr'], samp)

        # if for some reason we couldn't get a sample, just repeat the last one
        while (len(X)<c['batch_size']):
            X.append(X[-1])
            y.append(y[-1])

        X = np.array(X)

        if self.model_type == 'sample-cnn':
            X = np.expand_dims(X, -1)

        return X,np.array(y)

if __name__ == '__main__':

    # from pathlib import Path
    # files = [f for f in os.listdir(c['folder_raw']) if f.endswith('-n.flac') or f.endswith('-cp.flac')]
    #
    # for f in files:
    #     for sr in [8, 4]:
    #         y, _ = librosa.load(os.path.join(c['folder_raw'],f), sr=int(sr*1000))
    #
    #         # Turn multiple channels into 1
    #         if len(np.shape(y)) > 1:
    #             y = np.mean(y, axis=1)
    #
    #         # clip to max of 1/-1 to match normal wavs
    #         np.clip(y, -1.0, 1.0, y)
    #
    #         wavfile.write(os.path.join(c['folder_raw'],Path(f).stem + '_{}khz_wavfile.wav'.format(sr)), sr, y)
    #         print("Wrote:", Path(f).stem + '_{}khz_wavfile.wav'.format(sr))
    # raise
    #
    #

    model_type = 'conv-model'
    CV = 0
    domains, subtypes_dict, percentages_dict = get_typ_and_domain_probabilities_dicts()
    files_tr, files_te = get_file_list(CV=CV, TEST=True)

    # setup(files_te, CV)

    tr_generator = DataGenerator(model_type, True, domains, subtypes_dict, percentages_dict, files_tr)
    val_generator = DataGenerator(model_type, False, domains, subtypes_dict, percentages_dict, files_te)
    val_generator_ff = DataGeneratorFullFile(model_type, files_te, CV)

    # print(val_generator_ff.X.shape)
    #
    # from tensorflow.keras.models import load_model
    # from config import config_data as c_d
    # from model_mobilenet import relu6
    # from tensorflow.keras.layers import DepthwiseConv2D
    # from model_samp_cnn import AudioVarianceScaling
    #
    # model_folder_name = 'conv-model_CV-0_new-llf-pretrain-reverb-no-msn_1' #'sample-cnn_CV-0_basic_7L_16F_1'
    #
    # for model_name in [2400,2700,2750]:
    #
    #     model_name = 'model.{}'.format(model_name)
    #     print(model_name)
    #
    #     folder_model = os.path.join(os.path.dirname(c_d['folder_data']), 'logs_and_checkpoints', model_folder_name)
    #     path_model_h5 = os.path.join(folder_model, '{}.hdf5'.format(model_name))
    #
    #
    #     model = load_model(path_model_h5, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
    #                                            'AudioVarianceScaling':AudioVarianceScaling})
    #     res = model.evaluate(x=val_generator_ff.X, y= val_generator_ff.y)
    #     prec = res[2]
    #     recall = res[3]
    #     f1 = 2*prec*recall/(prec+recall)
    #     print("F1: {:.4f}%".format(f1*100))

    # preds = model.predict(val_generator_ff.X).squeeze(-1)
    # classifications = np.array([1 if f>.5 else 0 for f in preds])
    #
    # y_id = np.argwhere(val_generator_ff.y==0).squeeze(-1)
    # classifications_coughs = classifications[y_id]
    # print(classifications_coughs)
    # print(sum(classifications_coughs), len(classifications_coughs))



    print("epoch_len:",tr_generator.__len__())
    while(1):
        X,y = tr_generator.__getitem__(0)
        print("Train batch\t", X.shape, y.shape)
        X_val, y_val = val_generator.__getitem__(0)
        print("Val Batch\t",X_val.shape, y_val.shape)
        print(val_generator_ff.X.shape, val_generator_ff.y.shape)
