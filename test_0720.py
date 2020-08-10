from scipy import signal as scipysig
from scipy.io import wavfile
import numpy as np
import pickle

import random

import matplotlib.pyplot as plt


pkl_file = './test_data/label_whosecough_uwct-1-0-bm_0.pkl'
wav_file = './test_data/wav_whosecough_uwct-1-0-bm_0.wav'

with open(pkl_file,'rb') as pf:
    aug = pickle.load(pf).astype("uint8")

_, wav = wavfile.read(wav_file,'w') # sr=16000, wav = list[80000], max=0.71435547


MAX_SAMPS = 4.86 # sr = 16000, sr*MAX_SAMPS = 77760

def visualize(wav,lbl,num_lbls = 175,ax=plt):
    '''
    visulization
    '''
    y,aug = wav,lbl

    pkl_array = np.asarray(aug)
    y_array = np.asarray(y)

    y_max = np.max(y_array)
  
    #ax.figure()
    if ax==plt:
        ax.xlim(0,y_array.shape[0])
    else:
        ax.set_xlim(0,y_array.shape[0])
    ax.plot(y_array)
    

    pos = np.linspace(ax.axis()[0],ax.axis()[1],num_lbls)
    if pkl_array.shape[0] > num_lbls :
        for i in range(pos.shape[0]):
            ax.text(pos[i],y_max,pkl_array[int( pkl_array.shape[0]/num_lbls * i)])
    else:
        #print(pkl_array.shape[0],y_array.shape[0])
        ax.text(0,y_max,'Bad labels: too little labels')



def random_crop(wav,lbl,sr=16000,extra_p = 0.25,output_l=0.972):

    '''
    random crop :
    para wav: input wave
    para lbl: input lable
    sr: sample rate, default 16000
    extra_p: ratio of expanded length to sr, default 0.25, so 
    default expanded length is 0.25*16000=4000, which will be 
    added on each side halfly
    output_l: ratio of output length to initial wav length
    '''

    ini_time = len(wav)/sr
    lbl_per_sec = int(len(lbl)/ini_time)

    output_wav_length , output_lbl_length = int(len(wav)*output_l) , int(len(lbl)*output_l)

    if extra_p > 0: # 0 if no need for expansion
        extra_samp_flag = 1

    if extra_samp_flag:

        extra_wav = int(sr*extra_p)
        extra_lbl = int(lbl_per_sec*extra_p)
        
        needed_wav,needed_lbl = extra_wav+output_wav_length-len(wav) , extra_lbl+output_lbl_length-len(lbl)

        needed_wav_head, needed_wav_tail = int(needed_wav/2) , needed_wav-int(needed_wav/2)
        needed_lbl_head, needed_lbl_tail = int(needed_lbl/2) , needed_lbl-int(needed_lbl/2)

        wav_pad , lbl_pad = (needed_wav_head,needed_wav_tail),  (needed_lbl_head,needed_lbl_tail)
        # zero padding

        full_wav = np.pad(np.asarray(wav),wav_pad,'constant')
        full_lbl = np.pad(np.asarray(lbl),lbl_pad,'constant')

        start_lbl = random.randint(0,full_lbl.shape[0]-output_lbl_length)
        start_wav = (int(sr/lbl_per_sec))*start_lbl

        print(start_lbl,start_wav)


        out_wav = full_wav[start_wav:start_wav+output_wav_length]
        out_lbl = full_lbl[start_lbl:start_lbl+output_lbl_length]
    
    else: # do not to add extra padding
        
        start_lbl = np.randint(0,len(lbl)-output_lbl_length)
        start_wav = (sr/lbl_per_sec)*start_lbl

        out_wav = full_wav[start_wav:start_wav+output_wav_length]
        out_lbl = full_lbl[start_lbl:start_lbl+output_lbl_length]
    
    return out_wav,out_lbl


def random_gain(wav,sf=1,db=None):

    '''
    random gain:
    to randomly adjust the amplitude

    para wav: input wav
    para sf: average gain ratio, default 1
    para db: the gain ratio range. default None. in format [db_min,db_max]

    '''

    if db is None:
        max_ampli = np.max(np.abs(wav))
        if max_ampli > .9:
            db_range = [-15,2]
        elif max_ampli > .3:
            db_range = [-5,10]
        else:
            db_range = [0,15]
    
    else:
        db_range = db

    ampli_gain = np.power(10,random.uniform(db_range[0],db_range[1])/20)

    wav_new = np.clip(wav*sf*(ampli_gain),-1,1)

    return wav_new, np.max(np.abs(wav_new)) 


def add_noise(wav,sigma=1e-2):
    '''
    add random noise
    para sigma: standard_deviation of noise, default 1e-2
    '''

    output_wav = np.copy(wav)
    if sigma==0:
        return output_wav
    
    noise = np.random.normal(0,sigma,size=len(wav))
    
    for i in range(len(wav)):
        output_wav[i] = wav[i]+noise[i]

    return output_wav

if __name__ == "__main__":
    
    print(wav)