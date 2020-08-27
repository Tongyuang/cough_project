from scipy import signal as scipysig
from scipy.io import wavfile
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

import config
from config import config as c
import os

def visualize(wav,lbl_real,lbl_predicted=None,name=None,ax=plt):
    '''
    visulization
    para wav: wave file
    para lbl_real: real lable file
    para lbl_pred: pred lable file
    para name: str-like, name of wav
    para ax: drawing tool, default plt
    '''
    if lbl_predicted is None:
        plot_lbl_predicted = False
    else:
        plot_lbl_predicted = True

    # to array-like
    wav_array = np.array(wav)
    lbl_array = np.array(lbl_real)
    pred_array = np.array(lbl_predicted)
    

    y_max = np.max(wav_array)
    
    _,axs = plt.subplots(nrows=2,ncols=1,sharex=True)
    axs = axs.flatten()


    axs[0].plot(wav_array)
    axs[0].set_ylabel('initial wave')
    axs[0].set_xlim([0,wav_array.shape[0]])
    axs[0].set_ylim([-1,1])
    if name is not None:
        axs[0].set_title(name)

    pos = np.linspace(ax.axis()[0],ax.axis()[1],lbl_real.shape[0])

    # visualization

    if plot_lbl_predicted:
        if lbl_array.shape[0] > 50 :
            axs[1].plot(pos,lbl_array,color='red',label='true')
            axs[1].plot(pos,pred_array,color='blue',label='prediction')
            axs[1].set_xlim([0,wav_array.shape[0]])
            axs[1].set_ylabel('lables')
            axs[1].legend()
        else:
            axs[1].text(0,y_max,'Bad labels: too little labels')
    else:
        if lbl_array.shape[0] > 50 :
            axs[1].plot(pos,lbl_array,color='red',label='true lables')
            axs[1].set_xlim([0,wav_array.shape[0]])
            axs[1].set_ylabel('lables')
            axs[1].legend()
        else:
            axs[1].text(0,y_max,'Bad labels: too little labels')


def random_crop(wav,lbl=None,extra_p = 0.25):

    '''
    random crop :
    para wav: input wavelables
    para lbl: input lable
    extra_p: ratio of expanded length to sr, default 0.25, so 
    default expanded length is 0.25*16000=4000, which will be 
    added on each side halfly
    
    '''

    if lbl is None:
        lbl = np.zeros(int(c['duration']/c['duration_t']))

    sr = c['sr'] # sr: sample rate, default 16000
    #output_l = c['output_length']/c['duration'] # output_l: ratio of output length to initial wav length

    ini_time = len(wav)/sr
    lbl_per_sec = int(1/config.duration_t)

    output_wav_length , output_lbl_length = int(sr*c['output_length']) , int(c['output_length']*lbl_per_sec)

    if extra_p > 0: # 0 if no need for expansion
        extra_samp_flag = 1

    if len(lbl)-output_lbl_length>=0:
        if extra_samp_flag:

            extra_wav = int(sr*extra_p)
            extra_lbl = int(lbl_per_sec*extra_p)
        
            needed_wav,needed_lbl = extra_wav+output_wav_length-len(wav) , extra_lbl+output_lbl_length-len(lbl)

            if needed_wav > 0 :
                needed_wav_head, needed_wav_tail = int(needed_wav/2) , needed_wav-int(needed_wav/2)
                needed_lbl_head, needed_lbl_tail = int(needed_lbl/2) , needed_lbl-int(needed_lbl/2)

                wav_pad , lbl_pad = (needed_wav_head,needed_wav_tail),  (needed_lbl_head,needed_lbl_tail)
                # zero padding

                full_wav = np.pad(np.asarray(wav),wav_pad,'constant')
                full_lbl = np.pad(np.asarray(lbl),lbl_pad,'constant')

                start_lbl = random.randint(0,full_lbl.shape[0]-output_lbl_length)
                start_wav = (int(sr/lbl_per_sec))*start_lbl

                out_wav = full_wav[start_wav:start_wav+output_wav_length]
                out_lbl = full_lbl[start_lbl:start_lbl+output_lbl_length]
            else: # too loooooooooooooong
                start_lbl = random.randint(0,len(lbl)-output_lbl_length)
                start_wav = (int(sr/lbl_per_sec))*start_lbl                

                out_wav = wav[start_wav:start_wav+output_wav_length]
                out_lbl = lbl[start_lbl:start_lbl+output_lbl_length]
    
        else: # do not add extra padding
            start_lbl = random.randint(0,len(lbl)-output_lbl_length)
            start_wav = (int(sr/lbl_per_sec))*start_lbl 

            out_wav = wav[start_wav:start_wav+output_wav_length]
            out_lbl = lbl[start_lbl:start_lbl+output_lbl_length]
        
    else: # too short, len(lbl) < output_lbl_length
        out_wav = np.zeros(output_wav_length)
        out_lbl = np.zeros(output_lbl_length)
        
        start_lbl = random.randint(0,output_lbl_length-len(lbl))
        start_wav = (int(sr/lbl_per_sec))*start_lbl 

        out_wav[start_wav:start_wav+len(wav)] = np.asarray(wav)
        out_lbl[start_lbl:start_lbl+len(lbl)] = np.asarray(lbl)

    return np.asarray(out_wav),np.asarray(out_lbl)



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

    return wav_new



def add_white_noise(wav,sigma=1e-2):
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

def add_background_noise(wav,db=[-20,-8]):
    '''
    add noise from current background audio files
    '''
    folder_demand = os.path.join(c['folder_aug'], 'demand')
    files_aug = [os.path.join(folder_demand,f) for f in os.listdir(folder_demand)]

    _, noise = wavfile.read(np.random.choice(files_aug,1)[0])

    max_ampli = np.max(np.abs(np.asarray(wav)))
    max_ampli_noise = np.max(np.abs(np.asarray(noise)))

    noise = random_gain(noise,sf=max_ampli/max_ampli_noise,db=db)
    noise,_ = random_crop(noise,lbl=None)
    return (wav+noise)

def add_loud_noise(wav_name,domain):

    def add_noise(wav_name,domain,silence_dict,noise,noise_ampli=0.75,silence_ampli=0.08,silence_range=8000):
        '''
        add loud noise where there's a long-time silence
        silence: when the amplitude is less than 0.08
        add noise that has max amplitude=noise_ampli*max(wav)

        para wav_name: initial wav_name
        para domain_name: domain_name
        para noise: noise
        '''
        
        domain_path = 'wav_dur_{}_{}'.format(config.sr_str,config.domain_name_dict[domain])
        wav_path = config.config_data['folder_data']+'/'+domain_path+'/'+wav_name +'.wav'
        
        _, wav = wavfile.read(wav_path) 
       
        wav = np.array(wav)
        noise = np.array(noise)

        wav_max_ampli = np.max((np.max(wav),np.abs(np.min(wav))))

        
        # if there's a long-time silence:

        silence_list = silence_dict[wav_name]
        
        if len(silence_list):
            for sublist in silence_list:
                start_idx,end_idx = sublist[0],sublist[1]
                # random select
                if noise.shape[0] < end_idx-start_idx:# noise too short
                    continue
                
                noise_start_idx = random.randint(0,noise.shape[0]-(end_idx-start_idx))
                noise_end_idx = noise_start_idx + (end_idx-start_idx)

                noise_part = noise[noise_start_idx:noise_end_idx]
                noise_part = ( noise_part / (np.max((np.max(noise_part),np.abs(np.min(noise_part))))) )* noise_ampli * wav_max_ampli 
                wav[start_idx:end_idx] += noise_part
    
        return np.asarray(wav)
    

    noise_file = config.noise_file_list[random.randint(0,len(config.noise_file_list)-1)]

    noise_file = config.noise_path + '/' + noise_file

    _, noise = wavfile.read(noise_file,'r')
    
    wav = add_noise(wav_name,domain,config.silence_dict,noise)

    return wav



                


if __name__ == "__main__":
    pass
      