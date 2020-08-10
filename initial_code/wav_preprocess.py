import numpy as np
import random, os, pickle
from scipy.io import wavfile
from scipy import signal as scipysig

import config
from config import config as c
from config import config_data as c_d

def get_dbs(max_ampl):

    if max_ampl > .9:
        return([-15, 2])
    elif max_ampl > .3:
        return([-5,10])
    else:
        return([0, 15])

def random_crop(size, add_extra=False):
    extra = .25*c['sr'] # quarter second extra (equates to .125 s on either side)
    
    def f(sound):
        org_size = len(sound)
        extra_samp = extra if add_extra else 0

        # Zero pad around sound to get a little extra so we can try it out at different start points in frame
        if (size + extra_samp) > org_size:
            needed = int(size-org_size + extra_samp)
            half = int((size-org_size + extra_samp)/2)
            pad = (half, needed-half)
            sound = np.pad(sound,pad,'constant')
            org_size = len(sound)

        #  do random crop
        start = random.randint(0, org_size - size)
        s = sound[start: start + size]
        return s
    return f

def mean_std_normalize():
    def f(sound):
        sound = sound - np.float32(np.mean(sound))
        return sound / np.float32(np.std(sound) + 1e-15)
    return f

def add_noise(noise_std=False):
    folder_demand = os.path.join(c['folder_aug'], 'demand')
    files_aug = [os.path.join(folder_demand,f) for f in os.listdir(folder_demand)]
    rg = random_gain()
    rc = random_crop(config.MAX_SAMPS)
    def f(sound):

        # if random.randint(0,1):
        #     return(sound)
        _, noise = wavfile.read(np.random.choice(files_aug,1)[0])
        noise = rc(noise)
        max_ampl = max(abs(sound))
        max_ampl_noise = max(abs(noise))
        noise = rg(noise, sf=max_ampl/max_ampl_noise, db=[-20,-8])
        # noise = np.random.randn(sound.shape[0], sound.shape[1]) * noise_std
        return (sound + np.float32(noise))
    return f

def random_scale(max_scale):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        input_size = sound.shape[1]
        output_size = int(sound.shape[1] * scale)
        xp = np.linspace(0, input_size, input_size)
        x = np.linspace(0, input_size, output_size)
        scaled_sound = np.interp(x, xp, sound[0])[np.newaxis, :]
        return scaled_sound
    return f

def random_gain():
    def f(sound, sf=1, db=None):

        # if random.randint(0,1):
        #     return(sound)

        db = get_dbs(max(abs(sound))) if db is None else db
        return np.clip(sound * sf * np.float32(np.power(10, random.uniform(db[0], db[1]) / 20.0)),-1.0,1.0)
    return f

# def loadRoomAugmentations(augDir):
#     augs = []
#     for file in [f for f in os.listdir(augDir) if f.endswith(".pickle")]:
#         with open(os.path.join(augDir, file), "rb") as filePointer:
#             aug = pickle.load(filePointer).astype("float64")
#             augs.append([aug, os.path.splitext(file)[0]])
#     return augs

def sqrtEnergy(data):
    return max(np.sqrt(np.sum(data ** 2)), 1e-15)

def reverb():
    augDir = os.path.join(c['folder_aug'], 'reverb')
    aug_files = [f for f in os.listdir(augDir) if f.endswith(".pickle")]
    def f(sound):

        # if random.randint(0,1):
        #     return(sound)

        idx = np.random.randint(len(aug_files))
        file = aug_files[idx]
        with open(os.path.join(augDir, file), "rb") as filePointer:
            aug = pickle.load(filePointer).astype("float64")
        sound_new = scipysig.convolve(sound, aug, mode='full')
        sound_new = sound_new[0:sound.shape[0]]
        sound_new /= sqrtEnergy(sound_new)
        sound_new *= sqrtEnergy(sound)
        sound_new = np.clip(sound_new,-1.0,1.0)
        return(sound_new)
    return f

if __name__ == '__main__':
    # sound = np.random.random((1,160000))
    # ff = random_scale(1.25)
    # s = ff(sound)
    an = add_noise()
    rg=random_gain()
    rc = random_crop(config.MAX_SAMPS, add_extra=True)
    rvb = reverb()
    # _, y_orig = wavfile.read(r'C:\Users\mattw12\Documents\Research\cough_count\test_data\wav_cough_16khz_2.wav')
    import librosa
    y_orig, _ = librosa.load(r'C:\Users\mattw12\Documents\Research\cough_count\model_predict_sample\wav_cough.wav',sr=c['sr'])

    for i in range(10):
        y = rc(y_orig)
        # y = rg(y)
        y = rvb(y)
        y = an(y)
        wavfile.write(r'C:\Users\mattw12\Documents\Research\cough_count\test_data\aug\wav_cough_8khz_reverb_{}.wav'.format(i), c['sr'], y)
