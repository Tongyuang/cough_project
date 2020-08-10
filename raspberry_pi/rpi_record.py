import pyaudio
import wave
import soundfile as sf
import os
import numpy as np
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mic_type', default='omni', help="Mic type - 'respeaker' or 'omni'")
args = parser.parse_args()

if args.mic_type not in ['omni','rs']:
    raise ValueError('invalid mic_type. must pass a flag with either rs or omni. i.e. --mic_type rs')

RESPEAKER_RATE = 48000 if args.mic_type=='omni' else 16000
RESPEAKER_RATE_SAVE = 16000
RESPEAKER_CHANNELS = 1 if args.mic_type=='omni' else 5
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 2100
RECORD_SECONDS = 60
RECORD_MINS = 120

folder=os.path.join('recorded_wavs',args.mic_type)
os.makedirs(folder,exist_ok=True)

p = pyaudio.PyAudio()

stream = p.open(
            rate=RESPEAKER_RATE,
            format=pyaudio.paFloat32,
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,)

print("* recording")


def save_to_wav(frames):

    frames = np.array(frames)

    if RESPEAKER_CHANNELS>1:
        frames = frames.reshape(-1, RESPEAKER_CHANNELS)
        frames_4mics_avg = np.mean(frames[:,1:],axis=1)
        frames_soundalgos = frames[:,0]

    time_string = datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
    path = os.path.join(folder, '{}_{}_sound_algo.wav'.format(args.mic_type, time_string))
    sf.write(path,frames_soundalgos, RESPEAKER_RATE_SAVE, 'PCM_16')
    path = os.path.join(folder, '{}_{}_4mics_avg.wav'.format(args.mic_type, time_string))
    sf.write(path,frames_4mics_avg, RESPEAKER_RATE_SAVE, 'PCM_16')

    print("Saved {:.1f} min of audio".format(RECORD_SECONDS/60))

try:
    for j in range(RECORD_MINS):
        frames = []
        stream.start_stream()
        for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            x = np.frombuffer(data, dtype=np.float32)
            if args.mic_type == 'omni':
                x = np.mean(x.reshape(-1, 3), axis=1)
            frames += list(x)
        stream.stop_stream()
        save_to_wav(frames)

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
except OSError as err:
    print("OS error:{}".format(err))

if frames:
    save_to_wav(frames)

print("\nNOTE: Saved 2 files. The first files is the output from the sound algorithms (i.e. beamforming, automatic gain control, noise suppression, etc), the second is the average of the raw output from the 4 mics. Both are collected at 16 khz")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mic_type', default='omni', help="Mic type - 'respeaker' or 'omni'")
    args = parser.parse_args()

    if args.mic_type not in ['omni','rs']:
        raise ValueError('invalid mic_type. must pass a flag with either rs or omni. i.e. --mic_type rs')

