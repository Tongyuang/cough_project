import os
import numpy as np
from scipy.io import wavfile

folder = r'C:\Users\mattw12\Downloads\s3_test'

def convert_files():
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]

    for f in files:
        wav_path = os.path.join(folder, f.replace('.txt', '.wav'))
        if not os.path.exists(wav_path):
            with open(os.path.join(folder, f), 'r') as handle:
                lines = np.array(handle.read()[1:-1].split(',')).astype('float')
                wavfile.write(wav_path, 16000, lines)


def combine_and_convert():

    file_1 = 'e43f0dec-e635-4e10-9ad1-58bfac0e097b.wav'
    file_2 = 'a4a6a256-d45b-4bbc-8145-473431bbf7b6.wav'

    _, a1 = wavfile.read(os.path.join(folder, file_1))
    _, a2 = wavfile.read(os.path.join(folder, file_2))
    wavfile.write(os.path.join(folder, 'test1.wav'),16000,np.append(a1,a2))

if __name__ == '__main__':
    convert_files()
    # combine_and_convert()