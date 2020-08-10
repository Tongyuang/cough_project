import librosa
import os
import subprocess
import matplotlib.pyplot as plt

folder = r'C:\Users\Administrator\Downloads'
file = r'5e6d9036c4384_1584238646.wav'
file2 = "test2.wav"
path = os.path.join(folder, file)
path2 = os.path.join(folder, file2)

subprocess.call([r"C:\FFmpeg\bin\ffmpeg.exe","-y","-i", path, path2], shell=True)

data, sr = librosa.load(path2, sr=16000)

plt.figure()
plt.plot(librosa.samples_to_time(range(len(data)), sr=sr),data)
plt.show()