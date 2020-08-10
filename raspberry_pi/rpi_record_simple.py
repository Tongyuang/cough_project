import pyaudio
import wave
import soundfile as sf
import os
import numpy as np
from datetime import datetime

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 60
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(
            rate=RESPEAKER_RATE,
            format=pyaudio.paFloat32,
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,)

print("* recording")

frames = np.array([])
try:
    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        x = np.frombuffer(data, dtype=np.float32)
        frames = np.append(frames,x)
    print("* done recording")
except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
p.terminate()

folder='4mic_array'
os.makedirs(folder,exist_ok=True)
time_string = datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
path = os.path.join(folder, '4mic_array_{}.wav'.format(time_string))
sf.write(path,frames, RESPEAKER_RATE, 'PCM_16')

