#!/usr/bin/env python3

# This script is kind of a catch-all; it loads in the saved model from .h5 files,
# builds the neural network, and can then do things like benchmark it for performance
# measurements (--time), output the values to the console for debugging (--test) or
# just do live classification.
import os
import numpy as np
import scipy.signal
import pyaudio
import sys, os, time
import librosa
import soundfile as sf
import array, csv
import argparse
from ctypes import *
from contextlib import contextmanager
from datetime import datetime
from scipy import stats
from vggish import vggish_params
from vggish import mel_features

from tensorflow.keras.models import load_model
import tensorflow as tf

import RPi.GPIO as GPIO

## SUPPRESS AT LEAST SOME OF THE GAZILLION WARNINGS
# Tensorflow
from tensorflow.compat.v1 import logging
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# For pyaudio
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(c_error_handler)

path_test_audio = "4mic_array/indiv/4mic_array_2020.01.10_15-55-34_farshid_cough.wav" #"4mic_array/cough_5-n_0.npy" #npy_test_uwct_3_0_rp.npy"

# Params
sr_mic=48000
mic_channels = 1
sr=vggish_params.SAMPLE_RATE
sr_multiplier=int(sr_mic/sr)

nmels=vggish_params.NUM_BANDS
window=vggish_params.STFT_WINDOW_LENGTH_SECONDS
hop=vggish_params.STFT_HOP_LENGTH_SECONDS
n_frames = vggish_params.NUM_FRAMES

# Don't ask how we got these numbers, just trust it
MIN_SAMPS = int(np.ceil((((n_frames * hop) + (window - hop)) * sr))) # min samps for 96 mel spec frames
MIN_SAMPS_HALF = int(MIN_SAMPS/2)
# MIN_SAMPS_HALF_FULL_FRAME = int(MIN_SAMPS/2) + int(((window - hop)/2)*sr) # min samps for 48 mel spec frames

# Other Params
batch_size = 4
FRAMES_PER_BUFFER = 24000
MIC_TYPES = ['respeaker', 'omni']
TFLITE= True
THRESH = .85
WC_THRESH=.45

COUGH_LATCH_REFILL = 1 # 2 extra frames
full_arr = np.array([]) #for saving raw data to analyze later
features_save = None #for saving raw data to analyze later
confidences_save = np.array([])
confidences_cough = []
big_x_save = None
X=[]
X_old=np.zeros(MIN_SAMPS_HALF)
features=None
cough_audio = None
cough_idx = 1
cough_countdown = 0
gpio_countdown = 0
num_coughs = 0
backlog=np.array([])
in_cough = False
s=None
TEST=False
doa=np.array([],dtype=int)
mic_type = None
Mic_tuning = None
pixel_ring = None
model_idx = [0,0]
times = np.array([])

time_string = datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
path_results = os.path.join('Results',time_string)
path_wavs = os.path.join(path_results,'wavs')
os.makedirs(path_results,exist_ok=True)
os.makedirs(path_wavs,exist_ok=True)
path_cough_csv = os.path.join(path_results,"cough.csv") #r"/home/rslsync/cough/cough.csv"
wc_enrollment_path = 'whosecough/wc_enrollments.npy'

def setup_mic(args):
    global Mic_tuning, pixel_ring, sr_mic, sr_multiplier, mic_type, use_gpio, mic_channels,THRESH
    
    try:    
        from usb_4_mic_array.tuning import Tuning
        from pixel_ring import usb_pixel_ring_v2
        import usb.core
        import usb.util

        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev:
            Mic_tuning = Tuning(dev)
            if not(args.no_led):
                pixel_ring = usb_pixel_ring_v2.PixelRing(dev)
                pixel_ring.off()
                pixel_ring.set_vad_led(0)
                #pixel_ring.set_brightness(0x02)
    except Exception as e:
        if args.mic_type == 'respeaker':
            raise e
        else:
            print('NOTE!: Couldnt open respeaker device, not using it for LEDs')

    if args.mic_type == 'respeaker':
        sr_mic = 16000
        mic_channels = 1#5
        THRESH = .75
    elif args.mic_type == 'omni':
        if not(args.no_led):
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(18,GPIO.OUT)
    else:
        raise ValueError ('Invalid mic type specified')
    
    mic_type = args.mic_type
    
    print("Mic type:", mic_type)
    print("Mic samplerate:", sr_mic)

def get_model(args):
    
    global model_idx
    
    model_path = "model.220_vggish_cv2_pt4thconv.tflite" if args.model_type == 'vggish' else "model_vggish_cv2.1757-0.21.tflite"
    model_path = os.path.join('models', model_path)
    print("Model path:", model_path)

    # Set up the tflite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    model_idx[0] = interpreter.get_input_details()[0]['index']
    model_idx[1] = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    # Adjust the input shape to take the batch size
    interpreter.resize_tensor_input(model_idx[0], [batch_size, input_shape[1], input_shape[2], input_shape[3]])
    interpreter.allocate_tensors()
    
    return(interpreter)

def start_audio(args):
    
    global s, sr_mic
    
    test_audio = None

    #Don't use real-time audio, load from file instead
    if args.test_from_file:
        if path_test_audio.endswith('.npy'):
            test_audio = np.load(file)
        else:
            test_audio,_ = librosa.load(path_test_audio,sr=sr)
            test_audio=test_audio
        sr_mic=sr
        print("Loaded data from numpy array:",path_test_audio)
    #Start audio
    else:
        pa = pyaudio.PyAudio()
        s = pa.open(
            input=True,
            format=pyaudio.paFloat32,
            channels=mic_channels,
            rate=sr_mic,
            frames_per_buffer=FRAMES_PER_BUFFER,
            #stream_callback=callback,
            input_device_index = 2
            )
        s.start_stream()
        print("Started streaming audio")
    
    if (sr_mic == 48000 and mic_channels>1) or (mic_channels>1 and mic_type != 'respeaker'):
        raise ValueError('Incompatible mic samplerate, mic channels, and/or mic type')

    return(test_audio)

#Store audio data in a numpy arary
def callback(in_data,frame_count, time_info, status):
    global X, full_arr, TEST

    if status:
        print("PyAudio callback status:", status)

    if TEST:
        x = in_data
    else:
        x = np.frombuffer(in_data, dtype=np.float32)
        if sr_mic == 48000:
            x = np.mean(x.reshape(-1, sr_multiplier), axis=1)
        if mic_channels > 1:
            x = np.mean(x.reshape(-1, mic_channels)[:,1:], axis=1)
            #x = x.reshape(-1, mic_channels)[:,2]
    #X = np.append(X,x)
    X += list(x)
    #full_arr = np.append(full_arr, x)
    return (None, pyaudio.paContinue)

#Mel spectrogram
def feature_transform(x):
    S = mel_features.log_mel_spectrogram(x, audio_sample_rate=sr, log_offset=vggish_params.LOG_OFFSET,
                                         window_length_secs=window, hop_length_secs=hop,
                                         num_mel_bins=nmels, lower_edge_hertz=vggish_params.MEL_MIN_HZ,
                                         upper_edge_hertz=vggish_params.MEL_MAX_HZ)
    
    S = np.expand_dims(S.T,axis=0)
    return(S)

def get_print_format(arr, dtype_int = False):
    arr_print = ''
    for a in arr:
        msg = ', ' if len(arr_print)>0 else ''
        val = '{:d}'.format(int(a)) if dtype_int else '{:.3f}'.format(a)
        arr_print += msg + val
    return(arr_print)

cnt=0
def process_next_frame(args, logger=None):
    
    global X, X_old, features, features_save, confidences_save, confidences_cough, big_x_save, cough_audio, cough_idx,\
    cough_countdown, gpio_countdown, num_coughs, in_cough, doa, backlog, times,cnt, first_time

    # Classify if have enough samples
    if len(X) + len(X_old) >= MIN_SAMPS:
        
        # Get mel spec
        X_new = np.array(X[:MIN_SAMPS_HALF]) #just get enough samples for 50% of a new frame of features
        X = X[MIN_SAMPS_HALF:]
        X_samp = np.append(X_old,X_new)#clip out samples running this time
        cough_audio = np.vstack((cough_audio, X_new)) if cough_audio is not None else X_new
        
        new_features = feature_transform(X_samp)
        features = np.concatenate((features, new_features),axis=0) if features is not None else new_features
        if args.test_from_file:
            features_save = np.concatenate((features_save, new_features),axis=0) if features_save is not None else new_features
        
        os.makedirs('test_wavs',exist_ok=True)
        #wav_path = os.path.join('test_wavs', 'wav_{}.wav'.format(cnt))
        #sf.write(wav_path, X_samp, sr,'PCM_16')
        #cnt+=1
        X_old = X_samp[MIN_SAMPS_HALF:]
        #if cnt>33:
        #    raise
        
    if features is not None and features.shape[0] >= batch_size:

        cough_this_batch = False
        
        # Assemble mel spec frames into batches of 19 frames
        if args.test_from_file:
            start=time.time()
            
        # Turn into format model wants (batch, frames, nmels, channels)
        big_x = np.expand_dims(np.transpose(features,(0,2,1)),-1).astype(np.float32)
        features = None
        
        if args.test_from_file:
            big_x_save = np.concatenate((big_x_save,np.expand_dims(big_x,0)),axis=0) if big_x_save is not None else np.expand_dims(big_x,0)

        # Do the inference
        interpreter.set_tensor(model_idx[0], big_x)
        interpreter.invoke()
        results = interpreter.get_tensor(model_idx[1]).squeeze()
        
        # Store times if testing from file
        if args.test_from_file:
            end=time.time()
            #print('Inference time: {:.2f}ms'.format(1000*(end-start)))
            times = np.append(times, end-start)
        
        # Running average of backlog
        backlog = np.append(backlog,len(X))
        if len(backlog) > 100:
            backlog = backlog[len(backlog)-100:]
            
        confidences = np.squeeze(results)
        if args.test_from_file:
            confidences_save = np.append(confidences_save, confidences)
        classifications = np.array([1 if r>THRESH else 0 for r in confidences])
        
        # Print results
        if args.verbose:
            confidences_print = [float('{:.2f}'.format(f*100)) for f in confidences]
            #msg = "Confidences: {0:f}%".format(100*np.mean(results.squeeze()))
            #msg2 = ", Prediction time: {0:.2f} ms".format(1000*(end-start))
            msg2 = ", backlog: {:d}".format(int(np.mean(backlog)))
            print("Confidences:", confidences_print, msg2)
            
        #print(classifications)

        # Determine if cough occurred
        for r in range(len(results)):
            #print("cc beginning", cough_countdown)
            #print(r, in_cough, cough_countdown)

            if classifications[r]: # means cough detected

                cough_this_batch=True # at least one cough frame detected this batch
                cough_countdown = COUGH_LATCH_REFILL #refill countdown whenever a cough is detected
                confidences_cough += [confidences[r]] #add confidence to confidence array
                
                # If not currently in cough, start cough logging. Otherwise, do nothing.
                if not in_cough:
                    in_cough = True
                    #if args.verbose:
                        #print("\n --> Cough start detected.")
                    if not(args.no_led):
                        if not args.whosecough:
                            if mic_type == 'omni':
                                GPIO.output(18,GPIO.HIGH)
                            if pixel_ring is not None:
                                pixel_ring.wakeup()
                        
            else: #means no cough detected

                if in_cough and cough_countdown <1:
                                        
                    date =datetime.now().strftime('%Y.%m.%d_%H-%M-%S')

                    #Get the confidences and classifications for past samples
                    end_samps = batch_size - r
                    start_samps = end_samps + cough_idx + 1 #go one back from when cough started to get some context
                    cough_audio_save = cough_audio[-start_samps:-end_samps].flatten()
                    #print(r,start_samps, end_samps, cough_audio_save.shape)
                    
                    # Get duration - this method is not perfect but gets pretty close
                    duration = (MIN_SAMPS_HALF * (cough_idx-COUGH_LATCH_REFILL))/sr
                    doa_mode = int(stats.mode(doa).mode[0]) if mic_type == 'respeaker' else 0

                    confidences_cough_print = [float('{:.2f}'.format(f*100)) for f in confidences_cough]
                    cur_time = datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
                    
                    print("\n", cur_time, "--> Cough with sigmoid confidences:", confidences_cough_print,
                          "approx duration: {0:.3f}s, doa:{1:d}".format(duration, doa_mode))

                    # Write data to csv
                    logger.writerow([date, duration, get_print_format(confidences_cough, dtype_int=False),
                                     get_print_format(np.array([1 if r>THRESH else 0 for r in confidences_cough]), dtype_int=True), doa_mode])
                    
                    # Save wav
                    wav_path = os.path.join(path_wavs, 'wav_{}_{}.wav'.format(num_coughs,date))
                    sf.write(wav_path, cough_audio_save, sr,'PCM_16')
                    #npy_path = os.path.join(path_wavs, 'audio_{}_{}.npy'.format(num_coughs,date))
                    #np.save(npy_path,cough_audio_save)
                    
                    if args.whosecough:
                        #start_wc = time.time()
                        emb = get_embedding_rpi(wc_net,path='',samp=cough_audio_save)
                        #emb_wc = time.time()
                        user_id = wc_get_results(emb, enrollment_centroids, thresh=WC_THRESH)
                        if pixel_ring is not None:
                            if user_id ==1:
                                pixel_ring.mono(0x0000FF)
                            elif user_id == 2:
                                pixel_ring.mono(0x00FF00)
                            elif user_id==-1:
                                pixel_ring.mono(0xFF0000)
                            else:
                                raise ValueError("Whosecough detected invalid user.")
                        #end_wc = time.time()
                        #print('Whosecough total inf time: {:.2f}ms, emb inf time: {:.2f}ms'.format(1000*(end_wc-start_wc),1000*(emb_wc-start_wc))) 
                    
                    in_cough = False
                    doa = np.array([],dtype=int)
                    cough_idx = 1
                    num_coughs += 1
                    confidences_cough=[]

            if in_cough:
                cough_countdown -=1
                cough_idx +=1
                if mic_type == 'respeaker':
                    doa = np.append(doa, Mic_tuning.direction)

        # Clip out old cough_audio
        cough_audio_len = cough_audio.shape[0]
        if cough_audio_len > 12:
            cough_audio = cough_audio[cough_audio_len-12:]

        # Turn off GPIO if no cough detected this batch
        if not(args.no_led) and not(cough_this_batch):
            if mic_type == 'omni':
                GPIO.output(18,GPIO.LOW)
            if pixel_ring is not None:
                if args.whosecough:
                    pixel_ring.mono(0)
                else:
                    pixel_ring.off()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='conv', help="Model type - 'conv' or 'vggish'")
    parser.add_argument('--test_from_file', action='store_true', default=False, help="Whether to use realtime audio (default) or load data from a file")
    parser.add_argument('--verbose', action='store_true', default=False, help="Whether to print sigmoid confidence on each predict")
    parser.add_argument('--no_led', action='store_true', default=False, help="Will not setup pixel ring for respeaker or gpio 18 for LED for omni")
    parser.add_argument('--mic_type', default='omni', help="Mic type - 'respeaker' or 'omni'")
    parser.add_argument('--whosecough', action='store_true', default=False, help='Whether to determine user when detect a cough')
    args = parser.parse_args()
   
    if args.model_type not in ['conv','vggish']:
        raise ValueError("invalid model type, must be 'conv' or 'vggish'")

    if args.whosecough:
        from rpi_inference import get_model_rpi, wc_get_results, get_enrollments, get_embedding_rpi
        wc_net = get_model_rpi()
        enrollment_centroids = get_enrollments(wc_enrollment_path)

    if not(args.test_from_file):
        setup_mic(args)
    interpreter = get_model(args)
    test_audio = start_audio(args)

    with open(path_cough_csv, "a") as file:
        writer = csv.writer(file, delimiter = ";")
        writer.writerow(["Time", "Approx Duration (s)", "Cough Confidences", "Cough Classifications", "DOA",])

        #Main Loop
        if args.test_from_file:
            TEST=True
            n = FRAMES_PER_BUFFER
            #for i in range(10):
            for i in np.arange(0,len(test_audio),n):    
                callback(test_audio[i:i+n],None,None,None)
                process_next_frame(args, writer)
            path_save = os.path.join(path_results, 'features.npy')
            np.save(path_save, features_save)
            np.save(os.path.join(path_results, 'big_x.npy'),big_x_save)
            np.save(os.path.join(path_results, 'confidences.npy'),confidences_save)
            print('Avg inference time: {:.2f}ms, {:.2f}ms'.format(1000*np.mean(times[1:])))
            print("Saved features to:", path_save, "Features size:", features_save.shape)
        else:
            try:
                while(1):
                    try:
                        data = s.read(2100)
                        callback(data,None,None,None)
                        process_next_frame(args, writer)
                        #time.sleep(.001)
                        file.flush()
                    # if the buffer overflows, will get this error
                    except OSError as err:
                        print("OS error: {0}".format(err))

                        # clear out the buffers and start collecting again
                        X=np.array([])
                        X_old=np.zeros(MIN_SAMPS_HALF)
                        start_audio(args)
            except KeyboardInterrupt:
                s.stop_stream()
                s.close()
                if pixel_ring is not None:
                    pixel_ring.off()
                #np.save('full_arr.npy',full_arr)
                print("Done!")
