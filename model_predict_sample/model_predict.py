# This script demonstrates how the cough detection model makes a prediction
import numpy as np
import librosa
import tensorflow as tf

sr = 8000 # sample rate
MIN_SAMPS = 6561 # don't ask why we use this number, just trust me

batch_size = 1
model_idx = [0,0]

def get_model(model_path):
    global model_idx

    # Set up the tflite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    model_idx[0] = interpreter.get_input_details()[0]['index']
    model_idx[1] = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    # Adjust the input shape to take the batch size
    interpreter.resize_tensor_input(model_idx[0], [batch_size] + list(input_shape[1:]))
    interpreter.allocate_tensors()

    return (interpreter)

def predict(X, interpreter):

    # Do the inference
    interpreter.set_tensor(model_idx[0], X)
    interpreter.invoke()
    output = interpreter.get_tensor(model_idx[1]).squeeze()

    # model output is a sigmoid. Close to 1 = cough, close to 0 = not a cough
    cough = 1 if output > .5 else 0

    return(cough, output)


if __name__ == '__main__':

    model_path = "model.sample-cnn_CV-0_8khz_basic_7L-128F_0.750_bs1.tflite"
    interpreter = get_model(model_path)

    for i, wav_file in enumerate(['wav_cough.wav', 'wav_no_cough.wav']):
        X, _ = librosa.load(wav_file, sr=sr)
        X = X[:MIN_SAMPS] #clip to the right size
        X = np.expand_dims(np.expand_dims(X,-1),0) # needs to be of shape (batch_size, samp_len, 1)
        cough, output = predict(X, interpreter)
        wav_typ = 'cough' if i == 0 else 'no_cough'
        print("Wav_type:", wav_typ,"| Cough?", cough, "| Model output: {:.8f}".format(output))


