from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import load_model
from model_mobilenet import relu6
from tensorflow.keras.layers import DepthwiseConv2D
import os, time
import numpy as np
import config
from config import config as c
from config import config_data as c_d
from model_mobilenet import MobileNetv2_simple, MobileNetv2
from model_conv import conv_model
from model_samp_cnn import SampleCNN, ModelConfig, AudioVarianceScaling
from metrics import F1Score
from model_vggish import vggish_model
from evaluate_full_file import DataGeneratorFullFile
import tfcoreml
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

models_folder = os.path.join(os.path.dirname(c_d['folder_data']), 'models')
tflite_folder = os.path.join(models_folder, 'tflite')
mlmodels_folder = os.path.join(models_folder, 'mlmodels')
os.makedirs(tflite_folder, exist_ok=True)
os.makedirs(mlmodels_folder, exist_ok=True)

def convert_to_tflite_and_mlmodel():

    # batch_size=8
    for batch_size in [1]:
        folder = os.path.join(os.path.dirname(c_d['folder_data']), 'logs_and_checkpoints')
        training_run_name = 'sample-cnn_CV-0_8khz_basic_7L-128F_0'
        model_name = 'model.sample-cnn_CV-0_8khz_basic_7L-128F_0.750'

        path_model = os.path.join(folder, training_run_name, '{}.h5'.format(model_name))

        with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):

            # model = conv_model(batch_size=batch_size, llf_pretrain=True)
            model = SampleCNN(ModelConfig(block='res1', multi=False, num_blocks=7, init_features=128, batch_size=batch_size))
            model.load_weights(path_model)

            # model = load_model(path_model, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
            #                                 'AudioVarianceScaling': AudioVarianceScaling, 'F1Score': F1Score})
            # # print(model.layers)
            # for layer in model.layers:
            #     print(layer.name)
            # raise

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            open(os.path.join(folder, training_run_name, '{}_bs{}.tflite'.format(model_name, batch_size)), "wb").write(tflite_model)
            continue

            # from shutil import copyfile
            # copyfile(path_model, os.path.join(folder, training_run_name, '{}.h5'.format(model_name)))

            print(model.inputs[0].name)
            model.summary()
            input_name = model.inputs[0].name.split(':')[0]
            keras_output_node_name = model.outputs[0].name.split(':')[0]
            graph_output_node_name = keras_output_node_name.split('/')[-1]

            model = tfcoreml.convert(path_model,
                                     input_name_shape_dict={input_name: (batch_size, config.MAX_SAMPS)},
                                     output_feature_names=[graph_output_node_name],
                                     minimum_ios_deployment_target='13',
                                     add_custom_layers=True,
                                     custom_conversion_functions={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}
                                     )
            model.save(os.path.join(folder, training_run_name, r'{}_bs{}.mlmodel'.format(model_name, batch_size)))

def test_model_speed_tflite():

    model_type = 'vggish'#'sample_cnn' #'sample_cnn' #'conv'

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    time_results =[]
    for batch_size in [1]:

        model_path = os.path.join(tflite_folder, r'{}_bs{}.tflite'.format(model_type,batch_size))

        model = conv_model(batch_size=batch_size)
        # model = MobileNetv2_simple(batch_size=batch_size)# conv_model(batch_size=batch_size)
        # model = MobileNetv2(batch_size=batch_size)
        # model = SampleCNN(ModelConfig(block='res1', multi=False, num_blocks=7, init_features=16))
        # model = vggish_model(use_mel_spec=False, llf_pretrain=True, pretrain=False)
        # model.summary()

        tf.saved_model.save(model, './')

        for opt in [False, True]:#True]:

            converter = tf.lite.TFLiteConverter.from_saved_model('./', )
            if opt:
                # dg = DataGeneratorFullFile('',0)#'sample-cnn', 0)
                # def representative_dataset_gen():
                #     for _ in range(5):
                #         yield [dg.tflite_converter_get_item(batch_size)]

                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                dg = DataGeneratorFullFile('sample-cnn', 0)
                # converter.representative_dataset = representative_dataset_gen

            # converter.inference_input_type = tf.int16
            tflite_model = converter.convert()

            open(model_path, "wb").write(tflite_model)

            model_idx = [0, 0]

            interpreter = tf.lite.Interpreter(model_path=model_path)

            model_idx[0] = interpreter.get_input_details()[0]['index']
            model_idx[1] = interpreter.get_output_details()[0]['index']
            input_shape = interpreter.get_input_details()[0]['shape']

            # print(input_shape)
            input_shape_full = [batch_size] + list(input_shape[1:])
            samp = np.ones(tuple(input_shape_full)).astype(np.float32)#int16

            # print(samp.shape)
            # print(input_shape_full)

            # Adjust the input shape to take the batch size
            interpreter.resize_tensor_input(model_idx[0], input_shape_full)
            interpreter.allocate_tensors()

            #input_details = interpreter.get_input_details()[0]
            #output_details = interpreter.get_output_details()[0]
            #scale_out, zero_point_out = output_details['quantization']
            #scale, zero_point = input_details['quantization']
            #tflite_integer_input = samp / 1 #/ scale + zero_point
            #tflite_integer_input = tflite_integer_input.astype(input_details['dtype'])

            times = []
            for i in range(100):
                start = time.time()
                interpreter.set_tensor(model_idx[0], samp)
                interpreter.invoke()
                output = interpreter.get_tensor(model_idx[1]).squeeze()
                # interpreter.set_tensor(input_details['index'], tflite_integer_input)
                # interpreter.invoke()
                # tflite_integer_output = interpreter.get_tensor(output_details['index'])
                end = time.time()
                # Manually dequantize the output from integer to float
                # tflite_output = tflite_integer_output.astype(np.float32)
                # tflite_output = (tflite_output - zero_point_out) * scale_out

                time.sleep(.5)
                times.append((end-start)*1000/batch_size)

            print("BATCH SIZE:", batch_size, "| Quantization:", opt)
            print("Tflite Mean: {0:.12f} ms, Tflite Median: {1:.12f} ms".format(np.mean(times), np.median(times)))
            print("")
            time_results.append([batch_size, opt, np.mean(times), np.median(times)])
            print(times[:5], times[-10:])

        for res in time_results:
            print("BATCH SIZE:", res[0], "| Quantization:", res[1])
            print("Tflite Mean: {0:.12f} ms, Tflite Median: {1:.12f} ms".format(res[2], res[3]))
            print("")

def run_model():

    model_path = r'Z:\research\cough_count\logs_and_checkpoints\sample-cnn_CV-0_8khz_basic_7L-128F_0\model.sample-cnn_CV-0_8khz_basic_7L-128F_0.750_bs2.tflite'
    model_idx = [0, 0]

    interpreter = tf.lite.Interpreter(model_path=model_path)
    model_idx[0] = interpreter.get_input_details()[0]['index']
    model_idx[1] = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    print(input_shape)

    samp = np.ones((2, 6561, 1)).astype(np.float32)
    print(samp.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(model_idx[0], samp)
    interpreter.invoke()
    output = interpreter.get_tensor(model_idx[1]).squeeze()
    print(output)


def run_tvm():
    import tvm
    import tvm.relay as relay
    import librosa

    model = conv_model(batch_size=1)
    mod, params = relay.frontend.from_keras(model)
    target = 'cuda'
    ctx = tvm.gpu(0)
    with relay.build_config(opt_level=3):
        executor = relay.build_module.create_executor('graph', mod, ctx, target)

    for i, wav_file in enumerate(['wav_cough.wav', 'wav_no_cough.wav']):
        X, _ = librosa.load(wav_file, sr=c['sr'])
        X = X[:config.MAX_SAMPS]
        X = np.expand_dims(X,0)

        dtype = 'float32'
        tvm_out = executor.evaluate()(tvm.nd.array(X.astype(dtype)), **params)
        top1_tvm = tvm_out.asnumpy()[0]
        print(top1_tvm)


if __name__ == '__main__':

    # model=conv_model_with_MF_or_llf(True)

    # path = './conv_model_pb'
    # os.makedirs(path, exist_ok=True)
    # tf.saved_model.save(model, path)

    convert_to_tflite_and_mlmodel()
    # test_model_speed_tflite()
    # run_model()
    # run_tvm()


