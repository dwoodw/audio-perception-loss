import os
import tensorflow as tf
import numpy as np
import parse
import matplotlib as plot
import matplotlib.pyplot as plot
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


model_config = {'augmentation' : False, # Random attenuation of source signals to improve generalisation performance (data augmentation)
                'batch_size' : 1, # Batch size
                'datasets' : 'all', # use all datasets if more than one available for a given task
                'data_path' : "/home/dwoodward/masters/data/SASSEC/SASSEC_anonymized.csv", # Set this to where the preprocessed dataset should be saved
                'audio_path' : "/home/dwoodward/masters/data/SASSEC/Signals",
                'epochs' : 100, #number of epochs to train
                "estimates_path" : "/home/daniel/audioSource/asre_transfer/asre_tf22/estimates/", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                'expected_sr': 44100,  # Downsample all audio input to this sampling rate
                'fft_size' : 4096,
                'hop' : 1024,
                "inhouse_path_train" : "/home/daniel/audioSource/datasets/2stem/train/",
                "inhouse_path_valid" : "/home/daniel/audioSource/datasets/2stem/valid/",
                "init_sup_sep_lr" : 1e-4, # Supervised separator learning rate
                'keepFreqs' : 2048,
                "log_dir" : "logs", # Base folder for logs files
                "model_base_dir" : "/home/daniel/Uni/BSS_metric/estimates/", # Base folder for model checkpoints
                'network' : 'unet', # Type of network architecture, either unet (our model) or unet_spectrogram (Jansson et al 2017 model)
                "num_frames": 1024 * 319 + 4096, # DESIRED number of time frames in the output waveform per samples (could be changed when using valid padding)
                "audio_len": 500000,
                "num_snippets_per_track" : 10,# If train_rand_mode is 'per_min', then we are grabbing 10 snippets per minute. If train_rand_mode is 'per_song' then we are grabbing 10 snippets per track
                'num_workers' : 7, # Number of processes used for each TF map operation used when loading the dataset
                'num_projections' : 5,
                'mono_downmix' : False,
                'epoch_it' : 5,
                'return_spectrogram' : True,
                'restore_checkpoint' : False,
                'task' : 'metric', # Type of separation task. 'vocals' : Separate music into voice and accompaniment. 'multi_instrument': Separate music into guitar, bass, vocals, drums and other (Sisec)
                'loss_function' : 'weighted_l1',#presently implemented costs are 'weighted_l1' and 'l1', if 'loss_function' is not included in config, it will default to l1
                'source_names' : ['Ratingscore'],
                'keepFreqs' : 2048,
                }

def create_dataset_shapes(output_shape, input_shape):
    output_shapes = dict()
    output_shapes.update({'audio' :   input_shape})
    output_shapes.update({'Ratingscore' : output_shape})

    return output_shapes

def create_dataset_types():
    output_types = dict()
    output_types.update({'audio' :   tf.float32})
    output_types.update({'Ratingscore' : tf.float32})

    return output_types

def generate_dataset(model_config):
    csv_data = parse.parseCSV(model_config['data_path'])

    data = parse.parseAudio(csv_data, model_config['audio_path'], batch=1)
    data_length = data['audio_test'][0].shape[1]
    #print(data_length)
    audio = np.zeros((model_config['audio_len'], 2, 2))
    audio[:data_length,:,0] = np.transpose(data['audio_test'][0])
    audio[:data_length,:,1] = np.transpose(data['audio_ref'][0])


    audio_dict = dict()
    audio_dict['audio'] =  audio
    audio_dict['Ratingscore'] =  data['Ratingscore']
    #print(audio_dict['audio'].shape)

    yield audio_dict

def stft(audio):
    keepFreqs = model_config['keepFreqs']
    frame_len= model_config['fft_size']
    hop = model_config['hop']
    frames = 512 #number of frames taken from audio
    sr = 44100
    stft_time_max = 500000/sr #calculates number of time bins and converts to seconds

    split1, split2 = tf.split(audio, 2, 3)
    diff = tf.math.subtract(split1, split2)
    diff_flat = tf.reshape(diff, [tf.shape(diff)[0], tf.shape(diff)[1], tf.shape(diff)[2]])

    diff_flat = tf.keras.layers.Permute((2,1))(diff_flat)
    stfts = tf.signal.stft(diff_flat, frame_length=frame_len, frame_step=hop, fft_length=frame_len, window_fn=tf.signal.hann_window)
    print(stfts.numpy().shape)
    stfts = tf.keras.layers.Permute((1,3,2))(stfts)
    stfts = tf.reverse(stfts, [2])
    mix_mag_o = tf.abs(stfts)
    mix_mag = mix_mag_o[:,:,:keepFreqs,:]
    
    return mix_mag

def display(stfts):
    # Plot the spectrogram
    frames = 512 #number of frames taken from audio
    sr = 44100
    frame_length = model_config['fft_size']
    frame_step = model_config['hop']
    stft_time_max = (frames*frame_step - frame_step + frame_length)/sr #calculates number of time bins and converts to seconds

    #for log scale
    max_stft = max(stfts.flatten())
    print(max_stft)
    stfts = 10*np.log((stfts/max_stft)+1e-3)
    ax = plot.subplot(111)
    im = plot.imshow(stfts, cmap=plot.get_cmap('plasma'),  vmin=stfts.min(), vmax=stfts.max(),  extent=[0, stft_time_max, 0, 22050], aspect='auto')
    plot.xlabel('Time (Seconds)')
    plot.ylabel('Frequency (Hz)')
    plot.title('Short Time Fourier Transform Spectrogram')

    # colorbar
    cbar = plot.colorbar()
    cbar.set_label('Db level')
    plot.show()

def get_padding(shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''
        return [shape[0], shape[1], 2, 2]

def feature_labels(element,source_names):
    feature = element['audio']
    labels = {k : el for k, el in element.items() if k in source_names}

    return feature, labels

def main():
    disc_input_shape = [model_config["batch_size"],  model_config['audio_len'], 2, 2]  # Shape of input

    sep_input_shape = get_padding(np.array(disc_input_shape)) 
    sep_output_shape = [1]
    print('sep input', sep_input_shape, type(sep_input_shape))
    print('sep output', sep_output_shape, type(sep_output_shape))
    #sep input [1, 440999, 2, 2] <class 'list'>
    #sep output [1] <class 'list'>


    print(create_dataset_types())
    print(create_dataset_shapes(sep_output_shape, sep_input_shape[1:4]))
    dataset = tf.data.Dataset.from_generator(lambda: generate_dataset(model_config), 
                                                                    (create_dataset_types()), 
                                                                    (create_dataset_shapes(sep_output_shape, sep_input_shape[1:4])))

    dataset = dataset.map(lambda x : feature_labels(x, model_config['source_names']))
    train_dataset = dataset.batch(model_config["batch_size"],drop_remainder = True)
    inputs = tf.keras.Input(shape = sep_input_shape[1:], batch_size=model_config['batch_size'])


    for element in train_dataset:
        audio_spec = abs(stft((element[0]))).numpy()

    display(audio_spec[0,0,:,:])
    print("\n\nDataset Type:", type(train_dataset))

    print("Elements")
    for element in train_dataset:
        print(type(element))
        print(len(element))
        print(type(element[0]))
        print(type(element[1]))
        print(element[1].keys())
        print((element[1]["Ratingscore"].numpy()[0][0]))

if __name__ == '__main__':
    main()