from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os
import math
import datetime
import random

import testModel
import parse
import museval 

ex = Experiment('Unet Keras Training', ingredients=[config_ingredient])

@ex.config
# Executed for training, sets the seed value to the Sacred config so that Sacred fixes the Python and Numpy RNG to the same state everytime.
def set_seed():
    seed = 1337

def generate_dataset(model_config):
        #get the dictionary where key = name of dataset value = list of csv files and signal folders (subfolders)
    dataset_dict = parse.get_dataset_filenames(model_config)

    #list of datasets and weights for random sampling
    datasets_list, subdatasets_list, weights_list = parse.get_sampling_weights(dataset_dict)

    csv_dataset_list = list()    
    #load the csv files for the subdatasets
    for datasets in dataset_dict:
        for subdatasets in dataset_dict[datasets][0]:
            #parse all the datasets e.g ...data/SiSEC08/SisEC08_anonymized.csv and append all data to csv dataset list
            csv_data = parse.parseCSV(subdatasets)
            csv_dataset_list.append(csv_data)

    while True:
        #loading weights for the datasets for randomness
        dataset_selection = random.choices(list(range(len(subdatasets_list))), weights=weights_list)[0]

        #sort lists because on different machines and setups sometimes result in different orders leading to errors 
        #when trying to find the equivilent csv to folder below
        dataset_dict[datasets_list[dataset_selection]][1].sort()
        dataset_dict[datasets_list[dataset_selection]][0].sort()

        #loop across the length of csv files (1 refers to Signal folders, 0 would be .csv files)
        #dataset_dict = names from config file
        #datasets_list = similar to dictionary, a list of datasets which contains multiples for different folders
        for idx in range(len(dataset_dict[datasets_list[dataset_selection]][1])):
            #If statement to find the index for the folder which corresponds to the csv file
            if dataset_dict[datasets_list[dataset_selection]][0][idx].find(subdatasets_list[dataset_selection]) != -1:
                audio_path = dataset_dict[datasets_list[dataset_selection]][1][idx]

        data = parse.parseAudio(csv_dataset_list[dataset_selection], audio_path)
        data_length = data['audio_test'][0].shape[1]


        audio = np.zeros((2, 2, model_config['audio_len']+model_config['features_len'],))
        audio[0,:, :data_length] = data['audio_test'][0]
        audio[1,:, :data_length] = data['audio_ref'][0]

        features = np.zeros((2,4))
        sdr, isr, sir, sar = museval.evaluate(audio[1, :, :],  audio[0, :,:], win=44100, hop=44100, mode='v4', padding=True)
        feats = [sdr, isr, sir, sar]
        for idx in range(len(feats)):
            feats[idx] = feats[idx].mean(axis = 1)

            for channels in range(2):
                if np.isnan(feats[idx][channels]):
                    feats[idx][channels] = -30
                elif np.isinf(feats[idx][channels]):
                    feats[idx][channels] = 50
                

            features[:, idx] = feats[idx]
        for idx in range(2):
            audio[idx,:, model_config['audio_len']:model_config['audio_len']+model_config['features_len']] = features


        #print(features)
        audio_dict = dict()
        audio_dict['audio'] =  audio
        audio_dict['Ratingscore'] =  data['Ratingscore']

        yield audio_dict

def create_dataset_types():
    output_types = dict()
    output_types.update({'audio' :   tf.float32})
    output_types.update({'Ratingscore' : tf.float32})

    return output_types


def feature_labels(element,source_names):
    feature = element['audio']
    labels = {k : el for k, el in element.items() if k in source_names}

    return feature, labels

def get_padding(shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''
        return [shape[0], 2, 2, shape[3]]

def create_dataset_shapes(output_shape, input_shape):
    output_shapes = dict()
    output_shapes.update({'audio' :   input_shape})
    output_shapes.update({'Ratingscore' : output_shape})

    return output_shapes


@config_ingredient.capture
def train(model_config, experiment_id):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], 2, 2,  model_config['audio_len']+model_config['features_len']]  # Shape of input

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
    train_dataset = dataset.batch(model_config["batch_size"],drop_remainder = True).prefetch(2)
    print('gotten to train dataset\n', type(train_dataset))


    inputs = tf.keras.Input(shape = sep_input_shape[1:], batch_size=model_config['batch_size'])

    model = dict()
    outputs = dict()
    for source in model_config["source_names"]:
        model[source] = testModel.Unet(model_config)
        outputs[source] = model[source](inputs)
    
    unet_model = tf.keras.Model(inputs=inputs, outputs = outputs)

    #set up tensorboard log file
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    
    #set up checkpoint saving
    check_file = model_config['model_base_dir'] + '/'+str(experiment_id) + '/' + str(experiment_id) +'.{epoch:03d}-l-{loss:.5f}'
    model_check_callback = tf.keras.callbacks.ModelCheckpoint(filepath = check_file, save_weights_only = True, monitor = 'loss', mode = 'min', save_best_only = True)

    #reducing learning rate on plateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_lr=0.0000001)

    unet_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), loss = 'mse')
    unet_model.fit(train_dataset, epochs= model_config["epochs"], steps_per_epoch= model_config["epoch_it"],  callbacks = [model_check_callback])
    
 

    
@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")

    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    train()


