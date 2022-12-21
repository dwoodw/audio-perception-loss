# For inference on the datasets 

import tensorflow as tf
import testModel
import parse
import numpy as np
from Config import config_ingredient
from sacred import Experiment
import testModel

import pandas as pd
from openpyxl import Workbook
import scipy
import museval
from pesq import pesq

ex = Experiment('Unet Keras Training', ingredients=[config_ingredient])

def get_padding(shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''
        return [shape[0], 2, 2, shape[3]]

def feature_labels(element,source_names):
    feature = element['audio']
    labels = {k : el for k, el in element.items() if k in source_names}

    return feature, labels

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def write_excel(csv_dataset, name):
    #create title headers for the data in excel 
    with pd.ExcelWriter(name, engine="openpyxl", mode='w') as writer:  
        excel_title_data = pd.DataFrame(data = csv_dataset.keys())
        excel_title_data.to_excel(writer, sheet_name = 'sheet', index=False, startrow=0, startcol=0 ,header=False)

        # .T is needed because data is written in rows by default 
        excel_data = pd.DataFrame(data = csv_dataset)
        excel_data.to_excel(writer, sheet_name = 'sheet', startrow=1, startcol=0, index=False)
    return

def get_features(audio, model_config):


    features = np.zeros((1, 2, 4))
    sdr, isr, sir, sar = museval.evaluate(audio[1, :, :],  audio[0, :,:], win=44100, hop=44100, mode='v4', padding=True)
    feats = [sdr, isr, sir, sar]
    for idx in range(len(feats)):
        #print('init feats sahpe', feats[idx].shape)
        feats[idx] = feats[idx].mean(axis = 1)
        #print('feats avg ', feats[idx].shape)
        for channels in range(2):
            if np.isnan(feats[idx][channels]):
                feats[idx][channels] = -30
            elif np.isinf(feats[idx][channels]):
                feats[idx][channels] = 50
            
            #print('feature shape:', features.shape)
            #print('feats shape:', feats[idx].shape)
            features[0, :, idx] = feats[idx]

  
        for idx in range(2):
            audio[idx,:, model_config['audio_len']:model_config['audio_len']+model_config['features_len']] = features


        #print(audio.shape)
        audio_dict = dict()
        audio_dict['input_1'] =  np.expand_dims(audio, axis = 0)
    
    return audio_dict

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    #get the dictionary where key = name of dataset value = list of csv files and signal folders (subfolders)
    dataset_dict = parse.get_dataset_filenames(model_config)
    #list of datasets and weights for random sampling
    datasets_list, subdatasets_list, weights_list = parse.get_sampling_weights(dataset_dict)
    #print(dataset_dict, '\n')
    #print(datasets_list, '\n')
    #print(subdatasets_list, '\n')


    csv_dataset_list = list()    
    #load the csv files for the subdatasets
    for datasets in dataset_dict:
        for subdatasets in dataset_dict[datasets][0]:
            #parse all the datasets e.g ...data/SiSEC08/SisEC08_anonymized.csv and append all data to csv dataset list
            csv_data = parse.parseCSV(subdatasets)
            csv_dataset_list.append(csv_data)

    disc_input_shape = [model_config["batch_size"], 2, 2,  model_config['audio_len']+model_config['features_len']]  # Shape of input

    sep_input_shape = get_padding(np.array(disc_input_shape)) 
    sep_output_shape = [1]
    print('sep input', sep_input_shape, type(sep_input_shape))
    print('sep output', sep_output_shape, type(sep_output_shape))

    inputs = tf.keras.Input(shape = sep_input_shape[1:], batch_size = 1) #batch_size=model_config['batch_size'])
    print('inputs', inputs)
    model = dict()
    outputs = dict()
    for source in model_config["source_names"]:
        model[source] = testModel.Unet(model_config)
        outputs[source] = model[source](inputs)

    unet_model = tf.keras.Model(inputs=inputs, outputs = outputs)
    #print(unet_model.summary())

    #print(model['Ratingscore'])
    unet_model.load_weights('/home/dwoodward/masters/audio-perception-loss/models/conv2/29110.017-l-2617.08203')

    type(unet_model)
    #for every imported csv dataset list (6)
    for data_idx in range(len(csv_dataset_list)):
        #create a predicted rating score list of zeros for each parsed csv file

        csv_dataset_list[data_idx]['predicted_ratingscore'] = zerolistmaker(len(csv_dataset_list[data_idx]['Trials']))

        #iterate through the different csv/signal folders for each dataset 
        #dataset_dict is a dictionary for each dataset (eg key - SAOC, SiSEC08) with a nested list of the path(s) the csv and signals folder
        #(1-3)
        for idx in range(len(dataset_dict[datasets_list[data_idx]][1])):
            #If statement to find the index for the folder which corresponds to the csv file
            if dataset_dict[datasets_list[data_idx]][0][idx].find(subdatasets_list[data_idx]) != -1:
                #audio path to the folder e.g /home/dwoodward/masters/data/PEASS-DB/Signals
                audio_path = dataset_dict[datasets_list[data_idx]][1][idx]

                #loop for length of each dataset to import data
                for num_trial in range(len(csv_dataset_list[data_idx]['Trials'])):
                    data = parse.parseAudio(csv_dataset_list[data_idx], audio_path, batch_idx= num_trial, inference = 1)
                    data_length = data['audio_test'][0].shape[1]
                    audio = np.zeros((2, 2, model_config['audio_len']+model_config['features_len'],))
                    audio[0,:, :data_length] = data['audio_test'][0]
                    audio[1,:, :data_length] = data['audio_ref'][0]
                    #print(data['audio_test'])
                    #audio = np.expand_dims(audio, axis=0)

                    audio_dict = dict()
                    audio_dict =  get_features(audio, model_config)
                    #print(audio_dict['input_1'].shape)

                    ##calculate ratingscore
                    rating = unet_model.predict(audio_dict, batch_size=1)
                    print('rating score: ',(rating['Ratingscore']))
                    csv_dataset_list[data_idx]['predicted_ratingscore'][num_trial] = rating['Ratingscore'][0,0]

                    print(num_trial, ' of ', len(csv_dataset_list[data_idx]['Trials']))

        #sheet_name = dataset_dict
        write_excel(csv_dataset_list[data_idx], str(datasets_list[data_idx]) + str(data_idx) + '.xlsx')
                    
                    

    
        
                    


                





    