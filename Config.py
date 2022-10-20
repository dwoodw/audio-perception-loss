import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    # Base configuration
    model_config = {'augmentation' : False, # Random attenuation of source signals to improve generalisation performance (data augmentation)
                    'batch_size' : 8, # Batch size
                    'datasets' : ['PEASS-DB',  'SAOC',  'SASSEC',  'SiSEC08'], # use all datasets if more than one available for a given task
                    'data_base_dir' : "/home/dwoodward/masters/data/",
                    'data_path' : "/home/dwoodward/masters/data/SASSEC/SASSEC_anonymized.csv", # Set this to where the preprocessed dataset should be saved
                    'audio_path' : "/home/dwoodward/masters/data/SASSEC/Signals",
                    'epochs' : 100, #number of epochs to train
                    "estimates_path" : "/home/dwoodward/masters/audio-perception-loss/estimates", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!
                    'expected_sr': 44100,  # Downsample all audio input to this sampling rate
                    'fft_size' : 4096,
                    'hop' : 1024,
                    "inhouse_path_train" : "/home/daniel/audioSource/datasets/2stem/train/",
                    "inhouse_path_valid" : "/home/daniel/audioSource/datasets/2stem/valid/",
                    "init_sup_sep_lr" : 1e-4, # Supervised separator learning rate
                    'keepFreqs' : 1024,
                    "log_dir" : "logs", # Base folder for logs files
                    "model_base_dir" : "/home/dwoodward/masters/audio-perception-loss/models", # Base folder for model checkpoints
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
                    'loss_function' : 'weighted_l1'#presently implemented costs are 'weighted_l1' and 'l1', if 'loss_function' is not included in config, it will default to l1
                    }
    experiment_id = np.random.randint(0,1000000)

    # Set output sources
    if model_config["task"] == "metric":
        model_config["source_names"] = ['Ratingscore']
    else:
        raise NotImplementedError
    model_config["num_sources"] = len(model_config["source_names"])
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2

@config_ingredient.named_config
def baseline():
    print("Training baseline model")

@config_ingredient.named_config
def AAUmachine():
    print("Starting")
    model_config = {'augmentation' : False, # Random attenuation of source signals to improve generalisation perform$                    
                    'batch_size' : 8, # Batch size
                    'datasets' : 'all', # use all datasets if more than one available for a given task
                    'data_path' : "/home/dwoodw19/thesis/SASSEC/SASSEC_anonymized.csv", # Set this to where the prep$                    
                    'audio_path' : "/home/dwoodw19/thesis/SASSEC/Signals",
                    'epochs' :  1000, #number of epochs to train
                    "estimates_path" : "/home/dwoodw19/thesis/BSS_metric/estimates", # SET THIS PATH TO WHERE YOU WA$                    
                    'expected_sr': 44100,  # Downsample all audio input to this sampling rate
                    'fft_size' : 4096,
                    'hop' : 1024,
                    "inhouse_path_train" : "/home/daniel/audioSource/datasets/2stem/train/",
                    "inhouse_path_valid" : "/home/daniel/audioSource/datasets/2stem/valid/",
                    "init_sup_sep_lr" : 1e-4, # Supervised separator learning rate
                    'keepFreqs' : 1024,
                    "log_dir" : "logs", # Base folder for logs files
                    "model_base_dir" : "/home/dwoodw19/thesis/BSS_metric/estimates", # Base folder for model checkpo$                    
                    'network' : 'Convnet', # Type of network architecture, either unet (our model) or unet_spectrogr$                    
                    "num_frames": 1024 * 319 + 4096, # DESIRED number of time frames in the output waveform per samp$  
                    "audio_len" : 500000,                  
                    "num_snippets_per_track" : 10,# If train_rand_mode is 'per_min', then we are grabbing 10 snippet$                    
                    'num_workers' : 7, # Number of processes used for each TF map operation used when loading the da$                    
                    'num_projections' : 5,
                    'mono_downmix' : False,
                    'return_spectrogram' : True,
                    'restore_checkpoint' : False,
                    'epoch_it' : 2500,
                    'task' : 'metric', # Type of separation task. 'vocals' : Separate music into voice and accompani$                    
                    'loss_function' : 'weighted_l1'#presently implemented costs are 'weighted_l1' and 'l1', if 'loss$                    }
                    }

    experiment_id = np.random.randint(0,1000000)

    # Set output sources
    if model_config["task"] == "metric":
        model_config["source_names"] = ['Ratingscore']
    else:
        raise NotImplementedError
    model_config["num_sources"] = len(model_config["source_names"])
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2


