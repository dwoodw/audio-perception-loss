'''
Parser for SASSEC dataset as found at https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
Written by Daniel Woodward
'''

import numpy as np
import os
import librosa
import random

def parseCSV(filename):
    '''
    parseCSV returns a dictionary of lists corresponding to CSV file 

    Testname - name of dataset E.g. SASSEC
    Listener - anonymised listener number E.g. listener 14
    Trials   - Test file shown to participant 
    Condition - Algorithm or original label tested
    Ratingscore - Mushra Ratings Scoree
    '''
    with open(filename) as f:
        out = dict()
        out['Testname'] = list()
        out['Listener'] = list()
        out['Trials'] = list()
        out['Condition'] = list()
        out['Ratingscore'] = list()
        for idx, line in enumerate(f):
            if idx == 0:
                pass
            elif idx > 0:
                Testname, Listener, Trials, Condition, Ratingscore = line.split(",")
                out['Testname'].append(Testname)
                out['Listener'].append(Listener)
                out['Trials'].append(Trials)
                out['Condition'].append(Condition)
                out['Ratingscore'].append(int(Ratingscore.replace('\n', '')))
        print('Finished Parsing CSV file')
        return out

def stereoCheck(audio_data):

    if audio_data.ndim == 1:
        audio_data = np.stack((audio_data, audio_data))
    elif audio_data.ndim != 2:
        print('error in audio dimensions')
    
    return audio_data

def create_dict(csv_data, audio_test, audio_ref, num_list):
    #print(len(audio_test), len(audio_ref))

    audio = dict()
    audio['audio_test'] = audio_test
    audio['audio_ref'] = audio_ref
    audio['Ratingscore'] = list()
    for idx in num_list:
        audio['Ratingscore'].append(csv_data['Ratingscore'][idx])

    return audio



def parseAudio(csv_data, audio_folder = 'SASSEC/SASSEC/Signals', batch = -1):
    '''
    parseAudio returns a dictionary of lists corresponding to CSV file 
    audio - stereo audio files 
    '''
    audio_ref = list()
    audio_test = list()
    
    if batch == -1:
        num_list = range(len(csv_data['Testname']))
    else:
        num_tests = len(csv_data['Testname'])
        num_list = random.sample(range(0, num_tests), batch)

    #print(batch, num_list)
    #for each audio name in SASSEC data
    for idx in num_list:
        audio_files = csv_data['Trials'][idx]
        algo_num = csv_data['Condition'][idx]
        audio_path = os.path.join(audio_folder, algo_num, audio_files + '.wav')
        test, _ = librosa.core.load(audio_path, sr=None, mono=False, offset=0.0, duration=None, dtype='float32')
        test = stereoCheck(test)
        audio_test.append(test)

        audio_path = os.path.join(audio_folder, 'orig', audio_files + '.wav')
        ref, _ = librosa.core.load(audio_path, sr=None, mono=False, offset=0.0, duration=None, dtype='float32')
        ref = stereoCheck(ref)
        audio_ref.append(ref)
        

    return create_dict(csv_data, audio_test, audio_ref, num_list)

def dataset(csv_name, audio_folder, batch = 1):
    '''
    Returns a Dictionary of lists batch size long each with:
    audio_test - stereo audio data for input
    audio_ref - the reference stereo audio data for input
    RatingScore - the ratings score from the listening test

    '''

    csv_data = parseCSV(csv_name)
    audio_data = parseAudio(csv_data, audio_folder, batch)
    return audio_data



def main():
    file_name = '/home/dwoodward/masters/data/SASSEC/SASSEC_anonymized.csv'
    audio_folder = '/home/dwoodward/masters/data/SASSEC/Signals'

    batch = 1
    batch_data = dataset(file_name, audio_folder, batch)
    print(batch_data, '\n', batch_data)

if __name__ == '__main__':
    main()
