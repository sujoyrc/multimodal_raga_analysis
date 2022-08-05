import numpy as np
import argparse
from music_raga_gendata_nithya import Feeder_kinetics
from keras.utils import to_categorical
import os
from tqdm import tqdm
import sys
import pdb
sys.path.append('../../CommonScripts/')
from common_utils import checkPath

inputDataFolder = '../Seqs/JSON-Video/'
outputDataFolder = '../Seqs/finalVideoInception/easy_1-channel-separate/'
splitFolder = '../Seqs/splits/easy_1/'

def gen_data(data_path, label_path,
            train, validation_labels,
            num_person_in=1,  # observe the first 5 persons
            num_person_out=1,  # then choose 2 persons with the highest score
            max_frame=300):
    '''
    The function generates the train/validation set

    Parameters
    ----------
    '''

    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame,
        train=train,
        validation_labels=validation_labels)

    sample_name = feeder.sample_name    # list of json files
    sample_label = []
    fp = np.zeros((len(sample_name), 3, max_frame, 11, num_person_out), dtype=np.float32)    # shape=(# of samples, 3, 300, 11, 1)
    ids = []
    for i, s in tqdm(enumerate(sample_name)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

        ids.append(s.replace('.json', ''))
    # X_vals = np.reshape(np.swapaxes(fp, 1, 2), (len(sample_name), max_frame, 33))   #[all x values, all y values, all confidence values]
    # pdb.set_trace()
    X_vals = np.reshape(np.swapaxes(fp, 1, 2), (len(sample_name), max_frame, 11, 3))   #[all x values, all y values, all confidence values] (, 300, 3, 11)
    y_vals = sample_label
    
    return (X_vals, y_vals, ids)

def main(arg):
    for split_file in os.listdir(arg.split_folder):
        print (split_file)
        splits = split_file.rsplit('-', 1)
        with open(os.path.join(arg.split_folder, split_file), 'r') as f:
            test_files = f.readlines()
            test_files = [test_file.rstrip('\n') for test_file in test_files]

            data_path = inputDataFolder + 'Data Raw/'
            label_path = inputDataFolder + 'Label Raw/music_solo_label.json'
            print('validation set')
            
            X_val, y_val, ids_val = gen_data(data_path, label_path, train=False, validation_labels=test_files)

            print('music_solo ', 'train')

            X_train, y_train, ids_train = gen_data(data_path, label_path, train=True, validation_labels=test_files)

        np.savez(os.path.join(checkPath(outputDataFolder), f"{split_file.rsplit('-', 1)[1].rsplit('.', 1)[0]}-norm.npz"), X_train=X_train, X_test=X_val, y_train=y_train, y_test=y_val, train_ids=ids_train, test_ids=ids_val, mask_train=None, mask_test=None, channels=['video'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='music-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default=inputDataFolder)
    parser.add_argument(
        '--out_folder', default=outputDataFolder)
    parser.add_argument(
        '--split_folder', default=splitFolder
    )
    arg = parser.parse_args()
    main(arg)