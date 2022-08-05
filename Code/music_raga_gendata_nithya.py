import os
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath
import pdb

num_joint = 11
max_frame = 300
num_person_out = 1
num_person_in = 1

inputDataFolder = '../Seqs/JSON-Video/'
outputDataFolder = '../Seqs/finalDataVideo/hard_2/'
splitFolder = '../Seqs/splits/hard_2/'


class Feeder_kinetics(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8, "REye"},
    # {9, "LEye"},
    # {10, "MidHip"}
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
        train: If true, creating train set data
        validation_labels: list of songs in the validation set
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=5,
                 num_person_out=2,
                 train=True,
                 validation_labels=[]):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample
        self.train = train
        self.validation_labels=validation_labels

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)   # list of json files

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)
        if self.train: 
            # removes filenames that are in the validation set
            sample_id = [name.replace('.json','') for name in self.sample_name if name.rsplit('_', 1)[0] not in self.validation_labels]
            self.sample_name = [x for x in self.sample_name if x.rsplit('_', 1)[0] not in self.validation_labels]
        else:
            # keeps filenames in the validation set
            sample_id = [name.replace('.json','') for name in self.sample_name if name.rsplit('_', 1)[0] in self.validation_labels]
            self.sample_name = [x for x in self.sample_name if x.rsplit('_', 1)[0] in self.validation_labels]
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score
                

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            train, validation_labels,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame):
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
    fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)    # shape=(# of samples, 3, 300, 11, 1)
    ids = []
    for i, s in enumerate(sample_name):
        print('sample number {}'.format(i))
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        print('sample label {}'.format(label))
        sample_label.append(label)

        ids.append(s.replace('.json', ''))
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)

    return ids
    

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

    for split_file in os.listdir(arg.split_folder):
        print (split_file)
        splits = split_file.rsplit('-', 1)
        with open(os.path.join(arg.split_folder, split_file), 'r') as f:
            test_files = f.readlines()
            test_files = [test_file.rstrip('\n') for test_file in test_files]

            data_path = inputDataFolder + 'Data Raw/'
            label_path = inputDataFolder + 'Label Raw/music_solo_label.json'
            print('music_solo ', 'val')
            if len(splits) > 0:
                data_out_path = '{}/{}/{}_data_joint.npy'.format(arg.out_folder, splits[1].rsplit('.', 1)[0], 'val')
                label_out_path = '{}/{}/{}_label.pkl'.format(arg.out_folder, splits[1].rsplit('.', 1)[0], 'val')
            else:
                data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, 'val')
                label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, 'val')
            val_ids = gendata(data_path, label_path, checkPath(data_out_path), checkPath(label_out_path), train=False, validation_labels=test_files)

            print('music_solo ', 'train')
            if len(splits) > 0:
                data_out_path = '{}/{}/{}_data_joint.npy'.format(arg.out_folder, splits[1].rsplit('.', 1)[0], 'train')
                label_out_path = '{}/{}/{}_label.pkl'.format(arg.out_folder, splits[1].rsplit('.', 1)[0], 'train')
            else:
                data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, 'train')
                label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, 'train')
            train_ids = gendata(data_path, label_path, checkPath(data_out_path), checkPath(label_out_path), train=True, validation_labels=test_files)

        if len(splits) > 0:
            idOutput = '{}/{}/ids.npz'.format(arg.out_folder, splits[1].rsplit('.', 1)[0])
        else:
            idOutput = '{}/ids.npz'.format(arg.out_folder)
        np.savez(idOutput, val=val_ids, train=train_ids)
