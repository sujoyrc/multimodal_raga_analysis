from copy import copy
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import tensorflow as tf
from numpy.lib.utils import deprecate
from numpy.random import seed
from scipy.stats import mode
seed(42)   

from contextlib import redirect_stdout
import wandb
from wandb.keras import WandbCallback
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, Dense, Flatten, Input, BatchNormalization, Activation, Dropout, Concatenate
from keras.regularizers import l2
from keras.constraints import MaxNorm, MinMaxNorm
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback, ReduceLROnPlateau
import keras.backend as K
from keras import Model
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
from functools import reduce
import operator
import sys
import math
import copy
import pdb


class Data:
    def __init__(self, data_file):
        '''
        Parameters
            data_file (str): file path to data npz file
        '''
        self.data_file = data_file
        self.raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']

    def load(self):
        '''
        Loads the data into X, y, ids tuples from data file

        Returns
            ((X_train, y_train, train_ids), (X_test, y_test, test_ids))
        '''

        data = np.load(self.data_file, allow_pickle=True)
        # train data
        X_train = data['X_train']
        y_train = to_categorical(data['y_train'])
        train_ids = data['train_ids']
    
        # test data
        X_test = data['X_test']
        y_test = to_categorical(data['y_test'])
        test_ids = data['test_ids']
        
        channels = data['channels']

        return ((X_train, y_train, train_ids), (X_test, y_test, test_ids), channels)

    def log_and_load(self, params):
        '''
        Loads the data with load function and logs the data to wandb

        Parameters
            project_name (str): project_name on wandb
            run_name (str): run name on wandb

        Returns
            ((X_train, y_train, train_ids), (X_test, y_test, test_ids))
        '''
        with wandb.init(
                project=params['project_name'],
                name=params['data_artifact'],
                tags = params['tags'],
                job_type='dataset'
            ) as run:

            run.config.update({
                'data file': self.data_file,
                'data artifact name': params['data_artifact']
            })
            ((X_train, y_train, train_ids), (X_test, y_test, test_ids), channels) = self.load()

            data_art = wandb.Artifact(params['data_artifact'], 
            type="dataset", 
            description=self.data_file.rsplit('/', 2)[1], 
            metadata={
                "data_file_path": self.data_file,
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "channels": channels
                }, 
            )

            datasets = [[X_train, y_train, train_ids], [X_test, y_test, test_ids]]
            names = ['train', 'test']
            for name, data in zip(names, datasets):
                with data_art.new_file(name + '.npz', mode='wb') as file:
                    X = data[0]
                    np.savez(file, X_0=X, y=data[1], ids=data[2])

            if params['log_metadata']:

                print('storing train table')
                # store a train table
                train_table = wandb.Table(columns=[], data=[])
                train_table.add_column('unique_id', train_ids)
                train_table.add_column('orig_audio', [wandb.Audio('../Audio/OrigAudioSeqs/' + train_id + '.wav') for train_id in train_ids])
                train_table.add_column('ss_audio', [wandb.Audio('../Audio/SSAudioSeqs/' + train_id + '.wav') for train_id in train_ids])
                for ind, channel in enumerate(channels):
                    channel_images = []
                    for row in range(X_train.shape[0]):
                        plt.plot(np.arange(0, 12, 0.01), X_train[row, :, ind])
                        channel_images.append(wandb.Image(plt))
                        plt.close()
                    train_table.add_column(channel, channel_images)
                train_table.add_column('class_id', np.argmax(y_train, axis=1))
                train_table.add_computed_columns(lambda ndx, row:{
                    'class_label': self.raga_labels[int(row['class_id'])]
                })
                data_art['train_table'] = train_table

                print('storing test table')
                # store a test table
                test_table = wandb.Table(columns=[], data=[])
                test_table.add_column('unique_id', test_ids)
                test_table.add_column('orig_audio', [wandb.Audio('../Audio/OrigAudioSeqs/' + test_id + '.wav') for test_id in test_ids])
                test_table.add_column('ss_audio', [wandb.Audio('../Audio/SSAudioSeqs/' + test_id + '.wav') for test_id in test_ids])
                for ind, channel in enumerate(channels):
                    channel_images = []
                    for row in range(X_test.shape[0]):
                        plt.plot(np.arange(0, 12, 0.01), X_test[row, :, ind])
                        channel_images.append(wandb.Image(plt))
                        plt.close()
                    test_table.add_column(channel, channel_images)
                test_table.add_column('class_id', np.argmax(y_test, axis=1))
                test_table.add_computed_columns(lambda ndx, row:{
                    'class_label': self.raga_labels[int(row['class_id'])]
                })
                data_art['test_table'] = test_table

            run.log_artifact(data_art)

        return ((X_train, y_train, train_ids), (X_test, y_test, test_ids))

class ModelInstance:
    def __init__(self, params, no_classes):
        '''
        Parameters
            params (dict): Dictionary like object with model parameters
            seq_len (int): Length of input sequences
            no_classes (int): Number of output classes in the model
        '''
        self.params = params
        self.no_classes = no_classes
        self.model = None
        self.raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']
        self.data = None    # maintains a version of the training data last used

    def _create_conv_layer(self, block_params, prev_layer, override_stride=False):
        '''
        Creates a convolution + normalisation + activation + pooling block for layers with the index i
        
        Parameters
            block_params (dict): dictionary of layers and hyperparameters to add to the block
            prev_layer (keras.layers): layer after which this convolution layer is to be added 
            override_stride (bool): if true, will set the stride of everything as 1

        Returns
            prev_layer (keras.layers): the last layer in the convolution block created
        '''
        if f'conv' in list(block_params.keys()):
            # extract params for the conv layer

            # kernel_constraint
            if 'kernel_constraint' in list(block_params[f'conv'].keys()):
                if block_params[f'conv']['kernel_constraint']['type'] == 'MaxNorm':
                    kernel_constraint = MaxNorm(max_value=block_params[f'conv']['kernel_constraint']['max_value'])
                elif block_params[f'conv']['kernel_constraint']['type'] == 'MinMaxNorm':
                    # pdb.set_trace()
                    kernel_constraint = MinMaxNorm(min_value=block_params[f'conv']['kernel_constraint']['min_value'], max_value=block_params[f'conv']['kernel_constraint']['max_value'], axis=[0, 1, 2])
                else:
                    raise Exception('Invalid kernel_constraint type')
            else:
                kernel_constraint = None

            # bias_constraint
            if 'bias_constraint' in list(block_params[f'conv'].keys()):
                if block_params[f'conv']['bias_constraint']['type'] == 'MaxNorm':
                    bias_constraint = MaxNorm(max_value=block_params[f'conv']['bias_constraint']['max_value'])
                elif block_params[f'conv']['bias_constraint']['type'] == 'MinMaxNorm':
                    bias_constraint = MinMaxNorm(min_value=block_params[f'conv']['bias_constraint']['min_value'], max_value=block_params[f'conv']['bias_constraint']['max_value'])
                else:
                    raise Exception('Invalid bias_constraint type')
            else:
                bias_constraint = None

            if override_stride:
                block_params['conv']['strides'] = 1
            # create the conv layer
            conv_layer = Conv1D(
                filters=block_params[f'conv']['filter'], 
                kernel_size=block_params[f'conv']['kernel_size'], 
                activation=block_params[f'conv']['activation'], 
                strides=block_params[f'conv']['strides'], 
                kernel_regularizer=l2(l=block_params[f'conv']['kernel_regularizer']['l2']['l']) if 'kernel_regularizer' in list(block_params[f'conv'].keys()) else None, 
                bias_regularizer=l2(l=block_params[f'conv']['bias_regularizer']['l2']['l']) if 'bias_regularizer' in list(block_params[f'conv'].keys()) else None, 
                kernel_constraint=kernel_constraint, 
                bias_constraint=bias_constraint, padding=block_params[f'conv']['padding'] if 'padding' in list(block_params['conv']) else 'valid',
                use_bias=block_params[f'conv']['use_bias'] if 'use_bias' in list(block_params['conv'].keys()) else True)(prev_layer) 

            prev_layer = conv_layer     # set prev_layer for the succeeding layer
        
        # next add batch norm layer
        if f'bn' in list(block_params.keys()):
            bn_layer = BatchNormalization()(prev_layer)
            prev_layer = bn_layer

        # next add activation layer
        if f'act' in list(block_params.keys()):
            act_layer = Activation(activation=block_params[f'act']['activation'])(prev_layer)
            prev_layer = act_layer
        
        # next add pool layer
        if f'pool' in list(block_params.keys()):
            if override_stride:
                block_params['pool']['strides'] = 1
            if block_params[f'pool']['type'] == 'Max':
                pool_layer = MaxPooling1D(pool_size=block_params[f'pool']['pool_size'], strides=block_params[f'pool']['strides'], padding=block_params['pool']['padding'] if 'padding' in list(block_params['pool'].keys()) else 'valid')(prev_layer)
            elif block_params[f'pool']['type'] == 'Average':
                pool_layer = AveragePooling1D(pool_size=block_params[f'pool']['pool_size'], strides=block_params[f'pool']['strides'], padding=block_params['pool']['padding'] if 'padding' in list(block_params['pool'].keys()) else 'valid')(prev_layer)
            prev_layer = pool_layer

        # add dropout layer
        if f'drop' in list(block_params.keys()):
            drop = Dropout(rate = block_params[f'drop']['rate'])(prev_layer)
            prev_layer = drop

        return prev_layer

    def _create_dense_layer(self, block_params, prev_layer):
        '''
        Creates a dense + batch norm + activation + dropout layers block with layers and hyperparameters defined by block_params
        
        Parameters
            block_params (dict): dictionary of layers and hyperparameters to add to the block
            prev_layer (keras.layers): layer after which this convolution layer is to be added 

        Return
            prev_layer (keras.layers): last layer in the block created
        '''

        # add dense layer
        if f'dense' in list(block_params.keys()):
            # extract params for dense layer

            # kernel_constraint
            if 'kernel_constraint' in list(block_params[f'dense'].keys()):
                if block_params[f'dense']['kernel_constraint']['type'] == 'MaxNorm':
                    kernel_constraint = MaxNorm(max_value=block_params[f'dense']['kernel_constraint']['max_value'], axis=1)
                else:
                    raise Exception('Invalid kernel_constraint type')
            else:
                kernel_constraint = None

            # bias_constraint
            if 'bias_constraint' in list(block_params[f'dense'].keys()):
                if block_params[f'dense']['bias_constraint']['type'] == 'MaxNorm':
                    bias_constraint = MaxNorm(max_value=block_params[f'dense']['bias_constraint']['max_value'])
                else:
                    raise Exception('Invalid bias_constraint type')
            else:
                bias_constraint = None

            dense = Dense(block_params[f'dense']['units'], 
            activation=block_params[f'dense']['activation'], 
            kernel_regularizer=l2(l=block_params[f'dense']['kernel_regularizer']['l2']['l']) if 'kernel_regularizer' in list(block_params[f'dense'].keys()) else None, 
            bias_regularizer=l2(l=block_params[f'dense']['kernel_regularizer']['l2']['l']) if 'kernel_regularizer' in list(block_params[f'dense'].keys()) else None, 
            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(prev_layer)
            prev_layer = dense
            
            # add bn layer
            if f'bn' in list(block_params.keys()):
                bn = BatchNormalization()(prev_layer)
                prev_layer = bn

            # add act layer
            if f'act' in list(block_params.keys()):
                act_layer = Activation(activation=block_params[f'act']['activation'])(prev_layer)
                prev_layer = act_layer

            # add dropout layer
            if f'drop' in list(block_params.keys()):
                drop = Dropout(rate = block_params[f'drop']['rate'])(prev_layer)
                prev_layer = drop

        return prev_layer

    def _create_flat_layer(self, block_params, prev_layer):
        '''
        Creates a flat + drop block with layers and hyperparameters defined by block_params
        
        Parameters
            block_params (dict): dictionary with layer and hyperparameter definitions
            prev_layer (keras.layers): layer after which this convolution layer is to be added 

        Return
            dense (keras.layers): dense layer created
        '''
        if 'flatten' in list(block_params.keys()):
            # add flatten layer

            if isinstance(prev_layer, list):
            # if there are multiple previous layers, first concatenate inputs to this layer instead of adding a flatten
                prev_layer = Concatenate()([Flatten()(p) for p in prev_layer])
            else:
                # else just flatten the output from input layers
                prev_layer = Flatten()(prev_layer)

        if 'drop' in list(block_params.keys()):
            # add drop layer
            prev_layer = Dropout(rate=block_params['drop']['rate'])(prev_layer)

        return prev_layer

    def _create_concatenate_layer(self, block_params, prev_layer):
        '''
        Concatenates list of prev_layers for layer with block_params

        Parameters
            block_params (dict): hyperparameters to add to layer
            prev_layer (keras.layers): returns the last layer created in this block
        '''
        assert isinstance(prev_layer, list), "input to concat layer should be a list"
        prev_layer = Concatenate(axis = block_params['axis'] if 'axis' in list(block_params.keys()) else -1)(prev_layer)
        if f'bn' in list(block_params.keys()):
            bn = BatchNormalization()(prev_layer)
            prev_layer = bn

        if 'act' in list(block_params.keys()):
            prev_layer = Activation(activation=block_params['act']['activation'])(prev_layer)

        if 'drop' in list(block_params.keys()):
            # add drop layer
            prev_layer = Dropout(rate=block_params['drop']['rate'])(prev_layer)
        return prev_layer

    def _create_inception_module(self, block_name, block_params, prev_layer):
        '''
        Creates an inception module with block_params

        Parameters
            block_name (str): name of inception module
            block_params (dict): dictionary of details of the inception module
            prev_layer (keras.layers): layer to connect the inception module to

        Results
            prev_layer (keras.layers): last layer of the inception block
        '''
        inception_params = copy.deepcopy(self.params['inception_module'][block_params['inception']])
        if 'override_stride' in list(block_params.keys()):
            override_stride = block_params['override_stride']
        else:
            override_stride = False
        layers =  list(inception_params.keys())
        layer_objs = {
            'input': prev_layer
        }     # dictionary with layer objects
        for layer_ind, layer in enumerate(layers):
            if layer == 'resnet' or layer == 'common':
                # make this better please :)
                continue
            # if params are redifined in block_params then replace them
            if 'overwritten_params' in list(block_params.keys()):
                if layer in list(block_params['overwritten_params'].keys()):
                    # some params for this layer may be redifined in block_params
                    for block in list(block_params['overwritten_params'][layer].keys()):
                        for param in list(block_params['overwritten_params'][layer][block].keys()):
                            inception_params[layer][block][param] = block_params['overwritten_params'][layer][block][param]  # overwrite the params

            if isinstance(inception_params[layer]['input'], list):
                prev_layer = [layer_objs[input_layer] for input_layer in inception_params[layer]['input']]
            else:
                prev_layer = layer_objs[inception_params[layer]['input']]
            
            # if kernel size of this block is specified by the "common kernel size" of this inception module
            if 'conv' in list(inception_params[layer].keys()) and inception_params[layer]['conv']['kernel_size'] == 'common':
                inception_params[layer]['conv']['kernel_size'] = inception_params['common']['conv']['kernel_size']
            layer_objs[layer] = self._create_block(block_name + '_' + layer, inception_params[layer], prev_layer, override_stride)

        if "resnet" in list(inception_params.keys()):
            # if resnet is in the keys, add a 1x1 conv layer to change the size of output and sum it with the input
            layer_objs['conv1x1-input'] = self._create_conv_layer({
                'conv': {
                    'filter': layer_objs[layers[-1]].shape.as_list()[-1],
                    "kernel_size": 1,
                        "activation": "linear",
                        "strides": 1,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                    "use_bias": False
                },
                'bn': {}
                
            }, layer_objs['input'])
            layer_objs['add-resnet'] = Activation('relu')(layer_objs['conv1x1-input'] + layer_objs[layers[-1]])
            layers.extend(['conv1x1-input', 'add-resnet'])

        return layer_objs[layers[-1]]

    def _create_block(self, block_name, block_params, prev_layer, override_stride=False):
        '''
        Create the block

        Parameters
            block_name (str): name of the block
            block_params (dict): dictionary of layers and hyperparameters for a given block
            prev_layer (keras.layers): the layer to connect this block to
            override_stride (bool): if True, all strides will be set to 1

        Returns
            prev_layer (keras.layers): the last layer present in this block
        '''
        if 'conv' in list(block_params.keys()) or 'pool' in list(block_params.keys()):
            prev_layer = self._create_conv_layer(block_params, prev_layer, override_stride=override_stride)
        elif 'flatten' in list(block_params.keys()):
            prev_layer = self._create_flat_layer(block_params, prev_layer)
        elif 'dense' in list(block_params.keys()):
            prev_layer = self._create_dense_layer(block_params, prev_layer)
        elif 'concat' in list(block_params.keys()):
           prev_layer = self._create_concatenate_layer(block_params, prev_layer)
        else:
            raise Exception(f'{block_name} not recognised')
        
        return prev_layer

    def build_model(self, input_shape, run):
        '''
        (used only for newer params.json files (Expt8 and above))
        Creates model according to self.params['layers'] dictionary.
        One activation layer, batch normalization and pooling layer are assumed to be attached to a conv layer in terms of numbering index. However for a given index it is possible that a conv layer isn't present but other types of layers are present.

        Parameters
            input_shape (tuple): shape of input data
            run (wandb.Run): The wandb Run object
        '''
       
        layers = [l for l in list(self.params['layers'].keys()) if l != 'output']
        inputs = {art_name: Input(shape=tuple(in_shape[1:])) for art_name, in_shape in input_shape.items()}    # input layer
        layer_objs = copy.copy(inputs)
        # load models
        if "model_arts" in list(self.params.keys()):
            for model_art in list(self.params['model_arts'].keys()):
                artifact = run.use_artifact('snnithya/Gesture Analysis/' + self.params['model_arts'][model_art]['model_name'], 'model')
                artifact_dir = artifact.download()
                model = load_model(os.path.join(artifact_dir, os.listdir(artifact_dir)[0]))
                model.trainable = self.params['model_arts'][model_art]['trainable']
                layer_objs[model_art] = Model(model.input, model.get_layer(self.params['model_arts'][model_art]['layer_name']).output)(layer_objs[self.params['model_arts'][model_art]['input']])

        # add layers after merge
        for layer_ind, layer in enumerate(layers):
            if self.params['layers'][layer]['input'] == 'input':
                self.params['layers'][layer]['input'] = list(inputs.keys())[0]
            if '-' in layer:
                layer_name = layer.split('-', 1)[1]
            else:
                layer_name = layer
            block_params = self.params['layers'][layer]
            if isinstance(block_params['input'], list):
                prev_layer = [layer_objs[input_layer] for input_layer in block_params['input']]
            else:
                prev_layer = layer_objs[block_params['input']]
            if 'inception' in list(block_params.keys()):
                layer_objs[layer_name] = self._create_inception_module(layer, block_params, prev_layer)
            else:
                layer_objs[layer_name] = self._create_block(layer, block_params, prev_layer)
        # create output layer
        prev_layer = layer_objs[self.params['layers']['output']['input']]  # set the previous layer based on if merge was used
        outputs = Dense(self.no_classes, activation='softmax', kernel_regularizer=l2(l=self.params['layers']['output']['kernel_regularizer']['l2']['l']) if 'kernel_regularizer' in list(self.params['layers']['output'].keys()) else None, bias_regularizer=l2(l=self.params['layers']['output']['kernel_regularizer']['l2']['l']) if 'kernel_regularizer' in list(self.params['layers']['output'].keys()) else None)(prev_layer)

        # return model
        model = Model(inputs=list(inputs.values()), outputs=outputs)
        self.model = model
        

    def log_and_build_model(self):
        '''
        Logs the model being built on wandb
        '''
        with wandb.init(
                project=self.params['project_name'],
                name=self.params['run_name'],
                tags = self.params['tags'],
                config = self.params,
                job_type='build model',
                settings=wandb.Settings(start_method="fork")
            ) as run:

            # build model
            self.build_model()

            # add initialised model as an artifact
            model_artifact = wandb.Artifact(self.params['run_name'], type="model", description="Initialized model", metadata=self.params)

            model_name_split = self.params['model_filename'].rsplit('.', 1)
            self.model.save(model_name_split[0] + '-initialized.' + model_name_split[1])

            model_artifact.add_file(model_name_split[0] + '-initialized.' + model_name_split[1])
            run.log_artifact(model_artifact)

    def _extract_data(self):
        '''
        Extracts the X, y and ids for train and test sets from the data extracted from the npz file
        '''
        # pdb.set_trace()
        X_train = []
        y_train = self.data[list(self.data.keys())[0]][0][1]
        X_test = []
        y_test = self.data[list(self.data.keys())[0]][1][1]
        train_ids = self.data[list(self.data.keys())[0]][0][2]
        test_ids = self.data[list(self.data.keys())[0]][1][2]
        for data_art in list(self.data.keys()):
            temp_train = []
            temp_test = []
            for id in train_ids:
                train_id = np.where(self.data[data_art][0][2] == id)[0][0]
                temp_train.append(self.data[data_art][0][0][train_id])     
            X_train.append(np.array(temp_train))
            for id in test_ids:
                test_id = np.where(self.data[data_art][1][2] == id)[0][0]
                temp_test.append(self.data[data_art][1][0][test_id])     
            X_test.append(np.array(temp_test))

        return X_train, y_train, X_test, y_test, train_ids, test_ids

    def train_model(self, data=None, run=None, log_evaluation=True):
        '''
        Train model

        Parameters:
            data (str): run name of data artifact stored
            run (wandb.run): if not None, it will contain the wandb run variable
            log_evaluation (bool): if True, will log the train predictions during training
        '''
        
        X_train, y_train, X_test, y_test, train_ids, test_ids = self._extract_data()
        
        # model structure
        print(self.model.summary())
        if not os.path.isfile(os.path.join(self.params['model_filename'].rsplit('/', 1)[0], 'structure.txt')):
            # store the model structure in a txt file if not already stored
            with open(os.path.join(self.params['model_filename'].rsplit('/', 1)[0], 'structure.txt'), 'w') as f:
                with redirect_stdout(f):
                    print(self.model.summary())

        # check if a learning rate schedule is used
        if 'learning_schedule' in list(self.params['optimizer'].keys()):
            if self.params['optimizer']['learning_schedule'] == 'ExponentialDecay':
                self.params['optimizer']['learning_schedule'] = ExponentialDecay(self.params['optimizer']['learning_rate'], self.params['optimizer']['decay_epochs']*(X_train[0].shape[0]//self.params['batch_size']), self.params['optimizer']['decay_rate'], staircase=True)
                # replace learning rate with learning schedule object
                self.params['optimizer']['learning_rate'] = self.params['optimizer']['learning_schedule']

        # set model optimizer
        if self.params['optimizer']['type'] == 'SGD':
            opt = SGD(learning_rate=self.params['optimizer']['learning_rate'] if 'learning_rate' in list(self.params['optimizer'].keys()) else None, momentum=self.params['optimizer']['momentum'] if 'momentum' in list(self.params['optimizer'].keys()) else None, decay=self.params['optimizer']['decay'] if 'decay' in list(self.params['optimizer'].keys()) else None)
        elif self.params['optimizer']['type'] == 'Adam':
            if 'clipnorm' in list(self.params['optimizer'].keys()):
                kwargs = {'clipnorm': self.params['optimizer']['clipnorm']}
            else:
                kwargs = {}
            opt = Adam(learning_rate=self.params['optimizer']['learning_rate'] if 'learning_rate' in list(self.params['optimizer'].keys()) else 0.001, 
            beta_1=self.params['optimizer']['beta_1'] if 'beta_1' in list(self.params['optimizer'].keys()) else 0.9,
            beta_2=self.params['optimizer']['beta_2'] if 'beta_1' in list(self.params['optimizer'].keys()) else 0.999,
            **kwargs
            )
        elif self.params['optimizer']['type'] == 'Adagrad':
            opt = Adagrad(learning_rate=self.params['optimizer']['learning_rate'] if self.params['optimizer']['learning_rate'] is not None else 0.01)
        elif self.params['optimizer']['type'] == 'Adadelta':
            opt = Adadelta(learning_rate=self.params['optimizer']['learning_rate'] if 'learning_rate' in list(self.params['optimizer'].keys()) else None)
        elif self.params['optimizer']['type'] == 'RMSprop':
            opt = RMSprop(learning_rate=self.params['optimizer']['learning_rate'] if 'learning_rate' in list(self.params['optimizer'].keys()) else None)
        else:
            raise Exception('Optimizer is not valid')
            
        # configure callbacks
        callback_values = []    # list of callbacks to use during model training

        # model checkpoint

        model_checkpoint_callback = ModelCheckpoint(
        filepath=self.params['model_filename'],
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

        callback_values.append(model_checkpoint_callback)

        if self.params['early_stopping']:
            es = EarlyStopping(monitor='val_accuracy', min_delta=0, mode='max', patience=100, verbose=1)

            callback_values.append(es)
        
        # log lr
        if 'learning_schedule' in list(self.params['optimizer'].keys()):
            log_lr = LambdaCallback(
                on_epoch_end= lambda epoch, metrics : run.log({'learning_rate': K.eval(self.model.optimizer.lr(self.model.optimizer.iterations) if isinstance(self.model.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule) else self.model.optimizer.lr)}, step=epoch))

            callback_values.append(log_lr)

        # add ReduceLROnPlateau if needed
        if 'learning_schedule' in list(self.params['optimizer'].keys()) and self.params['optimizer']['learning_schedule'] == 'ReduceLROnPlateau':
            reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
            factor=self.params['optimizer']['factor'] if 'factor' in list(self.params['optimizer'].keys()) else 0.5, 
            min_lr=self.params['optimizer']['min_lr'] if 'min_lr' in list(self.params['optimizer'].keys()) else 0.000001, patience=25, cooldown=10)
            callback_values.append(reduce_lr)


        # log loss/accuracy metrics during training with wandb
        if log_evaluation:
            callback_values.append(WandbCallback(monitor="val_accuracy",log_evaluation=True, log_weights=True, log_gradients=True, training_data=(X_train, y_train), validation_data=(X_test, y_test), log_evaluation_frequency=50))
        else:
            callback_values.append(WandbCallback(monitor="val_accuracy", log_evaluation=False))
        # compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # pdb.set_trace()# fit model
        history = self.model.fit(X_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], callbacks=callback_values, validation_data=(X_test, y_test), shuffle=True)  

        return run   

    def log_train_model(self, sweep=True, data=None, log_train_metadata=False, log_evaluation=False):
        '''
        Logs training of model on wandb

        Parameters:
        data (tuples): ((X_train, y_train, train_ids), (X_test, y_test, test_ids))
        sweep (bool): if true, this function is called during a sweep
        log_train_metadata (bool): if true, will log train predictions across training history
        log_evaluation (bool): if true logs the predictions on a trained model
        '''

        with wandb.init(
                project=self.params['project_name'],
                tags = self.params['tags'],
                job_type='training',
                settings=wandb.Settings(start_method="fork")
            ) as run:
            if isinstance(self.params['run_name'], int):
                # use the randomly assigned name and prepend the number to the run name
                run.save()
                self.params['run_name'] = str(self.params['run_name']) + '-' + run.name
            run.name = self.params['run_name']    # update run name
            if sweep:
                for key in list(run.config.keys()):
                    nested_keys = key.split('.')
                    reduce(operator.getitem, nested_keys[:-1], self.params)[nested_keys[-1]] = run.config[key]
            # else:
            #     run.config.update(self.params)

            # load data from the artifact
            if data is not None:
                self.data = data
            else:
                self.data = {}
                if isinstance(self.params['data_artifact'], str):
                    self.params['data_artifact'] = [self.params['data_artifact']]
                for data_art in self.params['data_artifact']:
                    dataset = run.use_artifact(data_art + ':latest')
                    data_dir = dataset.download()
                    self.channels = dataset.metadata['channels']
                    train_data = np.load(os.path.join(data_dir, 'train' + '.npz'), allow_pickle=True)
                    test_data = np.load(os.path.join(data_dir, 'test.npz'), allow_pickle=True)
                    
                    X_train, y_train, train_ids = train_data['X_0'], train_data['y'], train_data['ids']
                    X_test, y_test, test_ids = test_data['X_0'], test_data['y'], test_data['ids']
                    
                    # add the data as a class variable
                    self.data[data_art] = ((X_train, y_train, train_ids), (X_test, y_test, test_ids))
            
            # store shape of inputs
            input_shape = {d_key: d[0][0].shape for d_key, d in self.data.items()}

            # build model
            self.build_model(input_shape, run=run)

            model_params = wandb.Artifact("model-architecture", type="json-file")
            with model_params.new_file('params-' + datetime.now().strftime('%M_%S') + '.json', 'w') as file:
                json_content = json.dumps(self.params)
                file.write(json_content)
            run.log_artifact(model_params)

            run = self.train_model(run=run, log_evaluation=log_train_metadata)

            model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description=f"model architecture defined at {self.params['model_filename']}",
            metadata=self.params)
            model_artifact.add_file(self.params['model_filename'])
            run.log_artifact(model_artifact)

            train_loss, train_acc, test_loss, test_acc, train_preds, test_preds, train_ids, test_ids = self.evaluate(log_metadata=log_evaluation)
            # add test loss and acc
            run.summary.update({'train loss': train_loss, 'train accuracy': train_acc, 'test loss': test_loss, 'test accuracy': test_acc})
            # add the cm artifact
            
            cm_artifact = wandb.Artifact(
                "confusion-matrices", 
                type="Evaluation",
                description=f"Confusion matrix from model with lowest validation loss during training from {self.params['model_filename']}")
            cm_artifact.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-cm.png')
            run.log_artifact(cm_artifact)

            # log metadata
            if log_evaluation:
                # log evaluation predictions tabls
                train_table = wandb.Table(columns=[], data=[])
                train_table.add_column('unique_id', train_ids)
                train_table.add_column('true_class', np.argmax(self.data[list(self.data.keys())[0]][0][1], axis=1))
                train_table.add_column('predicted_class', np.argmax(train_preds, axis=1))
                for label in range(train_preds.shape[1]):
                    train_table.add_column('prediction_probability_' + str(label), train_preds[:, label])

                test_table = wandb.Table(columns=[], data=[])
                test_table.add_column('unique_id', test_ids)
                test_table.add_column('true_class', np.argmax(self.data[list(self.data.keys())[0]][1][1], axis=1))
                test_table.add_column('predicted_class', np.argmax(test_preds, axis=1))
                for label in range(test_preds.shape[1]):
                    test_table.add_column('prediction_probability_' + str(label), test_preds[:, label])

                preds = wandb.Artifact("Predictions", "evaluation")
                preds['train_table'] = train_table
                preds['test_table'] = test_table
                run.log_artifact(preds)

        if sweep:
            # update model_filename in params, useful in case of sweeps
            model_no = int(self.params['model_filename'].rsplit('_', 1)[1].rsplit('.', 1)[0])
            self.params['model_filename'] = self.params['model_filename'].rsplit('_', 1)[0] + str(model_no+1) + '.hdf5'
            print(f'Updated model_filename param to {self.params["model_filename"]}')
            
    def plot_metrics(self, history):
        '''Plots the train metrics in model folder'''
        metrics = defaultdict(int)
        min_val_ind = np.argmin(history.history['val_loss'])
        metrics['min_val_ind'] = min_val_ind
        metrics['train_loss'] = history.history['loss'][min_val_ind]
        metrics['train_accuracy'] = history.history['accuracy'][min_val_ind]
        metrics['val_loss'] = history.history['val_loss'][min_val_ind]
        metrics['val_accuracy'] = history.history['val_accuracy'][min_val_ind]

        return metrics

    def evaluate(self, data=None, log_metadata=True):
        '''
        Evaluates model
        
        Parameters
            data (tuple): data of form ((X_train, y_train), (X_test, y_test)) if you don't want to use self.data for evaluation
            log_metadata (bool): logs each prediction along with probabilities
        '''

        model = load_model(self.params['model_filename'])
        
        X_train, y_train, X_test, y_test, train_ids, test_ids = self._extract_data()

        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=1, batch_size=self.params['batch_size'])
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1, batch_size=self.params['batch_size'])
        
        # save cms to local systems
        # plot confusion matrices
        figcm, axcm = plt.subplots(1, 2, figsize=(20, 10))

        # train cm
        sns.heatmap(confusion_matrix(np.argmax(y_train, axis=1), np.argmax(model.predict(X_train), axis=1)), xticklabels=self.raga_labels, yticklabels=self.raga_labels, annot=True, fmt="d", ax=axcm[0])
        axcm[0].set(xlabel='Predicted Label', ylabel='True Label', title='Train CM')

        # test cm
        sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1)), xticklabels=self.raga_labels, yticklabels=self.raga_labels, annot=True, fmt="d", ax=axcm[1])
        axcm[1].set(xlabel='Predicted Label', ylabel='True Label', title='Test CM')

        # save fig
        figcm.savefig(self.params['model_filename'].rsplit('.', 1)[0] + '-cm.png')

        if log_metadata:
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
        else:
            train_preds = None
            test_preds = None

        return train_loss, train_acc, test_loss, test_acc, train_preds, test_preds, train_ids, test_ids
    
    def map_id_to_files(self, ids):
        '''
        Maps a list of ids to a file (file name determined by removing the last underscore)
        '''

        mapping = defaultdict(list)
        for id in ids:
            mapping[id.rsplit('_', 1)[0]].append(id)
        return mapping

    def evaluate_class2(self, X, y, ids, model, split_inputs=False):
        
        cm = np.zeros((y[0].shape[0],y[0].shape[0])).astype('int') # set a cm as zero array of shape (# classes, # classes)
        loss = 0
        file_classification = []    # maintains classification accuracy for each file
        mapping = self.map_id_to_files(ids)
        for fileName, id_values in mapping.items():
            id_values = np.array(id_values)
            X_temp = X[(id_values[:, None] == ids).sum(axis=0).nonzero()[0], :, :] if not split_inputs else [X_channel[(id_values[:, None] == ids).sum(axis=0).nonzero()[0], :, :] for X_channel in X]
            y_temp = y[(id_values[:, None] == ids).sum(axis=0).nonzero()[0]]
            true_y = y[(id_values[:, None] == ids).sum(axis=0).nonzero()[0]][0].argmax() # check that this is the same value as only value in y_temp

            preds = model.predict(X_temp)
            for _, pred in enumerate(preds):
                # check shape of pred
                loss -= math.log(pred[true_y] + sys.float_info.epsilon)
            
            loss = loss/len(preds)  #take the mean of all subsequences from a file
            
            cm[true_y][mode(np.argmax(preds, axis=1))[0][0]] += 1 # takes the lower value if 2 labels are predicted the same number of times
            
            file_classification.append([fileName, ((np.argmax(preds, axis=1) == true_y).nonzero())[0].shape[0], preds.shape[0], np.around(((np.argmax(preds, axis=1) == true_y).nonzero())[0].shape[0]/preds.shape[0], 2), loss, true_y])
            file_classification[-1].extend(np.mean(preds, axis=0))
        
        acc = np.sum(np.diag(cm))/np.sum(cm)
        col_names = ['File Name', '# Correct Predictions', '# Files', 'Classification Accuracy',  'Mean Loss', 'True Label']
        col_names.extend([f'Prob_{i}' for i in range(len(preds[0]))])    # add mean probabilities of each class
        file_classification = pd.DataFrame(file_classification, columns=col_names)

        return loss, acc, cm, file_classification
        
    def evaluate_class(self, data=None, split_inputs=None):
        '''
        Evaluates model by taking predictions by taking a majority vote over all subsequences from a given song.
        
        Parameters
            data (tuple): data of form ((X_train, y_train, train_ids), (X_test, y_test, test_ids)) if you don't want to use self.data for evaluation
            log (bool): if True, this function was called from log_and_evaluate in which case the cm will directly be logged into wandb
            split_inputs (bool): if true, each channel is processed separately
            summaryFile (str): csv file that matches each id to a singer/raga
        '''
        if split_inputs is None:
            # if split inputs arg is not provided, check the value in the params
            split_inputs = self.params['split_channel']
        # load model and data
        model = load_model(self.params['model_filename'])
        if data is not None:
            ((X_train, y_train, train_ids, _), (X_test, y_test, test_ids, _)) = data
        else:
            ((X_train, y_train, train_ids, _), (X_test, y_test, test_ids, _)) = self.data
        
        if split_inputs:
            X_train = [X_train[:, :, i] for i in range(X_train.shape[2])]
            X_test = [X_test[:, :, i] for i in range(X_test.shape[2])]

        train_loss, train_acc, train_cm, train_files = self.evaluate_class2(X_train, y_train, train_ids, model, split_inputs)

        test_loss, test_acc, test_cm, test_files = self.evaluate_class2(X_test, y_test, test_ids, model, split_inputs)
        
        # store cms locally
        figcm, axcm = plt.subplots(1, 2, figsize=(20, 10))
        # train cm
        sns.heatmap(train_cm, xticklabels=self.raga_labels, yticklabels=self.raga_labels, annot=True, fmt="d", ax=axcm[0])
        axcm[0].set(xlabel='Predicted Label', ylabel='True Label', title='Train CM')
        # test cm
        sns.heatmap(test_cm, xticklabels=self.raga_labels, yticklabels=self.raga_labels, annot=True, fmt="d", ax=axcm[1])
        axcm[1].set(xlabel='Predicted Label', ylabel='True Label', title='Test CM')
        # save fig
        figcm.savefig(self.params['model_filename'].rsplit('.', 1)[0] + '-class-cm.png')

        # store csv files locally
        train_files.to_csv(self.params['model_filename'].rsplit('.', 1)[0] + '-train_files.csv', index=False)
        test_files.to_csv(self.params['model_filename'].rsplit('.', 1)[0] + '-test_files.csv', index=False)

        return train_loss, train_acc, test_loss, test_acc

    def log_and_evaluate(self, data=None, log_metadata=False): 
        '''
        Evaluates the model and logs the report in wandb

        Parameters
            data (tuple): the train and test data to evaluate the model on
        '''
        with wandb.init(
                project=self.params['project_name'],
                tags = self.params['tags'],
                job_type='evaluation',
                settings=wandb.Settings(start_method="fork")
            ) as run:
            if isinstance(self.params['run_name'], int):
                # use the randomly assigned name and prepend the number to the run name
                run.save()
                self.params['run_name'] = str(self.params['run_name']) + '-' + run.name
            run.name = self.params['run_name']    # update run name
            split_inputs = self.params['split_channel']
    
            if data is None:
                self.data = {}
                if isinstance(self.params['data_artifact'], str):
                    self.params['data_artifact'] = [self.params['data_artifact']]
                for data_art in self.params['data_artifact']:
                    dataset = run.use_artifact(data_art + ':latest')
                    data_dir = dataset.download()
                    self.channels = dataset.metadata['channels']
                    train_data = np.load(os.path.join(data_dir, 'train' + '.npz'), allow_pickle=True)
                    test_data = np.load(os.path.join(data_dir, 'test.npz'), allow_pickle=True)
                    
                    # implies the X data is already stored in the artifact
                    X_train, y_train, train_ids = train_data['X_0'], train_data['y'], train_data['ids']
                    test_data = np.load(os.path.join(data_dir, 'test' + '.npz'), allow_pickle=True)
                    X_test, y_test, test_ids = test_data['X_0'], test_data['y'], test_data['ids']
                    
                    # add the data as a class variable
                    self.data[data_art] = ((X_train, y_train, train_ids), (X_test, y_test, test_ids))
            
            # load model
            artifact = run.use_artifact(self.params['model_filename'], type='model')
            artifact_dir = artifact.download()
            self.params['model_filename'] = os.path.join(artifact_dir, [x for x in os.listdir(artifact_dir) if x.endswith('.hdf5')][0])
            train_loss, train_acc, test_loss, test_acc, train_preds, test_preds, train_ids, test_ids = self.evaluate()

            # add train loss and acc
            run.summary.update({'train loss': train_loss, 'train accuracy': train_acc})
            # add test loss and acc
            run.summary.update({'test loss': test_loss, 'test accuracy': test_acc})

            cm_artifact = wandb.Artifact(
                "confustion-matrices", 
                type="Evaluation",
                description=f"Confusion matrix from model with lowest validation loss during training from {self.params['model_filename']}")
            cm_artifact.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-cm.png')
            run.log_artifact(cm_artifact)

            if log_metadata:
                # log evaluation predictions tabls
                train_table = wandb.Table(columns=[], data=[])
                train_table.add_column('unique_id', train_ids)
                train_table.add_column('true_class', np.argmax(self.data[list(self.data.keys())[0]][0][1], axis=1))
                train_table.add_column('predicted_class', np.argmax(train_preds, axis=1))
                for label in range(train_preds.shape[1]):
                    train_table.add_column('prediction_probability_' + str(label), train_preds[:, label])

                test_table = wandb.Table(columns=[], data=[])
                test_table.add_column('unique_id', test_ids)
                test_table.add_column('true_class', np.argmax(self.data[list(self.data.keys())[0]][0][1], axis=1))
                test_table.add_column('predicted_class', np.argmax(test_preds, axis=1))
                for label in range(test_preds.shape[1]):
                    test_table.add_column('prediction_probability_' + str(label), test_preds[:, label])

                preds = wandb.Artifact("Predictions", "evaluation")
                preds['train_table'] = train_table
                preds['test_table'] = test_table
                run.log_artifact(preds)

    def log_and_evaluate_class(self, data=None): 
        '''
        Evaluates the model and logs the report in wandb

        Parameters
            data (tuple): the train and test data to evaluate the model on
        '''
        with wandb.init(
                project=self.params['project_name'],
                tags = self.params['tags'],
                job_type='evaluation',
                settings=wandb.Settings(start_method="fork")
            ) as run:
            if isinstance(self.params['run_name'], int):
                # use the randomly assigned name and prepend the number to the run name
                run.save()
                self.params['run_name'] = str(self.params['run_name']) + '-' + run.name
            run.name = self.params['run_name']    # update run name
            if data is None:
                dataset = run.use_artifact(self.params['data_artifact'] + ':latest')
                data_dir = dataset.download()
                train_data = np.load(os.path.join(data_dir, 'train' + '.npz'), allow_pickle=True)
                X_train, y_train, train_ids, mask_train = train_data['X_0'], train_data['y'], train_data['ids'], train_data['mask']
                test_data = np.load(os.path.join(data_dir, 'test' + '.npz'), allow_pickle=True)
                X_test, y_test, test_ids, mask_test = test_data['X_0'], test_data['y'], test_data['ids'], test_data['mask']
                # add the data as a class variabl
                self.data = ((X_train, y_train, train_ids, mask_train), (X_test, y_test, test_ids, mask_test))
            
            # load model
            artifact = run.use_artifact(self.params['model_filename'], type='model')
            artifact_dir = artifact.download()
            self.params['model_filename'] = os.path.join(artifact_dir, [x for x in os.listdir(artifact_dir) if x.endswith('.hdf5')][0])
            train_loss, train_acc, test_loss, test_acc = self.evaluate_class()

            # add train loss and acc
            run.summary.update({'train class accuracy': train_acc})
            # add test loss and acc
            run.summary.update({'test class accuracy': test_acc})

            cm_artifact = wandb.Artifact(
                "confustion-matrices-class", 
                type="Evaluation",
                description=f"Confusion matrix from model with lowest validation loss during training with majority voting for each song from {self.params['model_filename']}")
            cm_artifact.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-class-cm.png')
            run.log_artifact(cm_artifact)

            files_artifact = wandb.Artifact(
                "file-level-classification", 
                type="Evaluation",
                description=f"File level classification with majority voting {self.params['model_filename']}")
            files_artifact.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-train_files.csv')
            files_artifact.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-test_files.csv')
            run.log_artifact(files_artifact)