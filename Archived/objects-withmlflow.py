from keras import callbacks
from mlflow.tracking.fluent import log_artifact
from numpy.random import seed
seed(42)   
#from tensorflow.random import set_seed
#set_seed(42) # set seed for reproducible results

from keras.layers.core import Dropout
from contextlib import redirect_stdout
import mlflow
import wandb
from wandb.keras import WandbCallback
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, Dense, Flatten, Input, BatchNormalization, Activation
from keras.regularizers import l2, l1
from keras.constraints import MaxNorm
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras import Model
import keras.models
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import pdb

class Data:
    def __init__(self, train_file, test_file):
        '''
        Parameters
            train_file (str): file path to train data
            test_file (str): file path to test data
        '''
        self.train_path = train_file
        self.test_path = test_file

    def load(self):
        '''
        Loads the data into X, y tuples from both train and test files
        
        Returns
            ((X_train, y_train), (X_test, y_test))
        '''
        # train data
        train_df = pd.read_csv(self.train_path)
        X_train = train_df.iloc[:, :-1].values
        X_train = np.reshape(X_train, (-1, train_df.shape[1]-1, 1))
        y_train = to_categorical(train_df.iloc[:, -1].values)
        # test data
        test_df = pd.read_csv(self.test_path)
        X_test = test_df.iloc[:, :-1].values
        X_test = np.reshape(X_test, (-1, train_df.shape[1]-1, 1))
        y_test = to_categorical(test_df.iloc[:, -1].values)

        return ((X_train, y_train), (X_test, y_test))

    def log_and_load(self, project_name, run_name):
        '''
        Loads the data with load function and logs the data to wandb

        Parameters
            project_name (str): project_name on wandb
            run_name (str): run name on wandb
        '''
        with wandb.init(
                project=self.params['project_name'],
                name=self.params['run_name'],
                tags = self.params['tags'],
                config = self.params,
                job_type='dataset'
            ) as run:
            ((X_train, y_train), (X_test, y_test)) = self.load()

            data_art = wandb.Artifact(self.train_path.rsplit('/', 2)[1], 
            type="dataset", 
            description=self.train_path.rsplit('/', 2)[1], 
            metadata={
                "train_file_path": self.train_path, 
                "test_file_path": self.test_path,
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape
                }, 
            )

            datasets = [[X_train, y_train], [X_test, y_test]]
            names = [self.train_path.rsplit('.', 1)[0] + '.npz', self.test_path.rsplit('.', 1)[0] + '.npz']
            for name, data in zip(names, datasets):
                with data_art.new_file(name, mode='wb') as file:
                    np.savez(file, x=data[0], y=data[1])
            run.log_artifact(data_art)

        return ((X_train, y_train), (X_test, y_test))

class ModelInstance:
    def __init__(self, params, seq_len, no_classes):
        '''
        Parameters
            params (dict): Dictionary like object with model parameters
            seq_len (int): Length of input sequences
            no_classes (int): Number of output classes in the model
        '''
        self.params = params
        self.seq_len = seq_len
        self.no_classes = no_classes
        self.model = None
        self.raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']     

    def _create_conv_layer(self, i, prev_layer):
        '''
        Creates a convolution + normalisation + activation + pooling block for layers with the index i
        ### not ready to handle multi-head convolutions - only sequential models can be created right now
        
        Parameters
            i (str): index of the layers; if the convolution/norm/activation/pooling layer is not present for the ith index, it is ignored
            prev_layer (keras.layers): layer after which this convolution layer is to be added 
        '''
        layer = self.params['layers'][i]
        layer_inds = [key.rsplit('_', 1)[1] for key in list(layer.keys())]
        layer_inds.sort()   # used to find the number of parallel conv layers in this block; this is useful when there are conv blocks with multiple kernels being used
        min_ind = int(layer_inds[0])
        max_ind = int(layer_inds[-1])
        for ind in range(min_ind, max_ind + 1):     # has to be only 1 right now
            # first add conv layer
            if f'conv_{ind}' in list(layer.keys()):
                # extract params

                # kernel_constraint
                if 'kernel_constraint' in list(layer[f'conv_{ind}'].keys()):
                    if layer[f'conv_{ind}']['kernel_constraint']['type'] == 'MaxNorm':
                        kernel_constraint = MaxNorm(max_value=layer[f'conv_{ind}']['kernel_constraint']['max_value'])
                    else:
                        raise Exception('Invalid kernel_constraint type')
                else:
                    kernel_constraint = None
            
                # bias_constraint
                if 'bias_constraint' in list(layer[f'conv_{ind}'].keys()):
                    if layer[f'conv_{ind}']['bias_constraint']['type'] == 'MaxNorm':
                        bias_constraint = MaxNorm(max_value=layer[f'conv_{ind}']['bias_constraint']['max_value'])
                    else:
                        raise Exception('Invalid bias_constraint type')
                else:
                    bias_constraint = None

                # kernel regularizer
                if 'kernel_regularizer' in list(layer[f'conv_{ind}'].keys()):
                    if 'l1' in layer[f'conv_{ind}']['kernel_regularizer'].keys():
                        if 'l' in list(layer[f'conv_{ind}']['kernel_regularizer']['l1'].keys()):
                            kernel_regularizer = l1(layer[f'conv_{ind}']['kernel_regularizer']['l1']['l'])
                        else:
                            kernel_regularizer='l1'
                    elif 'l2' in layer[f'conv_{ind}']['kernel_regularizer'].keys():
                        if 'l' in list(layer[f'conv_{ind}']['kernel_regularizer']['l2'].keys()):
                            kernel_regularizer = l2(layer[f'conv_{ind}']['kernel_regularizer']['l2']['l'])
                        else:
                            kernel_regularizer = 'l2'
                    else:
                        raise Exception('Kernel regularizer cannot be instantiated')
                else:
                    kernel_regularizer = None

                # bias regularizer
                if 'bias_regularizer' in list(layer[f'conv_{ind}'].keys()):
                    if 'l1' in layer[f'conv_{ind}']['bias_regularizer'].keys():
                        if 'l' in list(layer[f'conv_{ind}']['bias_regularizer']['l1'].keys()):
                            bias_regularizer = l1(layer[f'conv_{ind}']['bias_regularizer']['l1']['l'])
                        else:
                            bias_regularizer = 'l1'
                    elif 'l2' in layer[f'conv_{ind}']['bias_regularizer'].keys():
                        if 'l' in list(layer[f'conv_{ind}']['bias_regularizer']['l2'].keys()):
                            bias_regularizer = l2(layer[f'conv_{ind}']['bias_regularizer']['l2']['l'])
                        else:
                            bias_regularizer = 'l2'
                    else:
                        raise Exception('Bias regularizer cannot be instantiated')
                else:
                    bias_regularizer = None

                # kernel initializer
                if 'kernel_initializer' in list(layer[f'conv_{ind}'].keys()):
                    kernel_initializer = layer[f'conv_{ind}']['kernel_initializer']
                else:
                    kernel_initializer = 'glorot_uniform'

                # bias initializer
                if 'bias_initializer' in list(layer[f'conv_{ind}'].keys()):
                    bias_initializer = layer[f'conv_{ind}']['bias_initializer']
                else:
                    bias_initializer = 'glorot_uniform'

                conv_layer = Conv1D(filters=layer[f'conv_{ind}']['filter'], kernel_size=layer[f'conv_{ind}']['kernel_size'], activation=layer[f'conv_{ind}']['activation'], strides=layer[f'conv_{ind}']['strides'], kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(prev_layer) 

                prev_layer = conv_layer     # set prev_layer for the succeeding layer
            
            # next add batch norm layer
            if f'bn_{ind}' in list(layer.keys()):
                bn_layer = BatchNormalization()(prev_layer)
                prev_layer = bn_layer

            # next add activation layer
            if f'act_{ind}' in list(layer.keys()):
                act_layer = Activation(activation=layer[f'act_{ind}']['activation'])(prev_layer)
                prev_layer = act_layer
            
            # next add pool layer
            if f'pool_{ind}' in list(layer.keys()):
                if layer[f'pool_{ind}']['type'] == 'Max':
                    pool_layer = MaxPooling1D(pool_size=layer[f'pool_{ind}']['pool_size'], strides=layer[f'pool_{ind}']['strides'])(prev_layer)
                elif layer[f'pool_{ind}']['type'] == 'Average':
                    pool_layer = AveragePooling1D(pool_size=layer[f'pool_{ind}']['pool_size'], strides=self.params['layers'][f'pool_{ind}']['strides'])(prev_layer)
                prev_layer = pool_layer

        return prev_layer

    def _create_dense_layer(self, i, prev_layer):
        '''
        Creates a dense + batch norm + activation + dropout layers with the index i
        
        Parameters
            i (str): index of the layer
            prev_layer (keras.layers): layer after which this convolution layer is to be added 

        Return
            dense (keras.layers): dense layer created
        '''
        layer = self.params['layers'][i]
        layer_inds = [key.rsplit('_', 1)[1] for key in list(layer.keys())]
        layer_inds.sort()   # used to find the number of parallel conv layers in this block; this is useful when there are conv blocks with multiple kernels being used
        min_ind = int(layer_inds[0])
        max_ind = int(layer_inds[-1])
        for ind in range(min_ind, max_ind+1):

            # add dense layer
            if f'dense_{ind}' in list(layer.keys()):

                # extract params

                # kernel_constraint
                if 'kernel_constraint' in list(layer[f'dense_{ind}'].keys()):
                    if layer[f'dense_{ind}']['kernel_constraint']['type'] == 'MaxNorm':
                        kernel_constraint = MaxNorm(max_value=layer[f'dense_{ind}']['kernel_constraint']['max_value'])
                    else:
                        raise Exception('Invalid kernel_constraint type')
                else:
                    kernel_constraint = None

                # bias_constraint
                if 'bias_constraint' in list(layer[f'dense_{ind}'].keys()):
                    if layer[f'dense_{ind}']['bias_constraint']['type'] == 'MaxNorm':
                        bias_constraint = MaxNorm(max_value=layer[f'dense_{ind}']['bias_constraint']['max_value'])
                    else:
                        raise Exception('Invalid bias_constraint type')
                else:
                    bias_constraint = None
                
                # kernel regularizer
                if 'kernel_regularizer' in list(layer[f'dense_{ind}'].keys()):
                    if 'l1' in layer[f'dense_{ind}']['kernel_regularizer'].keys():
                        if 'l' in list(layer[f'dense_{ind}']['kernel_regularizer']['l1'].keys()):
                            kernel_regularizer = l1(layer[f'conv_{ind}']['kernel_regularizer']['l1']['l'])
                        else:
                            kernel_regularizer = 'l1'
                    elif 'l2' in layer[f'dense_{ind}']['kernel_regularizer'].keys():
                        if 'l' in list(layer[f'dense_{ind}']['kernel_regularizer']['l2'].keys()):
                            kernel_regularizer = l2(layer[f'dense_{ind}']['kernel_regularizer']['l2']['l'])
                        else:
                            kernel_regularizer = 'l2'
                    else:
                        raise Exception('Kernel regularizer cannot be instantiated.')
                else:
                    kernel_regularizer = None

                # bias regularizer
                if 'bias_regularizer' in list(layer[f'dense_{ind}'].keys()):
                    if 'l1' in layer[f'dense_{ind}']['bias_regularizer'].keys():
                        if 'l' in layer[f'dense_{ind}']['bias_regularizer']['l1'].keys():
                            bias_regularizer = l1(layer[f'dense_{ind}']['bias_regularizer']['l1']['l'])
                        else:
                            bias_regularizer='l1'
                    elif 'l2' in layer[f'dense_{ind}']['bias_regularizer'].keys():
                        if 'l' in layer[f'dense_{ind}']['bias_regularizer']['l2'].keys():
                            bias_regularizer = l2(layer[f'dense_{ind}']['bias_regularizer']['l2']['l'])
                        else:
                            bias_regularizer='l2'
                    else:
                        raise Exception('Bias regularizer cannot be instantiated.')
                else:
                    bias_regularizer = None

                # kernel initializer
                if 'kernel_initializer' in list(layer[f'dense_{ind}'].keys()):
                    kernel_initializer = layer[f'dense_{ind}']['kernel_initializer']
                else:
                    kernel_initializer = 'glorot_uniform'

                # bias initializer
                if 'bias_initializer' in list(layer[f'dense_{ind}'].keys()):
                    kernel_initializer = layer[f'dense_{ind}']['bias_initializer']
                else:
                    bias_initializer = 'glorot_uniform'

                dense = Dense(layer[f'dense_{ind}']['units'], activation=layer[f'dense_{ind}']['activation'], kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(prev_layer)
                prev_layer = dense
            
            # add bn layer
            if f'bn_{ind}' in list(layer.keys()):
                bn = BatchNormalization()(prev_layer)
                prev_layer = bn

            # add act layer
            if f'act_{ind}' in list(layer.keys()):
                act_layer = Activation(activation=layer[f'act_{ind}']['activation'])(prev_layer)
                prev_layer = act_layer

            # add dropout layer
            if f'drop_{ind}' in list(layer.keys()):
                drop = Dropout(rate = layer[f'drop_{ind}']['rate'])(prev_layer)
                prev_layer = drop

        return prev_layer

    def _create_output_layer(self, i, prev_layer):
        '''
        Creates a dense + batch norm + activation + dropout layers with the index i
        
        Parameters
            i (str): index of the layer
            prev_layer (keras.layers): layer after which this convolution layer is to be added 

        Return
            dense (keras.layers): dense layer created
        '''
        layer = self.params['layers']['output']
        # extract params

        # kernel_constraint
        if 'kernel_constraint' in list(layer.keys()):
            if layer['kernel_constraint']['type'] == 'MaxNorm':
                kernel_constraint = MaxNorm(max_value=layer['kernel_constraint']['max_value'])
            else:
                raise Exception('Invalid kernel_constraint type')
        else:
            kernel_constraint = None

        # bias_constraint
        if 'bias_constraint' in list(layer.keys()):
            if layer['bias_constraint']['type'] == 'MaxNorm':
                bias_constraint = MaxNorm(max_value=layer['bias_constraint']['max_value'])
            else:
                raise Exception('Invalid bias_constraint type')
        else:
            bias_constraint = None
        
        # kernel regularizer
        if 'kernel_regularizer' in list(layer.keys()):
            if 'l1' in layer['kernel_regularizer'].keys():
                if 'l' in (layer['kernel_regularizer']['l1'].keys()):
                    kernel_regularizer = l1(layer['kernel_regularizer']['l1']['l'])
                else:
                    kernel_regularizer = 'l1'
            elif 'l2' in layer['kernel_regularizer'].keys():
                if 'l' in (layer['kernel_regularizer']['l2'].keys()):
                    kernel_regularizer = l2(layer['kernel_regularizer']['l2']['l'])
                else:
                    kernel_regularizer = 'l2'
            else:
                raise Exception('Kernel regularizer cannot be instantiated.')
        else:
            kernel_regularizer = None

        # bias regularizer
        if 'bias_regularizer' in list(layer.keys()):
            if 'l1' in layer['bias_regularizer'].keys():
                if 'l' in layer['bias_regularizer']['l1'].keys():
                    bias_regularizer = l1(layer['bias_regularizer']['l1']['l'])
                else:
                    bias_regularizer='l1'
            elif 'l2' in layer['bias_regularizer'].keys():
                if 'l' in layer['bias_regularizer']['l2'].keys():
                    bias_regularizer = l2(layer['bias_regularizer']['l2']['l'])
                else:
                    bias_regularizer='l2'
            else:
                raise Exception('Bias regularizer could not be instantiated')
        else:
            bias_regularizer = None

        # kernel initializer
        if 'kernel_initializer' in list(layer.keys()):
            kernel_initializer = layer['kernel_initializer']
        else:
            kernel_initializer = 'glorot_uniform'

        # bias initializer
        if 'bias_initializer' in list(layer.keys()):
            bias_initializer = layer['bias_initializer']
        else:
            bias_initializer = 'glorot_uniform'

        dense = Dense(self.no_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(prev_layer)
        prev_layer = dense
    
        return prev_layer

    def _create_flat_layer(self, i, prev_layer):
        '''
        Creates a flat layer with the index i
        
        Parameters
            i (str): index of the layer
            prev_layer (keras.layers): layer after which this convolution layer is to be added 

        Return
            dense (keras.layers): dense layer created
        '''
        layer = self.params['layers'][i]
        if 'flatten_1' in list(layer.keys()):
            prev_layer = Flatten()(prev_layer)
        if 'drop_1' in list(layer.keys()):
            prev_layer = Dropout(rate=layer['drop_1']['rate'])(prev_layer)

        return prev_layer

    def build_model(self):
        '''
        Creates model according to self.params['layers'] dictionary.
        One activation layer, batch normalization and pooling layer are assumed to be attached to a conv layer in terms of numbering index. However for a given index it is possible that a conv layer isn't present but other types of layers are present.
        '''
        
        inputs = Input(shape=(self.seq_len, 1))     # input layer
        prev_layer = inputs

        # iterate through layer indices
        layers = [l for l in list(self.params['layers'].keys()) if l != 'output']
        layers.sort()
        for layer in layers:
            if 'conv_1' in list(self.params['layers'][layer].keys()) or 'pool_1' in list(self.params['layers'][layer].keys()):
                prev_layer = self._create_conv_layer(layer, prev_layer)
            elif 'flatten_1' in list(self.params['layers'][layer].keys()):
                prev_layer = self._create_flat_layer(layer, prev_layer)
            elif 'dense_1' in list(self.params['layers'][layer].keys()):
                prev_layer = self._create_dense_layer(layer, prev_layer)
            else:
                raise Exception(f'{layer} not recognised')
        
        # create output layer
        outputs = self._create_output_layer(layer, prev_layer)

        # return model
        model = Model(inputs=inputs, outputs=outputs)
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
                job_type='build model'
            ) as run:

            # build model
            self.build_model()

            # add initialised model as an artifact
            model_artifact = wandb.Artifact(self.params['run_name'], type="model", description="Initialized model", metadata=self.params)

            model_name_split = self.params['model_filename'].rsplit('.', 1)
            self.model.save(model_name_split[0] + '-initialized.' + model_name_split[1])

            model_artifact.add_file(model_name_split[0] + '-initialized.' + model_name_split[1])
            run.log_artifact(model_artifact)

    def train_model(self, data, mlflow_log=False, wandb_log=True, log_weights=False):
        '''
        Train model

        Parameters:
            data (str): run name of data artifact stored
            mlflow_log (bool): whether to log experiment in mlflow
            wandb_log (bool): whether to log experiment in wandb
            log_weights (bool): whether to log the min and max weight values in each epoch.
        '''
        ((X_train, y_train), (X_test, y_test)) = data
        # if mlflow_log:
        #     # set tracking_uri
        #     # mlflow.set_tracking_uri('/home/nithya/mlflow_runs/')
        #     # set experiment id
        #     experiment_id = mlflow.get_experiment_by_name(self.params['model_filename'].rsplit('/', 1)[0])
        #     if experiment_id is None:
        #         experiment_id = mlflow.create_experiment(self.params['model_filename'].rsplit('/', 1)[0])
        #     else:
        #         experiment_id = experiment_id.experiment_id
        #     run_name = self.params['model_filename'].rsplit('/', 1)[1].rsplit('.', 1)[0]
        #     mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        #     # log params
        #     mlflow.log_params({
        #         'batch_size': self.params['batch_size'],
        #         'epochs': self.params['epochs'],
        #         'optimizer': self.params['optimizer']['type'],
        #         'input-type': self.params['input-type'],
        #         'learning_rate': self.params['optimizer']['learning_rate'],
        #         'early_stopping': self.params['early_stopping']
        #         })
        #     if self.params['optimizer']['type'] == 'SGD':
        #         mlflow.log_params({
        #             'momentum': self.params['optimizer']['momentum'],
        #             'decay': self.params['optimizer']['decay']
        #         })
        #         steps_per_epoch = X_train.shape[0]//self.params['batch_size']
        #         for i in range(self.params['epochs']*steps_per_epoch):
        #             mlflow.log_metrics({
        #                 'lr': self.params['optimizer']['learning_rate'] * (1/(1+self.params['optimizer']['decay'] * i))
        #             }, step=i)
        #     mlflow.log_artifact(self.params['model_filename'].rsplit('.', 1)[0] + '-params.json')

        # model structure
        print(self.model.summary())
        if not os.path.isfile(os.path.join(self.params['model_filename'].rsplit('/', 1)[0], 'structure.txt')):
            # store the model structure in a txt file if not already stored
            with open(os.path.join(self.params['model_filename'].rsplit('/', 1)[0], 'structure.txt'), 'w') as f:
                with redirect_stdout(f):
                    print(self.model.summary())

        # set model optimizer
        if self.params['optimizer']['type'] == 'SGD':
            opt = SGD(learning_rate=self.params['optimizer']['learning_rate'] if 'learning_rate' in list(self.params['optimizer'].keys()) else None, momentum=self.params['optimizer']['momentum'] if 'momentum' in list(self.params['optimizer'].keys()) else None, decay=self.params['optimizer']['decay'] if 'decay' in list(self.params['optimizer'].keys()) else None)
        elif self.params['optimizer']['type'] == 'Adam':
            opt = Adam(learning_rate=self.params['optimizer']['learning_rate'] if 'learning_rate' in list(self.params['optimizer'].keys()) else None)
        elif self.params['optimizer']['type'] == 'Adagrad':
            # pdb.set_trace()
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
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        callback_values.append(model_checkpoint_callback)
        # log min and max weights
        if log_weights:
            json_log = open(self.params['model_filename'].rsplit('.', 1)[0]+ '-weight_log.csv', mode='wt', buffering=1)
            json_log.write('epoch,min weight,max weight\n')
            logger = LambdaCallback(on_epoch_end= lambda epoch, logs: json_log.write(f'{epoch},{np.min([np.min(np.ravel(lay.get_weights()[0])) for lay in self.model.layers[1:] if len(lay.get_weights()) > 0])},{np.max([np.max(np.ravel(lay.get_weights()[0])) for lay in self.model.layers[1:] if len(lay.get_weights()) > 0])}\n'), on_train_end=lambda logs: json_log.close())
            callback_values.append(logger)
        if self.params['early_stopping']:
            es = EarlyStopping(monitor='val_loss', mode='min', patience=100)
            callback_values.append(es)
        # if mlflow_log:
        #     # log loss/accuracy metrics during training
        #     mlflow_metrics_log = LambdaCallback(on_epoch_end= lambda epoch, logs:     mlflow.log_metrics({
        #             'Train Accuracy': logs['accuracy'],
        #             'Train Loss': logs['loss'],
        #             'Val Accuracy': logs['val_accuracy'],
        #             'Val Loss': logs['val_loss']
        #         }, step=epoch))
        #     callback_values.append(mlflow_metrics_log)
        if wandb_log:
            # log loss/accuracy metrics during training with wandb
            callback_values.append(WandbCallback(log_evaluation=True))
        # compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # fit model
        train_history = self.model.fit(X_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], callbacks=callback_values, validation_data=(X_test, y_test))
        if not wandb_log:
            metrics = self.plot_metrics(train_history)
        self.plot_cms(data)

        # if mlflow_log:
        #     # log model metrics    
        #     mlflow.log_metrics({
        #         'Model Train Loss': metrics['train_loss'],
        #         'Model Train Accuracy': metrics['train_accuracy'],
        #         'Model Test_Loss': metrics['val_loss'],
        #         'Model Test_Accuracy': metrics['val_accuracy'],
        #         'Epoch': metrics["min_val_ind"]
        #     })
        
        #     mlflow.keras.log_model(self.model, self.params['model_filename'].rsplit('/', 1)[1])
        #     mlflow.log_artifact(self.params['train_filename'])
        #     mlflow.log_artifact(self.params['test_filename'])
        #     mlflow.log_artifact(self.params['model_filename'].rsplit('.', 1)[0] + '-cm.png')
        #     # end run
        #     mlflow.end_run()
            

    def log_train(self, data_name, mlflow_log=False, log_weights=False):
        '''
        Logs training of model on wandb

        Parameters:
        data_name (str): run name of data artifact stored
        mlflow_log (bool): whether to log on mlflow as well
        log_weights (bool): logs minimum and maximum of weights in each layer of each epoch if True
        '''

        with wandb.init(
                project=self.params['project_name'],
                name=self.params['run_name'],
                tags = self.params['tags'],
                config = self.params,
                job_type='build model'
            ) as run:
            
            # load data from the artifact
            dataset = run.use_artifact(data_name + ':latest')
            dataset.download()
            train_data = np.load(self.params['train_file'])
            X_train, y_train = train_data['X'], train_data['y']
            test_data = np.load(self.params['test_file'])
            X_test, y_test = test_data['X'], test_data['y']

            # check if model has been initialised
            if self.model is None:
                raise Exception("Model hasn't been built yet")

            model_params = wandb.Artifact("model-architecture", type="json-file")
            model_params.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-params.json')
            run.log_artifact(model_params)

            self.train()

            model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description=f"model architecture defined at {self.params['model_filename']}",
            metadata=self.params)
            model_artifact.add_file(self.params['model_filename'])
            run.log_artifact(model_artifact)

            cm_artifact = wandb.Artifact(
            "cm", type="Evaluation-Confusion matrix",
            description=f"Confusion matrix from model with lowest validation loss during training")
            cm_artifact.add_file(self.params['model_filename'].rsplit('.', 1)[0] + '-cm.png')
            run.log_artifact(cm_artifact)
            
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

    def plot_cms(self, data):
        # load model and data
        model = keras.models.load_model(self.params['model_filename'])
        ((X_train, y_train), (X_test, y_test)) = data
        # plot confusion matrices
        figcm, axcm = plt.subplots(2, 1, figsize=(10, 20))
        
        # train cm
        sns.heatmap(confusion_matrix(np.argmax(y_train, axis=1), np.argmax(model.predict(X_train), axis=1)), xticklabels=self.raga_labels, yticklabels=self.raga_labels, annot=True, fmt="d", ax=axcm[0])
        axcm[0].set(xlabel='Predicted Label', ylabel='True Label', title='Train CM')

        # test cm
        sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1)), xticklabels=self.raga_labels, yticklabels=self.raga_labels, annot=True, fmt="d", ax=axcm[1])
        axcm[1].set(xlabel='Predicted Label', ylabel='True Label', title='Test CM')

        # save fig
        figcm.savefig(self.params['model_filename'].rsplit('.', 1)[0] + '-cm.png')