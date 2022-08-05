import keras
import wandb
import numpy as np
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

data_file = '../Seqs/finalVideoInception/easy_1-channel-separate/AG-norm.npz'
art_name = 'easy_1-AG-channel-separate'
model_filename = 'model.hdf5'

inception_blocks = {
        "a": {
                "common": {
                    "conv": {
                        "kernel_size": 5
                    }
                },
                "branch_1": {
                    "input": "input",
                    "conv": {
                        "filter": 10,
                        "kernel_size": 1,
                        "activation": "linear",
                        "strides": 2,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    },
                    "bn": {},
                    "act": {
                        "activation": "relu"
                    }
                },
                "branch_2_1": {
                    "input": "input",
                    "conv": {
                        "filter": 25,
                        "kernel_size": 1,
                        "activation": "linear",
                        "strides": 1,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    },
                    "bn": {},
                    "act": {
                        "activation": "relu"
                    }
                },
                "branch_2_2": {
                    "input": "branch_2_1",
                    "conv": {
                        "filter": 17,
                        "kernel_size": 3,
                        "activation": "linear",
                        "strides": 2,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    },
                    "bn": {},
                    "act": {
                        "activation": "relu"
                    }
                },
                "branch_3_1": {
                    "input": "input",
                    "conv": {
                        "filter": 19,
                        "kernel_size": 1,
                        "activation": "linear",
                        "strides": 1,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    },
                    "bn": {},
                    "act": {
                        "activation": "relu"
                    }
                },
                "branch_3_2": {
                    "input": "branch_3_1",
                    "conv": {
                        "filter": 23,
                        "kernel_size": 3,
                        "activation": "linear",
                        "strides": 1,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    },
                    "bn": {},
                    "act": {
                        "activation": "relu"
                    }
                },
                "branch_3_3": {
                    "input": "branch_3_2",
                    "conv": {
                        "filter": 29,
                        "kernel_size": 3,
                        "activation": "linear",
                        "strides": 2,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    },
                    "bn": {},
                    "act": {
                        "activation": "relu"
                    }
                },
                "branch_4_1": {
                    "input": "input",
                    "pool": {
                        "type": "Average",
                        "pool_size": 10,
                        "strides": 1,
                        "padding": "same"
                    }
                },
                "branch_4_2": {
                    "input": "branch_4_1",
                    "conv": {
                        "filter": 24,
                        "kernel_size": 1,
                        "activation": "linear",
                        "strides": 2,
                        "kernel_regularizer": {
                            "l2": {
                                "l": 0.01
                            }
                        },
                        "padding": "same"
                    }
                },
                "in_1_concat": {
                    "input": [
                        "branch_1",
                        "branch_2_2",
                        "branch_3_3",
                        "branch_4_2"
                    ],
                    "concat": {
                        "axis": -1
                    },
                    "drop": {
                        "rate": 0.3
                    }
                }
            },
            "b": {
            "common": {
                "conv": {
                    "kernel_size": 7
                }
            },
            "branch_1": {
                "input": "input",
                "conv": {
                    "filter": 23,
                    "kernel_size": 1,
                    "activation": "linear",
                    "strides": 2,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                },
                "bn": {},
                "act": {
                    "activation": "relu"
                }
            },
            "branch_2_1": {
                "input": "input",
                "conv": {
                    "filter": 18,
                    "kernel_size": 1,
                    "activation": "linear",
                    "strides": 1,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                },
                "bn": {},
                "act": {
                    "activation": "relu"
                }
            },
            "branch_2_2": {
                "input": "branch_2_1",
                "conv": {
                    "filter": 8,
                    "kernel_size": 3,
                    "activation": "linear",
                    "strides": 2,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                },
                "bn": {},
                "act": {
                    "activation": "relu"
                }
            },
            "branch_3_1": {
                "input": "input",
                "conv": {
                    "filter": 12,
                    "kernel_size": 1,
                    "activation": "linear",
                    "strides": 1,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                },
                "bn": {},
                "act": {
                    "activation": "relu"
                }
            },
            "branch_3_2": {
                "input": "branch_3_1",
                "conv": {
                    "filter": 18,
                    "kernel_size": 3,
                    "activation": "linear",
                    "strides": 1,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                },
                "bn": {},
                "act": {
                    "activation": "relu"
                }
            },
            "branch_3_3": {
                "input": "branch_3_2",
                "conv": {
                    "filter": 14,
                    "kernel_size": 3,
                    "activation": "linear",
                    "strides": 2,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                },
                "bn": {},
                "act": {
                    "activation": "relu"
                }
            },
            "branch_4_1": {
                "input": "input",
                "pool": {
                    "type": "Average",
                    "pool_size": 3,
                    "strides": 1,
                    "padding": "same"
                }
            },
            "branch_4_2": {
                "input": "branch_4_1",
                "conv": {
                    "filter": 28,
                    "kernel_size": 1,
                    "activation": "linear",
                    "strides": 2,
                    "kernel_regularizer": {
                        "l2": {
                            "l": 0.01
                        }
                    },
                    "padding": "same"
                }
            },
            "in_1_concat": {
                "input": [
                    "branch_1",
                    "branch_2_2",
                    "branch_3_3",
                    "branch_4_2"
                ],
                "concat": {
                    "axis": -1
                },
                "drop": {
                    "rate": 0.3
                }
            }
        }
}

def _create_conv_layer(block_params, prev_layer, override_stride=False):
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
        
        # create the conv layer
        conv_layer = Conv1D(
            filters=block_params[f'conv']['filter'], 
            kernel_size=block_params[f'conv']['kernel_size'], 
            activation=block_params[f'conv']['activation'], 
            strides=block_params[f'conv']['strides'], 
            kernel_regularizer=l2(l=block_params[f'conv']['kernel_regularizer']['l2']['l']) if 'kernel_regularizer' in list(block_params[f'conv'].keys()) else None, 
            bias_regularizer=l2(l=block_params[f'conv']['bias_regularizer']['l2']['l']) if 'bias_regularizer' in list(block_params[f'conv'].keys()) else None, 
            padding=block_params[f'conv']['padding'] if 'padding' in list(block_params['conv']) else 'valid',
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
    elif 'concat' in list(block_params.keys()):
        prev_layer = self._create_concatenate_layer(block_params, prev_layer)
    else:
        raise Exception(f'{block_name} not recognised')
    
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

def _create_inception_module(inception_name, prev_layer):
    '''
    Creates an inception module with block_params

    Parameters
        block_name (str): name of inception module
        block_params (dict): dictionary of details of the inception module
        prev_layer (keras.layers): layer to connect the inception module to

    Results
        prev_layer (keras.layers): last layer of the inception block
    '''
    inception_params = inception_blocks[inception_name]
    layers =  list(inception_params.keys())
    layer_objs = {
        'input': prev_layer
    }     # dictionary with layer objects
    for layer_ind, layer in enumerate(layers):
    
        prev_layer = layer_objs[inception_params[layer]['input']]
        
        # if kernel size of this block is specified by the "common kernel size" of this inception module
        print(layer)
        if 'conv' in list(inception_params[layer].keys()) and inception_params[layer]['conv']['kernel_size'] == 'common':
            inception_params[layer]['conv']['kernel_size'] = inception_params['common']['conv']['kernel_size']
        layer_objs[layer] = _create_block(inception_name + '_' + layer, inception_params[layer], prev_layer)

    return layer_objs[layers[-1]]

def load_data():
    with wandb.init(
            project='Gesture Analysis',
            name='Channel separate video model',
            tags = ['video data', 'channel separate'],
            job_type='dataset'
        ) as run:

        run.config.update({
            'data file': data_file,
            'data artifact name': 'easy_1-AG-channel-separate'
        })

        data_art = wandb.Artifact(art_name, 
        type="dataset", 
        description=data_file.rsplit('/', 2)[1])

        data_art.add_file(
            local_path=data_file
        )

def model():
    input_layers = [Input(shape=(300, 3)) for i in range(11)]
    conv_layers = [Conv1D(
                filters=8, 
                kernel_size=5, 
                activation='linear', 
                strides=1, 
                kernel_regularizer=l2(l=0.01), 
                use_bias=False,
                padding='same')(input_layers[i]) for i in range(11)]
    bn_layers = [BatchNormalization()(conv_layers[i]) for i in range(11)]
    act_layers = [Activation(activation='relu')(bn_layers[i]) for i in range(11)
    ]
    concat_layer = Concatenate(axis=-1)([act_layers[i] for i in range(11)])

    in_1 = _create_inception_module('a', concat_layer)
    in_2 = _create_inception_module('b', in_1)

    av_pool = AveragePooling1D(pool_size=73, strides=73)(in_2)
    flat = Flatten()(av_pool)
    drop = Dropout(rate=0.4)(flat)
    output = Dense(9, kernel_regularizer=l2())(drop)

    mod = Model(inputs=input_layers, outputs=output)

    return mod

def train(mod):
    with wandb.init(
            project='Gesture Analysis',
            name='Channel separate video model',
            tags = ['video data', 'channel separate'],
            job_type='dataset'
        ) as run:
        dataset = run.use_artifact(art_name + ':latest')
        data_dir = dataset.download()
        data = np.load(os.path.join(data_dir, 'data.npz'))

        X_train = data['X_train']
        X_train = np.array([np.reshape(X_train[:, 300, 3, i], (-1, 300, 3)) for i in range(11)])
        y_train = to_categorical(data['y_train'])
        X_test = data['X_test']
        X_test = np.array([np.reshape(X_test[:, 300, 3, i], (-1, 300, 3)) for i in range(11)])
        y_test = to_categorical(data['y_test'])

        opt = Adam(learning_rate=0.001)

        callback_values = []

        model_checkpoint_callback = ModelCheckpoint(
        filepath=model_filename,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        callback_values.append(model_checkpoint_callback)

        es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
        callback_values.append(es)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
        factor=0.9, 
        min_lr= 0.000001)
        callback_values.append(reduce_lr)

        callback_values.append(WandbCallback(log_evaluation=False))

        mod.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        mod.fit(X_train, y_train, batch_size=64, epochs=self.params['epochs'], callbacks=callback_values, validation_data=(X_test, y_test), shuffle=True)  

        train_loss, train_acc, test_loss, test_acc = eval(X_train, y_train, X_test, y_test)

        run.summary.update({'train loss': train_loss, 'train accuracy': train_acc, 'test loss': test_loss, 'test accuracy': test_acc})


def eval(X_train, y_train, X_test, y_test):
    mod = load_model(model_filename)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=1, batch_size=64)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1, batch_size=64)

    return train_loss, train_acc, test_loss, test_acc


def main():
    mod = model()
    train(mod)

main()