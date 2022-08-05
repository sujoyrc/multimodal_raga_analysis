from ast import arg
import os
import sys
sys.path.append('../../CommonScripts/')
import json
import argparse
from objects import Data, ModelInstance
import tensorflow as tf
from util_functions import *
import pdb
'''
Runs a model for split_1 data. Model architecture + hyperparameters is picked up froma JSON file and model runs are logged with wandb
'''
# variables
no_classes=9    # number of classes
LOG_TRAIN_METADATA = False
LOG_EVALUATION = True

def model_train_eval(data=None, params=None):
    '''
    Initialises, trains and evaluates a model

    Parameters
        data (Data): data object with X, y train and test data
        params (dict): dictionary of hyperparameters used for the model
    '''

    modelObj = ModelInstance(params, no_classes)
    modelObj.log_train_model(data, log_train_metadata=LOG_TRAIN_METADATA, log_evaluation=LOG_EVALUATION)

def main(args):
    args = argParse(args)
    set_gpus([int(args.gpu)])
    data, params = load_data(args)
    # data_vals = data.log_and_load(params)
    if data is None:
        data_vals = None
    else:
        data_vals = data.load()
    model_train_eval(data_vals, params)

if __name__=="__main__":
    args = sys.argv[1:]
    main(args)