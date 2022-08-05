import os
import sys
sys.path.append('../../CommonScripts/')
import json
import argparse
from objects import Data, ModelInstance
from util_functions import *
import wandb
import tensorflow as tf
import pdb
'''
Runs runs a sweep (hyperparameter tuning) on a given model architecture according to sweep config params. 
'''
# variables
seq_len=1200
no_classes=9
data=None
params=None

def model_train_eval():
    '''
    Initialises, trains and evaluates a model

    Parameters
        data (Data): data object with X, y train and test data
        params (dict): dictionary of hyperparameters used for the model
    '''
    global data, params
    # data, params = load_data(args)
    params['model_filename'], params['run_name'] = get_model_name(params['json_filename'])  # update model name
    # params['run_name'] = params['model_filename'].rsplit('/', 1)[1].rsplit('.', 1)[0]
    modelObj = ModelInstance(params, no_classes)
    modelObj.log_train_model()
    # modelObj.log_and_evaluate()

def main(args):
    global data, params
    
    args = argParse(args)
    set_gpus([int(args.gpu)])
    data, params = load_data(args)
    data_vals = None

    # load sweep config
    if 'sweep_file' not in list(params.keys()):
        raise Exception('Sweep config file not given as arguement')
    with open(params['sweep_file'], 'r') as f:
        sweep_config = json.load(f)
    
    # initialize sweep
    # sweep_id = wandb.sweep(sweep_config, project=params['project_name'])
    wandb.agent("61lqe6f9", project="Gesture Analysis", function=model_train_eval)

if __name__=="__main__":
    args = sys.argv[1:]
    main(args)