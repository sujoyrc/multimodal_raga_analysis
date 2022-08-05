import json
import argparse
from objects import Data
import os
import pdb
import tensorflow as tf

def set_gpus(gpu_ids: list):
    gpus = tf.config.list_physical_devices('GPU')
    try:
        available = [gpus[i] for i in gpu_ids]
        tf.config.set_visible_devices(available, 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"\n[GPU Setup] {len(gpus)} Physical GPUs | {len(logical_gpus)} Logical GPU\n")
    except RuntimeError as e:
        print("\n[GPU Setup Error]", e)
    except IndexError as e:
        print("\n[GPU SETUP] NO GPUs")

def argParse(args):
    '''
    Parses through args from command line arguements and returns them

    Parameters
        *args: args from command line; collected from sys.argv[1:]

    Returns
        parser_args(Namespace): parsed arguements
    '''

    parser = argparse.ArgumentParser(description="Train and Evaluate model")
    parser.add_argument("params_file", help="path to the JSON file with information about the model architecture")
    
    # optional arguements
    parser.add_argument("-sf", "--sweep_file", help="Path to sweep config file")
    parser.add_argument("--data_filename", help="Path to file with data")
    parser.add_argument("--model_filename", help="Path to model file")
    parser.add_argument("-data", "--data_artifact", help="Name of the data artefact to use")
    parser.add_argument("-b", "--batch_size", help="Overrides the default batch size presented in params_file")
    parser.add_argument("-e", "--epochs", help="Overrides the default number of epochs presented in params_file")
    parser.add_argument('-opt', "--optimizer", help="Overrides the default optimizer stated in params_file. Options are `Adam` and `SGD`")
    parser.add_argument('-lr', '--learning_rate', help="Overrides the defaultlearning rate present in the params_file")
    parser.add_argument('-m', '--momentum', help="Overrides the default momentum present in the params_file. Relevant only for SGD optimizer")
    parser.add_argument('-d', '--decay', help="Overrides the default decay for lr present in the params_file. Relevant only for SGD optimizer")
    parser.add_argument('-es', '--early_stopping', help='If provided will implement early stopping during the training')
    parser.add_argument('-pn', '--project_name', help="Name of project to store experiment runs in")
    parser.add_argument('-rn', '--run_name', help="Specify a run name for experiment tracking")
    parser.add_argument('-t','--tags', nargs='+', help='Adds tags to the model run with wandb tracking')
    parser.add_argument('-sc', '--split_channel', help="If true, will process each channel separately. DEPRECATED.")
    parser.add_argument('-msk', '--mask', default=False, help='If true will add the mask as a separate channel. DEPRECATED.')
    parser.add_argument('-mc', '--mask_channels', nargs='+', default=['pitch-pitch'], help='If sc=1, the attaches the mask to each channel listed here. By default attached the mask to the pitch channel. DEPRECATED.')
    parser.add_argument("-g", "--gpu", default=1, help="indicate which gpu to use {0 or 1}")

    return parser.parse_args(args)

def argParseEval(args):
    '''
    Parses through args from command line arguements and returns them for eval function

    Parameters
        *args: args from command line; collected from sys.argv[1:]

    Returns
        parser_args (Namespace): parsed arguements
    '''

    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("model_filename", help="Path to model file")
    parser.add_argument("data_artifact", help="Name of the data artefact to use")
    parser.add_argument("-g", "--gpu", help="indicate which gpu to use {0 or 1}")
    # optional arguements
    parser.add_argument("-p", "--params_file", help="path to the JSON file with information about the model architecture")
    parser.add_argument("-b", "--batch_size", help="Overrides the default batch ")
    parser.add_argument('-pn', '--project_name', help="Name of project to store experiment runs in")
    parser.add_argument('-rn', '--run_name', help="Specify a run name for experiment tracking")
    parser.add_argument('-t','--tags', nargs='+', help='Adds tags to the model run with wandb tracking')
    parser.add_argument('-sc', '--split_channel', help="If true, will process each channel separately. DEPRECATED.")
    parser.add_argument('-msk', '--mask', default=True, help='If true will add the mask as a separate channel. DEPRECATED.')
    parser.add_argument('-mc', '--mask_channels', nargs='+', default=['pitch-pitch'], help='If sc=1, the attaches the mask to each channel listed here. By default attached the mask to the pitch channel. DEPRECATED.')

    return parser.parse_args(args)

def load_data(args):
    '''
    Loads model hyperparameters and data

    Parameters
        args(Namespace): Namespace object containing the arguements passed through command line. Return value from argParse
    
    Returns
        data(Data): object containing train and test data
        params(dict): dictionary value with training parameters and model architecture
    '''
    # model architecture
    with open(args.params_file, 'r') as f:
        json_vals = f.read()
    params = json.loads(json_vals)
    params['json_filename'] = args.params_file  # stores the json file path, used as artifact for documentation purpose
    params['model_filename'] = args.model_filename if args.model_filename is not None else get_model_name(params['json_filename'])[0]  # add model filename to parameters
    if args.data_filename is not None:
        params['data_filename'] = args.data_filename
    if args.data_artifact is not None:
        params['data_artifact'] = args.data_artifact
    if args.sweep_file is not None:
        params['sweep_file'] = args.sweep_file
    # check if early stopping was given in json, else add the key
    
    # set early stopping based on commandline arg followed by params json values (in that order)
    if args.early_stopping is not None:
        params['early_stopping'] = args.early_stopping
    elif not 'early_stopping' in params.keys():
        params['early_stopping'] = False

    # set split channel bool
    if args.split_channel is not None:
        params['split_channel'] = bool(args.split_channel)
    elif not 'split_channel' in params.keys():
        params['split_channel'] = False
    # check if any training parameters were given in the command line
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.optimizer is not None:
        params['optimizer']['type'] = args.optimizer
        params['optimizer']['learning_rate'] = args.learning_rate
        if params['optimizer']['type'] == 'Adam':
            params['optimizer']['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else 0.001   # if this values are set to None, default learning rate will be considered with the optimizer
        elif args.optimizer == 'SGD':
            # momentum is relevant only for SGD
            params['optimizer']['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else 0.01   # if this values are set to None, default learning rate will be considered with the optimizer
            params['optimizer']['momentum'] = float(args.momentum) if args.momentum is not None else 0    # if this values are set to None, default momentum will be considered with the optimizer
            params['optimizer']['decay'] = float(args.decay) if args.decay is not None else 0
        elif args.optimizer == 'Adagrad':
            params['optimizer']['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else 0.001
        elif args.optimizer == 'Adadelta':
            params['optimizer']['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else 0.001
        elif args.optimizer == 'RMSprop':
            params['optimizer']['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else 0.001
    
    params['project_name'] = args.project_name if args.project_name is not None else "Gesture Analysis - Temp"
    params['run_name'] = args.run_name if args.run_name is not None else get_model_name(params['json_filename'])[1]
    # add command line tags to tags defined in json
    if 'tags' in list(params.keys()):
        if args.tags is not None:
            params['tags'].extend(args.tags)
    else:
        if args.tags is not None:
            params['tags'] = args.tags
        else:
            params['tags'] = []
    params['tags'].append(params['json_filename'].rsplit('/', 1)[0].rsplit("/", 1)[1])  # append model folder as a tag
    
    # load model data
    if 'data_filename' in list(params.keys()):
        data = Data(params['data_filename'])
    else:
        data = None

    if "log_metadata" not in params.keys():
        # option to log metadata while logging the dataset
        params['log_metadata'] = False

    # add mask variables
    params['mask'] = bool(int(args.mask))
    params['mask_channels'] = args.mask_channels
    params['gpu'] = args.gpu
    return data, params

def load_data_for_eval(args):
    '''
    Loads hyperparameters for data evaluation

    Parameters
        args(Namespace): Namespace object containing the arguements passed through command line. Return value from argParse
    
    Returns
        params(dict): dictionary value with training parameters and model architecture
    '''
    # parameters
    if args.params_file is not None:
        with open(args.params_file, 'r') as f:
            json_vals = f.read()
        params = json.loads(json_vals)
    else:
        params = {}
    params['json_filename'] = args.params_file  # stores the json file path, used as artifact for documentation purpose
    params['model_filename'] = args.model_filename if args.model_filename is not None else get_model_name(params['json_filename'])[0]  # add model filename to parameters
    if args.data_artifact is not None:
        params['data_artifact'] = args.data_artifact
    # set split channel bool
    if args.split_channel is not None:
        params['split_channel'] = bool(args.split_channel)
    elif not 'split_channel' in params.keys():
        params['split_channel'] = False
    # check if any training parameters were given in the command line
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    else:
        params['batch_size'] = None
    
    params['project_name'] = args.project_name if args.project_name is not None else "Gesture Analysis - Temp"
    params['run_name'] = args.run_name if args.run_name is not None else get_model_name(params['json_filename'])[1]
    # add command line tags to tags defined in json
    if 'tags' in list(params.keys()):
        if args.tags is not None:
            params['tags'].extend(args.tags)
    else:
        if args.tags is not None:
            params['tags'] = args.tags
        else:
            params['tags'] = []
    if params['json_filename'] is not None:
        params['tags'].append(params['json_filename'].rsplit('/', 1)[0].rsplit("/", 1)[1])  # append model folder as a tag
    # add mask variables
    params['mask'] = bool(int(args.mask))
    params['mask_channels'] = args.mask_channels
    return params

def get_model_name(params_file):
    '''
    Returns a name for the new model in the format of model_{i} where i is the index of the model being trained with the same params file
    
    Parameters
        params_file (str): file path to params file
    Returns
        model_name (str): file path to store model at
    '''
    
    model_folder = params_file.rsplit('/', 1)[0]    # model folder is defined as parent folder of params file
    i = 0   # search for lowest index not used yet
    while True:
        if not os.path.isfile(os.path.join(model_folder, f'{model_folder.rsplit("/", 1)[1]}-model_{i}.hdf5')):
            break
        else:
            i += 1
    return os.path.join(model_folder, f'{model_folder.rsplit("/", 1)[1]}-model_{i}.hdf5'), i