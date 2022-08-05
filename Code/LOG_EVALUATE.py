from util_functions import *
from objects import *
import sys

# variables
seq_len=1200
no_classes=9

def model_eval(params=None):
    '''
    Initialises, trains and evaluates a model

    Parameters
        params (dict): dictionary of hyperparameters used for the model
    '''

    modelObj = ModelInstance(params, seq_len, no_classes)
    modelObj.log_and_evaluate(log_metadata=True)
    #modelObj.log_and_evaluate_class()

def main(args):
    args = argParseEval(args)
    params = load_data_for_eval(args)
    # data_vals = data.log_and_load(params)
    model_eval(params)

if __name__=="__main__":
    args = sys.argv[1:]
    main(args)