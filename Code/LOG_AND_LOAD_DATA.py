import os
import sys
sys.path.append('../../CommonScripts/')
import json
import argparse
from objects import Data, ModelInstance
from util_functions import *
import pdb
'''
Logs datasets into wandb
'''

def main(args):
    args = argParse(args)
    data, params = load_data(args)
    _ = data.log_and_load(params)

if __name__=="__main__":
    args = sys.argv[1:]
    main(args)