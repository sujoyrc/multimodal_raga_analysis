'''
Basic utility functions used throughout the project
'''

import os
import warnings
import re
import pdb

def addBack(file_path):
    '''Adds a backslash to the file path
    
    Parameters
        file_path (str): file path 
    Returns
        file_path (str): file path with a trailing backslash
    '''
    if file_path[-1] == '/':
        return file_path
    else:
        return file_path+'/'

def checkPath(file_path):
    '''Check if a folder exists
    
    Parameters
        file_path (str): file path to check
    Returns
        file_path (str): folder or file path that exists now
    '''
    if re.match(".+\.[a-zA-z1-9]{3,5}$", file_path):
        # file_path is for a file
        file_path, file_name = file_path.rsplit('/', 1) # remove the file name
    else:
        file_name = ''
    # file_path is a folder path
    if not os.path.exists(file_path):
        # create folder
        os.makedirs(file_path)

    return os.path.join(file_path, file_name)

