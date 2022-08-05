import yaml
import os
import json
import numpy as np
import pandas as pd
from energyContoursfunc import energyContours
from voicingContoursfunc import voicingContours
from pitchContoursfunc import pitchContours

# read yaml files that defines hyper-parameters and the location of data
def read_config(path='config.yaml'):
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

# audiolist=np.loadtxt('/home/kamini/Comprehensibility/audiolist_comprehensibility.txt',dtype=str)
audiolist=['AG_1a_Jaun']

configName='config.yaml'
config=read_config(configName)
config['segflag']= '1'

for key in ['contourfeatfolder']:#,'infofolder','wordfeatfolder','eventfolder']:
    os.system('mkdir -p '+config[key])

for audioName in audiolist:
    print(audioName)
    config['audioName']=audioName
    audiopath=config['audiofolder']+audioName+'.wav'
    print('energy')
    energyContours(audiopath,config['contourfeatfolder'])
    print('voicing')
    voicingContours(audiopath,config['modelfolder'],config['contourfeatfolder'])
    print('pitch')
    pitchContours(audiopath,config['contourfeatfolder'],read_config('pitchConstconfig.yaml'))

