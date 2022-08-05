import pandas as pd
import json
import wandb
from collections import defaultdict

SUMMARY_FILE = '../Seqs/summary.csv'

DEST_FILE = '../Seqs/id_to_raga.json'

'''
This script creates a json file with filename: lis of unique_ids. This file is then logged to wandb
'''

def create_json(summaryFile=SUMMARY_FILE, destFile = DEST_FILE):
    summary = pd.read_csv(summaryFile)
    json_dict = {}
    for filename, df in summary.groupby(by='filename'):
        json_dict[filename] = df['unique_id'].values

    with open(destFile, 'w') as f:
        json.dump(json_dict, f)

def log_json(jsonFile=DEST_FILE):
    with wandb.init(
                project='Gesture Analysis',
                name=TASK + ' json upload',
                tags = 'raga prediction',
                job_type='dataset'
            ) as run:
        artifact = wandb.Artifact('Metadata for file predictions', 
            type="metadata", 
            description=f"Mapping of filenames to unique ids", 
            metadata={
                "data_file_path": jsonFile
                })
        artifact.add_file(jsonFile)
        run.log_artifact(artifact)

create_json()
log_json()