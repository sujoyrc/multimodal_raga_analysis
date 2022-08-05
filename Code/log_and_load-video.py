from email.policy import default
import wandb
import argparse
import os

dataArtifact = 'hard_2-SCh-video'
tags = ['hard_2', 'SCh', 'video']
dataFolder = '../Seqs/finalDataVideo/hard_2/SCh/'

def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='Uploading video data')

    parser.add_argument(
        '-pn',
        '--project_name',
        default='Gesture Analysis',
        help='name of the project on wandb'
    )
    parser.add_argument(
        '-da',
        '--data_artifact',
        default=dataArtifact,
        help='Name of the artifact on wandb'
    )
    parser.add_argument(
        '-t', '--tags',
        default=tags,
        help='tags to add to data artifact'
    )
    parser.add_argument(
        '-df', '--data_folder',
        default=dataFolder,
        help='path to data folder'
    )

    return parser

def main():
    p = get_parser()

    arg = p.parse_args()

    with wandb.init(
        project=arg.project_name,
        name=arg.data_artifact,
        tags = arg.tags,
        job_type='dataset'
    ) as run:

        run.config.update({
            'data folder': arg.data_folder,
            'data artifact name': arg.data_artifact
        })
        
        data_art = wandb.Artifact(arg.data_artifact, 
        type="dataset", 
        description=arg.data_folder
        )

        for fileName in os.listdir(arg.data_folder):
            data_art.add_file(local_path = os.path.join(arg.data_folder, fileName), name=fileName)
        run.log_artifact(data_art)

main()