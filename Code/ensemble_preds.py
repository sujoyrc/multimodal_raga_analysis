import wandb
from keras.models import load_model
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pdb

model_AG = "trained-model:v4168"
model_CC = "trained-model:v4171"
model_SCh = "trained-model:v4174"

model_names = ['AG', 'CC', 'SCh']
# data_video = "easy_1-SCh-finalVideo"
data_audio = "easy_1-SCh-audio-600"

data_arts = [data_audio]
model_arts = [model_AG, model_CC, model_SCh]

raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']

def _extract_data(data):
        '''
        Extracts the X, y and ids for train and test sets from the data extracted from the npz file
        '''
        # pdb.set_trace()
        X_train = []
        y_train = data[list(data.keys())[0]][0][1]
        X_test = []
        y_test = data[list(data.keys())[0]][1][1]
        train_ids = data[list(data.keys())[0]][0][2]
        test_ids = data[list(data.keys())[0]][1][2]
        for data_art in list(data.keys()):
            temp_train = []
            temp_test = []
            for id in train_ids:
                train_id = np.where(data[data_art][0][2] == id)[0][0]
                temp_train.append(data[data_art][0][0][train_id])     
            X_train.append(np.array(temp_train))
            for id in test_ids:
                test_id = np.where(data[data_art][1][2] == id)[0][0]
                temp_test.append(data[data_art][1][0][test_id])     
            X_test.append(np.array(temp_test))

        return X_train, y_train, X_test, y_test, train_ids, test_ids

def download_data(data_arts, run):
    '''
    downloads dataset from wandb

    Parameters
    ----------
    data_art    : str
        Name of the artifact (':latest' is automatically appened to this name in the code)
        
    run : wandb.Run
        Wandb run that has been initialised in the main function

    Returns
    -------
    [(X_train, y_train, train_id), (X_test, y_test, test_id)]
    '''
    data = {}
    for data_art in data_arts:
        artifact = run.use_artifact(f'snnithya/Gesture Analysis/{data_art}:latest', type='dataset')
        data_dir = artifact.download()
        train_data = np.load(os.path.join(data_dir, 'train' + '.npz'), allow_pickle=True)
        test_data = np.load(os.path.join(data_dir, 'test.npz'), allow_pickle=True)

        data[data_art] = [
            (train_data['X_0'], train_data['y'], train_data['ids']),
            (test_data['X_0'], test_data['y'], test_data['ids'])
        ]
        # # pdb.set_trace()
        # if 'pitch' in data_art:
        #     return [(train_data['X_0_mask'], train_data['y'], train_data['ids']), (test_data['X_0_mask'], test_data['y'], test_data['ids'])]
        # elif 'finalVideo' in data_art:
        #     return [(train_data['X_0'], train_data['y'], train_data['ids']), (test_data['X_0'], test_data['y'], test_data['ids'])]
    X_train, y_train, X_test, y_test, train_ids, test_ids = _extract_data(data)
    return [(X_train, y_train, train_ids), (X_test, y_test, test_ids)]

def download_model(model_art, run):
    '''
    Downloads a model from wandb

    Parameters
    ----------
    model_art   : str
        Name of model artifact (version number has to be attached)
    
    run : wandb.Run
        wandb run that is being used at the moment

    Returns
    -------
    model   : keras.Model
        Returns the keras model stored in wandb
    '''

    artifact = run.use_artifact(f'snnithya/Gesture Analysis/{model_art}', type='model')
    data_dir = artifact.download()
    for filename in os.listdir(data_dir):
        if filename.endswith('.hdf5'):
            model = load_model(os.path.join(data_dir, filename))
    return model

def ensemble_preds(models, data, data_art_name, run):
    '''
    Calculates predictions on a given train and test data separately and returns a mean of the softmax output

    Parameters
    ----------
    models : list of keras.Model objects
        Models that prediction have to be obtained from

    data : list of train and test data
        Of the form - [(X_train, y_train, train_id), (X_test, y_test, test_id)]

    data_art_name : str
        Name of the data artifact (used while logging the summary metrics)
    
    run : wandb.Run
        Run where the summary metrics will be logged
    
    Returns
    -------
    (preds_train, preds_test) : tuple of np.array
        Averaged softmax output
    '''
    preds_train = []
    preds_test = []
    for model_ind, model in enumerate(models):
        # train metrics
        loss, acc = model.evaluate(data[0][0], data[0][1])
        run.summary.update({
            f'train-loss-{data_art_name}_data-{model_names[model_ind]}_model': loss,
            f'train-accuracy-{data_art_name}_data-{model_names[model_ind]}_model': acc
        })

        # test metrics
        loss, acc = model.evaluate(data[1][0], data[1][1])
        run.summary.update({
            f'test-loss-{data_art_name}_data-{model_names[model_ind]}_model': loss,
            f'test-accuracy-{data_art_name}_data-{model_names[model_ind]}_model': acc
        })

        # predictions
        preds_train.append(model.predict(data[0][0]))
        preds_test.append(model.predict(data[1][0]))
    preds_train = np.mean(preds_train, axis=0)
    preds_test = np.mean(preds_test, axis=0)

    return (preds_train, preds_test, run)
    
def get_accuracy(y_true, y_pred):
    '''
    Get accuracy from true classes and predicted classes

    Parameters
    ----------
    y_true : np.array
        Array of true classes

    y_pred : np.array
        Array of predicted classes

    Returns accuracy
    '''
    correct_preds = np.count_nonzero(y_true == y_pred)

    return np.around(correct_preds/y_true.shape[0], 5)

def plot_cm(y_train, y_train_pred, y_test, y_test_pred):

    figcm, axcm = plt.subplots(1, 2, figsize=(20, 10))

     # train cm
    sns.heatmap(confusion_matrix(np.argmax(y_train, axis=1), np.argmax(y_train_pred, axis=1)), xticklabels=raga_labels, yticklabels=raga_labels, annot=True, fmt="d", ax=axcm[0])
    axcm[0].set(xlabel='Predicted Label', ylabel='True Label', title='Train CM')

    # test cm
    sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1)), xticklabels=raga_labels, yticklabels=raga_labels, annot=True, fmt="d", ax=axcm[1])
    axcm[1].set(xlabel='Predicted Label', ylabel='True Label', title='Test CM')

    return figcm

def main():
    print('Starting wandb run')
    with wandb.init(
        job_type="evaluation",
        project="Gesture Analysis"
    ) as run:
        print('Loading data')
        data = download_data(data_arts, run)

        print('Loading models')
        models = []
        for model_art in model_arts:
            models.append(download_model(model_art, run))

        print('Ensembling Predictions')
        train_preds, test_preds, run = ensemble_preds(models, data, data_arts[0].rsplit('-', 2)[1], run)
        # pdb.set_trace()
        train_table = wandb.Table(columns=[], data=[])
        train_table.add_column('unique_id', data[0][2])
        train_table.add_column('true_class', np.argmax(data[0][1], axis=1))
        train_table.add_column('predicted_class', np.argmax(train_preds, axis=1))
        for label in range(train_preds.shape[1]):
            train_table.add_column('prediction_probability_' + str(label), train_preds[:, label])

        test_table = wandb.Table(columns=[], data=[])
        test_table.add_column('unique_id', data[1][2])
        test_table.add_column('true_class', np.argmax(data[1][1], axis=1))
        test_table.add_column('predicted_class', np.argmax(test_preds, axis=1))
        for label in range(test_preds.shape[1]):
            test_table.add_column('prediction_probability_' + str(label), test_preds[:, label])

        preds = wandb.Artifact(f"Predictions-{data_arts[0].rsplit('-', 2)[1]}", "evaluation")
        preds['train_table'] = train_table
        preds['test_table'] = test_table
        run.log_artifact(preds)

        train_acc = get_accuracy(np.argmax(data[0][1], axis=1), np.argmax(train_preds, axis=1))
        test_acc = get_accuracy(np.argmax(data[1][1], axis=1), np.argmax(test_preds, axis=1))

        run.summary.update({
            f'train_accuracy-{data_arts[0].rsplit("-", 2)[1]}_data-ensemble_model': train_acc,
            f'test_accuracy-{data_arts[0].rsplit("-", 2)[1]}_data-ensemble_model': test_acc,
        })

        # cm 
        figcm = plot_cm(
            y_train=data[0][1],
            y_train_pred=train_preds,
            y_test=data[1][1],
            y_test_pred=test_preds
        )
        run.log({
            'Confusion Matrix': wandb.Image(figcm)
        })

main()