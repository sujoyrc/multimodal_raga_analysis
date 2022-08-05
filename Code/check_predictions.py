import wandb
from keras.models import load_model
import os
import numpy as np
from keras.models import Model
from keras.layers import Input
import json
import pdb

model_combine = 'trained-model:v3108'
video_model = 'trained-model:v2512'
audio_model = 'trained-model:v1612'
model_arts = [model_combine, video_model, audio_model]
data = ['easy_1-AG-finalVideo:v0', 'easy_1-AG-audio-1200:v0']

models = [None, None, None]
datas = {}

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
                # print(id, train_id)
                temp_train.append(data[data_art][0][0][train_id])     
            X_train.append(np.array(temp_train))
            for id in test_ids:
                test_id = np.where(data[data_art][1][2] == id)[0][0]
                # print(id, test_id)
                temp_test.append(data[data_art][1][0][test_id])     
            X_test.append(np.array(temp_test))

        return ((X_train, y_train, train_ids, None), (X_test, y_test, test_ids, None))

with wandb.init() as run:
    for ind, model_art in enumerate(model_arts):
        art = run.use_artifact('snnithya/Gesture Analysis/' + model_art)
        art_dir = art.download()
        models[ind] = load_model(os.path.join(art_dir, os.listdir(art_dir)[0]))

    art = run.use_artifact('snnithya/Gesture Analysis/' + data[0])
    art_dir = art.download()
    train_data = np.load(os.path.join(art_dir, 'train' + '.npz'), allow_pickle=True)
    test_data = np.load(os.path.join(art_dir, 'test.npz'), allow_pickle=True)
    X_train, y_train, train_ids, mask_train = train_data['X_0'], train_data['y'], train_data['ids'], train_data['mask']
    test_data = np.load(os.path.join(art_dir, 'test' + '.npz'), allow_pickle=True)
    X_test, y_test, test_ids, mask_test = test_data['X_0'], test_data['y'], test_data['ids'], test_data['mask']
    datas['video'] = ((X_train, y_train, train_ids, mask_train), (X_test, y_test, test_ids, mask_test))

    art = run.use_artifact('snnithya/Gesture Analysis/' + data[1])
    art_dir = art.download()
    train_data = np.load(os.path.join(art_dir, 'train' + '.npz'), allow_pickle=True)
    test_data = np.load(os.path.join(art_dir, 'test.npz'), allow_pickle=True)
    X_train, y_train, train_ids, mask_train = train_data['X_0'], train_data['y'], train_data['ids'], train_data['mask']
    test_data = np.load(os.path.join(art_dir, 'test' + '.npz'), allow_pickle=True)
    X_test, y_test, test_ids, mask_test = test_data['X_0'], test_data['y'], test_data['ids'], test_data['mask']
    datas['audio'] = ((X_train, y_train, train_ids, mask_train), (X_test, y_test, test_ids, mask_test))
    datas['combined'] = _extract_data(datas)

    # create new_models
    new_video_model = Model(models[1].input, models[1].get_layer('activation').output)
    new_audio_model = Model(models[2].input, models[2].get_layer('activation_1').output)
    inp_layers = [Input((300, 4)), Input((1200, 2))]
    inter_video_model = Model(models[0].get_layer('model').input, models[0].get_layer('model').output)(models[0].input[0])
    inter_audio_model = Model(models[0].get_layer('model_1').input, models[0].get_layer('model_1').output)(models[0].input[1])
    new_comb_model =  Model(models[0].input, [inter_video_model, inter_audio_model])

    to_store = {}
    pdb.set_trace()
    to_store['data_id'] = datas['audio'][0][2][0]
    to_store['true_class'] = datas['audio'][0][1][0]
    to_store['audio_input'] = np.reshape(datas['audio'][0][0][0], (1, 1200, 2))
    to_store['video_input'] = np.reshape(datas['video'][0][0][np.where(datas['video'][0][2] == to_store['data_id'])[0][0]], (1, 300, 4))
    print('video id', np.where(datas['video'][0][2] == to_store['data_id'])[0][0])
    to_store['combined_input'] = [np.reshape(datas['combined'][0][0][0][np.where(datas['combined'][0][2] == to_store['data_id'])[0][0]], (1, 300, 4)), 
    np.reshape(datas['combined'][0][0][1][np.where(datas['combined'][0][2] == to_store['data_id'])[0][0]], (1, 1200, 2))]
    print('combined id', np.where(datas['combined'][0][2] == to_store['data_id'])[0][0])
    to_store['audio_predict'] = new_audio_model.predict(to_store['audio_input'])
    to_store['video_predict'] = new_video_model.predict(to_store['video_input'])
    to_store['combined_predict'] = new_comb_model.predict(to_store['combined_input'])
    to_store['audio_difference'] = (to_store['audio_predict'] - to_store['combined_predict'][1]).sum()
    to_store['video_difference'] = (to_store['video_predict'] - to_store['combined_predict'][0]).sum()

    print(to_store)

