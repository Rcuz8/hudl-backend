import numpy as np
from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
from src.main.core.ai.utils.model.EZModel import Modeler as network
from firebase_admin import ml
from constants import data_headers_transformed, data_columns_KEEP
from src.main.core.ai.utils.data.hn import hx


def db_npNans_to_Nones(games):
    '''
    Begin with Games query list -> End with np.nans in data CONVERTED to Nones
    Ex. Begin with: [{'model_compositional_data' :[ {'name': 'UR D vs EC O'
                                        ,'quality_evauations': [{'quality': 0.96, 'name': 'pre_align_form'},...
    '''
    for game in games:
        for film in game['model_compositional_data']:
            for row in film['data']:
                for i in range(len(row['0'])):
                    if row['0'][i] != row['0'][i]:
                        row['0'][i] = None


def db_nones_to_npNans(games):
    '''
    data Nones -> np.nans
    '''
    for game in games:
        for film in game['model_compositional_data']:
            for row in film['data']:
                for i in range(len(row['0'])):
                    if row['0'][i] == None:
                        row['0'][i] = np.nan

# Matrix storage in firestore

def matrix_to_fb_data(matrix):
    return [{'0': row} for row in matrix]


def fb_data_to_matrix(fb_data):
    print('Recieved Firestore data: \n', fb_data)
    return [[row['0'] for row in data] for data in fb_data]


def bldr_make(config, train, test):
    if not isinstance(train, list):
        train = [train]
    if not isinstance(test, list):
        test = [test]
    data_bldr = bldr(config.io.inputs.eval(), config.io.outputs.eval(),
                     thresh=0.1) \
        .of_type('raw') \
        .add_tr_file_hn(hx).add_tst_file_hn(hx)\
        .inject_headers(data_headers_transformed) \
        .dont_remove_columns(data_columns_KEEP)\
        .train_bulk(train) \
        .eval_bulk(test).prepare(impute=False)

    nn = network(data_bldr) \
        .build(custom=True, custom_layers=config.keras.dimensions.eval(),
               optimizer=config.keras.learn_params.eval()[0]) \
        .summarize() \
        .train(config.keras.learn_params.eval()[1], batch_size=5)

    return nn


def deploy_model(keras_model, game_id, model_name=''):
    game_id += model_name
    # Keras -> firebase-compatible
    source = ml.TFLiteGCSModelSource.from_keras_model(keras_model)
    # Create the model object
    tflite_format = ml.TFLiteFormat(source)
    model = ml.Model(
        display_name=game_id,  # This is the name you use from your app to load the model.
        tags=[game_id],  # Optional tags for easier management.
        model_format=tflite_format)

    # Add the model to your Firebase project and publish it
    new_model = ml.create_model(model)
    ml.publish_model(new_model.model_id)
