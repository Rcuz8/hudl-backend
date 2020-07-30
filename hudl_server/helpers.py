import numpy as np
from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
from src.main.core.ai.utils.model.EZModel import EZModel as network
from firebase_admin import ml
import tensorflowjs as tfjs
from constants import data_headers_transformed, data_columns_KEEP, model_gen_configs
from src.main.core.ai.utils.data.hn import hx
from src.main.util.io import info, warn, ok
import keras
from hudl_server.Cloud import bucket



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


def partial_update(params, pct, msg=''):
    """Implements a partial update of a parent process, from a child process.

    Receives the parameters of its parent update process as well as the percentage completion
    of the current child process.

    Updates the parent process.

    Parameters
    ----------
    params : 3-Tuple
        Should contain the following:
            - Parent Update function
            - Parent process start status
            - Parent process end status)
    pct : float
        Percentage completion of the current child process.
    msg : str, optional
        Message to go with the update

    """
    update_fn, update_from, update_to = params
    dist = update_to - update_from
    gain = dist * pct
    end = update_from + round(gain, 2)
    update_fn(end, msg)


async def bldr_make(config, train, test, update_params, update_msg):
    if not isinstance(train, list):
        train = [train]
    if not isinstance(test, list):
        test = [test]

    handle_update = lambda pct: partial_update(update_params, pct, update_msg)

    data_bldr = bldr(config.io.inputs.eval(), config.io.outputs.eval()) \
        .with_type('raw') \
        .with_iterating_adjuster(hx) \
        .with_heads(data_headers_transformed) \
        .with_protected_columns(data_columns_KEEP) \
        .and_train(train) \
        .and_eval(test) \
        .prepare(impute=False)

    nn = network(data_bldr) \
        .build(custom=True, custom_layers=config.keras.dimensions.eval(),
               optimizer=config.keras.learn_params.eval()[0],forceSequential=True) \
        .train(config.keras.learn_params.eval()[1], batch_size=5, on_update=handle_update)

    return nn


def deploy_model(keras_model : keras.Model, game_id, model_name='', nodeploy=False):

    if not nodeploy:
        tfjs.converters.save_keras_model(keras_model, model_name)
        # bucket().adjust_paths(dir_path=model_name, model_id=game_id, model_name=model_name)
        bucket().upload_model(model_name, model_name, game_id)  # async ? if model hasnt saved could be issue

        # # Keras -> firebase-compatible
        #
        # source = ml.TFLiteGCSModelSource.from_keras_model(keras_model)
        # # Create the model object
        # tflite_format = ml.TFLiteFormat(source)
        # model = ml.Model(
        #     display_name=game_id,  # This is the name you use from your app to load the model.
        #     tags=[game_id],  # Optional tags for easier management.
        #     model_format=tflite_format)
        #
        # # Add the model to your Firebase project and publish it
        # new_model = ml.create_model(model)
        # ml.publish_model(new_model.model_id)
    else:
        '''print('Model = ', model_name)
        print('- Layer Weights -')
        for layer in keras_model.layers:
            print('Layer: ', layer.name, '\nConfig: ', layer.get_config(), '\nWeights: ', layer.get_weights())'''
        # tfjs.converters.save_keras_model(keras_model, model_name)
        # bucket().upload_model(model_name, model_name, game_id) # async ? if model hasnt saved could be issue

        # keras_model.save(model_name, save_format='tf')  # creates a TF file


async def tri_build(train, test, on_update, status_start, status_end):
    interval = (status_end - status_start) / 3
    intervals = [status_start,  status_start + (1 * interval),
                 status_start + (2 * interval),  status_start + (3 * interval)]
    print('tri_build() should notify at intervals: ', intervals)
    on_update(status_start, 'Building first Model..')
    # Build the models
    model_1 = await bldr_make(model_gen_configs.pre_align_form, train, test,
                        [on_update, intervals[0], intervals[1]],
                        'Building first Model..')
    model_2 = await bldr_make(model_gen_configs.post_align_pt, train, test,
                        [on_update, intervals[1], intervals[2]],
                        'Building second Model.')
    model_3 = await bldr_make(model_gen_configs.post_align_play, train, test,
                        [on_update, intervals[2], intervals[3]],
                        'Building third Model.')

    return model_1, model_2, model_3

