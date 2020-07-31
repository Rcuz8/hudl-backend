import numpy as np
from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
from src.main.core.ai.utils.model.EZModel import EZModel as network
from firebase_admin import firestore
import tensorflowjs as tfjs
from constants import data_headers_transformed, data_columns_KEEP, model_gen_configs
from src.main.core.ai.utils.data.hn import hx
from src.main.util.io import info, warn, ok
from hudl_server.Cloud import bucket
from pathlib import Path
import shutil
import os
import keras


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


def dual_builders(config, train, test):
    """Returns both compiled configs ( Train + Test ) & ( Train-only ) """
    val1 = bldr(config.io.inputs.eval(), config.io.outputs.eval()) \
            .with_type('raw') \
            .with_iterating_adjuster(hx) \
            .with_heads(data_headers_transformed) \
            .with_protected_columns(data_columns_KEEP) \
            .and_train(train) \
            .and_eval(test) \
            .prepare(impute=False)
    val2 = bldr(config.io.inputs.eval(), config.io.outputs.eval()) \
            .with_type('raw') \
            .with_iterating_adjuster(hx) \
            .with_heads(data_headers_transformed) \
            .with_protected_columns(data_columns_KEEP) \
            .and_train(train) \
            .and_train(test) \
            .prepare(impute=False)
    return val1, val2





async def bldr_make(config, train, test, update_params, update_msg):
    if not isinstance(train, list) or not isinstance(train[0], list) or not isinstance(train[0][0], list):
        train = [train]
    if not isinstance(test, list) or not isinstance(test[0], list) or not isinstance(test[0][0], list):
        test = [test]

    handle_update = lambda pct: partial_update(update_params, pct / 2, update_msg)

    train_test_bldr, train_only_bldr = dual_builders(config, train, test)

    info('Compiling Training/Evaluation Build.')
    training_network = network(train_test_bldr) \
        .build(custom=True, custom_layers=config.keras.dimensions.eval(),
               optimizer=config.keras.learn_params.eval()[0], forceSequential=True) \
        .train(config.keras.learn_params.eval()[1], batch_size=5, on_update=handle_update)

    training_accuracies = training_network.training_accuracies()

    info('Compiling Production Build.')
    production_network = network(train_only_bldr) \
        .build(custom=True, custom_layers=config.keras.dimensions.eval(),
               optimizer=config.keras.learn_params.eval()[0], forceSequential=True) \
        .train(config.keras.learn_params.eval()[1], batch_size=5, on_update=handle_update)

    production_network.set_training_accuracies(training_accuracies)

    return production_network


def __nest_update(new, parent):
    """Nested update for firestore reference.
    see: https://stackoverflow.com/a/63178463/6127225
    """
    return {parent + '.' + new: val for new, val in list(new.items())}


def deploy_model(keras_model: keras.Model, game_id, model_name='', nodeploy=False,
                 withDictionary=None, withTrainingAccuracies=None):
    if not nodeploy:



        dirpath = Path(model_name)
        if dirpath.exists() and dirpath.is_dir():
            info('PATH (',model_name,') exists. Contents: ')
            info(os.listdir(model_name))
            info('Removing..')
            shutil.rmtree(dirpath)
            info('Done.')
        tfjs.converters.save_keras_model(keras_model, model_name)
        # bucket().adjust_paths(dir_path=model_name, model_id=game_id, model_name=model_name)
        bucket().upload_model(model_name, model_name, game_id)  # async ? if model hasnt saved could be issue

        # Upload Dictionary
        info('Uploading the following dictionary: ', withDictionary)

        if withDictionary:
            firestore.client().collection('games_info').document(game_id).update(
                __nest_update(withDictionary, 'dictionary'))
        if withTrainingAccuracies:
            firestore.client().collection('games_info').document(game_id).update(
                __nest_update(withTrainingAccuracies, 'training_info'))

        dirpath = Path(model_name)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)


async def tri_build(train, test, on_update, status_start, status_end):
    interval = (status_end - status_start) / 3
    intervals = [status_start, status_start + (1 * interval),
                 status_start + (2 * interval), status_start + (3 * interval)]
    print('tri_build() should notify at intervals: ', intervals)
    on_update(status_start, 'Building first Model..')
    # Build the models ( 3 x 2 )
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
