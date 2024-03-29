# Firebase Init
from svr.Cloud import LOCAL
from firebase_admin import firestore, auth, get_app
from constants import relevent_data_columns_configurations as column_configs, QuickParams
from src.main.ai.data.Builder import huncho_data_bldr as bldr
import svr.helpers as helpers
from src.main.ai.data.hn import hx, odk_filter
from uuid import uuid4 as gen_id
from src.main.util.io import ok, warn
import json
import datetime
from tensorflow import keras

qp = QuickParams()

app = get_app()


def user_info(uid):
    db = firestore.client()
    try:
        doc = db.collection('clients').document(uid).get().to_dict()
        return doc
    except:
        warn('User Info document doesnt exist.')
        return {}


def new_client(email, password, name, phone_number, hudl_email, hudl_pass, team_name):
    print('Creating new client with: ')
    print('\tEmail: ', email)
    print('\tPass:  ', password)
    print('\tName:  ', name)
    print('\tphone: ', phone_number)
    print('\t----- HUDL Credentials -----: ')
    print('\tHUDL Email: ', hudl_email)
    print('\tHUDL Pass : ', hudl_pass)
    print('\t-------------------------- -: ')

    try:
        client = auth.Client(app=app).create_user(email=email, password=password,
                                                  display_name=name, phone_number=phone_number)
        print('Created new client with uid:', client.uid)
        db = firestore.client()
        db.collection('clients').document(client.uid) \
            .set({'name': client.display_name,
                  'hudl_email': hudl_email,
                  'hudl_pass': hudl_pass,
                  'team_name': team_name})
        return client

    except:
        return None


def get_clients():
    result = firestore.client().collection('clients').get()

    clients = []

    for client in result:
        cid = client.id
        client = client.to_dict()
        client['id'] = cid
        clients.append(client)

    return clients


def get_games(client_id):
    docs = firestore.client().collection('games_info') \
        .where('owner', '==', client_id) \
        .get()

    docs = [{**(doc.to_dict()), **{'id': doc.id}} for doc in docs]

    return docs


'''def drop_uneccessary(df):
    Dropper.drop_cols(df, ['BLITZ', 'COVERAGE', 'DEF FRONT', 'GAP', 'PLAY DIR', 'PASS ZONE'])
'''


def qa(name, names, datum, headers, client_id, nodeploy=False):
    analysis = bldr.empty() \
        .with_type('string') \
        .with_heads(headers) \
        .with_filenames(names) \
        .with_protected_columns(column_configs) \
        .with_important_columns(column_configs) \
        .and_eval(datum) \
        .analyze_data_quality(hx, evaluation_frame_filter_fn=odk_filter)

    print('Completed Data Quality Analysis.\n')
    print(analysis)
    db = firestore.client()
    new_game_id = str(gen_id()).replace('-', '')

    for item in analysis:
        item['data'] = helpers.matrix_to_fb_data(item['data'])

    analysis_info_sections = [{'name': item['name'], 'missing': item['missing'],
                               'quality_evaluations': item['quality_evaluations']}
                              for item in analysis]
    analysis_data_sections = [{'data': item['data']} for item in analysis]

    # Document the new analysis

    if not nodeploy:

        # Create game info object
        db.collection('games_info').document(new_game_id).set({'name': name, 'owner': client_id,
                                                               'created': datetime.datetime.now().isoformat(),
                                                               'films': analysis_info_sections})

        db.collection('games_data').document(new_game_id) \
            .set({'data': analysis_data_sections})

    # Done
    return new_game_id


async def __update(fn, pct, msg):
    if fn:
        await fn(pct, msg)


async def generate_model(game_id, test_film_index, on_progress=None, data_override=None, nodeploy=False):
    await on_progress(0, 'Collecting Data..')

    # 1. Get the train/test data (query db using the game id)
    '''

    Structure must be unboxed. Looks like this:

    games
        id
            data
                [
                    { data:
                        ... (fb-structured matrix data)
                    },
                    ...


    '''
    if not data_override:
        data = __fetch_game(game_id)
    else:
        data = data_override

    data = unbox(data)

    ok('Successfully unboxed firestore game data.')
    await __update(on_progress, 10, 'Building first Model..')

    # Now let's split train/test
    test = data.pop(test_film_index)
    train = data

    # Compile the data      ===> TODO: Per-file h(n) for their vs. our film
    models = await  helpers.tri_build(train, test, on_progress, 10, 85)
    model_names = ['prealignform', 'postalignpt', 'postalignplay']
    progress_pcts = [90, 95, 100]
    progress_msgs = ['Deploying second Model..', 'Deploying third Model..', 'Done.']

    ok('Successfully built all models.')

    game_id = ''.join(game_id.split('-'))

    # Deploy the models
    for i in range(len(models)):
        helpers.deploy_model(models[i].get_keras_model(), game_id, model_names[i],
                             nodeploy, models[i].dictionary, models[i].training_accuracies())
        await __update(on_progress, progress_pcts[i], progress_msgs[i])

    ok('Successfully deployed all models.')

    return 'OK'


def __fetch_game(gid, save_to=None):
    db = firestore.client()
    game = db.collection('games_data').document(gid).get().to_dict()  # Query
    if save_to:
        __save_json(game, 'svr/dbdata.json')
    return game


def strip_game_hyphens(gid):
    db = firestore.client()
    data = __fetch_game(gid)
    info = db.collection('games_info').document(gid).get().to_dict()  # Query
    newid = ''.join(gid.split('-'))
    db.collection('games_data').document(newid).set(data)
    db.collection('games_info').document(newid).set(info)
    db.collection('games_data').document(gid).delete()
    db.collection('games_info').document(gid).delete()


def __save_json(data: dict, path: str, local_check=True):
    if local_check:
        if LOCAL:
            path = path.replace('svr/', '')
    with open(path, 'w+') as fp:
        json.dump(data, fp)


def mock_model():
    model = keras.models.Sequential(
        [
            keras.layers.Dense(128, input_shape=(10,)),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(20)
        ])
    model.compile(loss='mse')
    return model


def sv():
    model = mock_model()
    # model.fit([[1,2,3]] * 10, [[1,2,3]] * 10, batch_size=2)
    helpers.sv(model, 'prealignform')


def unbox(response):
    data = response['data']  # Unbox
    data = [item['data'] for item in data]  # Unbox
    data = helpers.fb_data_to_matrix(data)  # Unbox (data itself)
    return data

# def fetch_model(id: str):
#     return storage.storag


# fetch_game('2ceaf5db-41b7-4010-8cde-15a6c6f36d33', save_to='svr/dbdata.json')


# analysis = bldr.empty() \
#     .of_type('string') \
#     .inject_headers(headers) \
#     .inject_filenames(names) \
#     .declare_relevent_columns(column_configs) \
#     .eval_bulk(datum) \
#     .analyze_data_quality(hx, evaluation_frame_filter_fn=odk_filter)
