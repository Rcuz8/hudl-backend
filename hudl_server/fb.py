from constants import relevent_data_columns_configurations as column_configs, QuickParams, \
    model_gen_configs
from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
from firebase_admin import firestore, initialize_app, auth, get_app, credentials
cred = credentials.Certificate('hudl_server/fbpk.json')
initialize_app(cred, {
    'storageBucket': 'hudpred.appspot.com'
})
import hudl_server.helpers as helpers
from src.main.core.ai.utils.data.hn import hx
from uuid import uuid4 as gen_id

qp = QuickParams()

app = get_app()


def user_info(uid):
    db = firestore.client()
    doc = db.collection('users').document(uid).get().to_dict()
    info = doc['info']
    return info

def new_client(email, password, name, phone_number, hudl_email, hudl_pass):
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
            .set({'name': client.display_name, 'hudl_email': hudl_email, 'hudl_pass': hudl_pass})
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

def qa(name, names, datum, headers, client_id):

    analysis = bldr.empty() \
        .of_type('string') \
        .inject_headers(headers) \
        .inject_filenames(names) \
        .declare_relevent_columns(column_configs) \
        .eval_bulk(datum) \
        .analyze_data_quality(hx)

    print('Completed Data Quality Analysis.\n')
    print(analysis)
    db = firestore.client()
    new_game_id = str(gen_id())

    for item in analysis:
        item['data'] = helpers.matrix_to_fb_data(item['data'])

    analysis_info_sections = [{'name': item['name'], 'missing': item['missing'],
                               'quality_evaluations': item['quality_evaluations']}
                              for item in analysis]
    analysis_data_sections = [{'data': item['data']} for item in analysis]

    # Document the new analysis

    # Create game info object
    db.collection('games_info').document(new_game_id).set({'name': name, 'owner': client_id,
                                                           'created': firestore.SERVER_TIMESTAMP,
                                                           'films': analysis_info_sections})

    db.collection('games_data').document(new_game_id) \
        .set({'data': analysis_data_sections})

    # Done
    return new_game_id

def generate_model(game_id, test_film_index):
    db = firestore.client()

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
    data = db.collection('games_data').document(game_id).get().to_dict()  # Query
    data = data['data']  # Unbox
    data = [item['data'] for item in data]  # Unbox
    data = helpers.fb_data_to_matrix(data)  # Unbox (data itself)

    print('Successfully unboxed firestore game data.')

    # Now let's split train/test
    test = data.pop(test_film_index)
    train = data

    # Compile the data      ===> TODO: Per-file h(n) for their vs. our film

    # Build the models
    model_1 = helpers.bldr_make(model_gen_configs.pre_align_form, train, test)
    model_2 = helpers.bldr_make(model_gen_configs.post_align_pt, train, test)
    model_3 = helpers.bldr_make(model_gen_configs.post_align_play, train, test)

    print('Successfully built all models.')

    game_id = ''.join(game_id.split('-'))
    # Deploy the models
    helpers.deploy_model(model_1.get_keras_model(), game_id, 'prealignform')
    helpers.deploy_model(model_2.get_keras_model(), game_id, 'postalignpt')
    helpers.deploy_model(model_3.get_keras_model(), game_id, 'postalignplay')

    print('Successfully deployed all models.')
    return 'OK'



