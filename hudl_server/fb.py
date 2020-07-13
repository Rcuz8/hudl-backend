from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
from firebase_admin import firestore, initialize_app, auth, get_app
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "hudl_server/fbpk.json"
import time
from datetime import timedelta
from uuid import uuid4 as gen_id
initialize_app()
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
        db.collection('clients').document(client.uid)\
            .set({'games': [], 'name': client.display_name, 'hudl_email': hudl_email, 'hudl_pass': hudl_pass})
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


def qa(names, datum, headers, client_id):
    analysis = bldr.empty()\
        .inject_headers(headers)\
        .inject_filenames(names)\
        .eval_bulk(datum)\
        .analyze_data_quality()

    print('Completed Data Quality Analysis.\n')
    db = firestore.client()
    new_game_id = str(gen_id())

    # Document the new analysis
    db.collection('games').document(new_game_id)\
        .set(analysis)
    # Add it to client's games
    db.collection('clients').document(client_id)\
        .update({'games': db.ArrayUnion([new_game_id])})

    # Done
    return new_game_id


    print(analysis)







