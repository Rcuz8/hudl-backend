
from flask import Flask, request, flash,json
from flask_cors import CORS
from hudl_server.fb import user_info,new_client,get_clients, qa, get_games, generate_model
from constants import TKN_3
import phonenumbers
app = Flask(__name__)
CORS(app)
import json

@app.route('/')
def hello_world():
    # tell_huncho()
    return 'Hello, World!'


def filedata(agg_data_str):
    '''
        Parses string matrix data

        Ex. "[['1,2,3','4,5,6'], ['1,2,3','4,5,6']]" (str)
        represents:

            ['1,2,3','4,5,6']
            ['1,2,3','4,5,6']

        This converts to:
            [['1,2,3', '4,5,6'], ['1,2,3', '4,5,6']] (matrix)

    '''

    # break out of enclosing brackets
    st = agg_data_str.index('[') + 1
    end = agg_data_str.rindex(']')
    agg_data_str = agg_data_str[st:end]
    full = []

    while (len(agg_data_str) > 0):

        try:
            st = agg_data_str.index('[') + 1
            end = agg_data_str.index(']')
            ds = agg_data_str[st:end]
        except:
            break

        try:
            if (ds.index("'") > -1):
                full.append([item.split(',') for item in ds.split("'")[1::2]])
        except:
            full.append([item.split(',') for item in ds.split('"')[1::2]])

        agg_data_str = agg_data_str[end + 1:]

    return full

@app.route('/userinfo')
def svr_user_info():
    uid = request.args.get('uid')
    # return user_info(uid)
    return {'name': 'Quavo Bossman', 'games': ['a', 'b'], 'isAdmin': True, 'subscription': 'A'}

@app.route('/clients')
def svr_clients():
    # clients = get_clients()
    clients = [{'name': 'Ryan Cocuzzo', 'games': ['e27e4763-834a-4594-b78f-219d271a0b62'], 'hudl_email': 'rcocuzzo@u.rochester.edu', 'hudl_pass': 'Pablothepreacher71', 'id': 'giP0g470mUeEe76VXCwlrCpT1xt2'}]
    print('Found Clients: \n', clients)
    return {'data':clients}

@app.route('/games')
def svr_games():
    uid = request.args.get('client_id')
    games = get_games(uid)
    print('Serving Games for uid (', uid, ')..  :\n', games)
    return {'data': games}


@app.route('/newclient')
def svr_new_client():
    email = request.args.get('email')
    password = request.args.get('password')
    name = request.args.get('name')
    phone_number = request.args.get('phone')
    hudl_email = request.args.get('hudl_email')
    hudl_pass = request.args.get('hudl_pass')
    if not (email and password and name and phone_number and hudl_email and hudl_pass):
        return json.dumps({"error": 'Bad Information for new client (null)'}), 500

    if len(phone_number) == 10:
        phone_number = phonenumbers.parse(phone_number, "US")
    else:
        try:
            phone_number = phonenumbers.parse(phone_number, None)
        except:
            return json.dumps({"error": 'Bad Information for new client (phone)'}), 500

    phone_number = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.E164)

    client = new_client(email, password, name, phone_number,hudl_email, hudl_pass)

    if client is not None:
        print(client)
        return {'uid': client.uid, 'display_name': client.display_name,
                'phone_number': client.phone_number, email: client.email }
    else:
        return json.dumps({"error": 'Unknown submission error'}), 500

@app.route('/new_model', methods=['GET', 'POST'])
def svr_gen_model():

    if request.method == 'POST':
        print('New Model Creation Request:')

        game_id = request.form['game_id']
        test_film_index = request.form['test_film_index']

        try:
            test_film_index = int(test_film_index)
        except:
            return 'invalid film index', 405

        print('\tGame ID            :', game_id)
        print('\tIndex of test film :', test_film_index, '(', type(test_film_index), ')')

        return generate_model(game_id, test_film_index)




@app.route('/new_qa', methods=['GET', 'POST'])
def svr_perform_qa():

    if request.method == 'POST':
        print('New QA Analysis Request:')

        name = request.form['name']
        names = request.form['names']
        datum_ = request.form['data']
        headers = request.form['headers']
        client_id = request.form['client_id']

        # Check valid
        if not (name and names and datum_ and headers and client_id):
            print('\tFailed!')
            return 'invalid input'

        # print(datum_)
        datum = datum_.split(TKN_3)
        names = names.split(',')
        headers = headers.split(',')

        return qa(name, names, datum, headers, client_id)













