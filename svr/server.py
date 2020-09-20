
import svr.core_functions as core
from constants import TKN_3
from aiohttp import web
from src.main.util.io import set_log_level
import phonenumbers
import socketio
import asyncio
import json
import nest_asyncio
from os import environ as env

PORT = env.get("PORT") or 8080
print("Received PORT", env.get("PORT"), "Will run on port", PORT)
nest_asyncio.apply()
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

set_log_level(0)

@sio.event
async def svr_test(sid, testA, atestB, btestC):
    print('testA=', testA)
    print('atestB=', atestB)
    print('btestC=', btestC)
    sio.emit('reply',' [testA, atestB, btestC]')

@sio.event
async def svr_user_info(sid, uid):
    info = core.user_info(uid)
    print('Getting user info for uid',uid, 'Response=', info)
    await sio.emit('reply', info)

@sio.event
async def svr_clients(sid):
    clients = core.get_clients()
    print('Found Clients: \n', clients)
    await sio.emit('reply', {'data': clients})


@sio.event
async def svr_games(sid, client_id):
    games = core.get_games(client_id)
    print('Serving Games for uid (', client_id, ')..  :\n', games)
    await sio.emit('games', {'data': games})

@sio.event
async def svr_new_client(sid, email, password, name, phone, hudl_email, hudl_pass, team_name):

    if not (email and password and name and phone and hudl_email and hudl_pass and team_name):
        await sio.emit('exception', json.dumps({"error": 'Bad Information for new client (null)'}))

    if len(phone) == 10:
        phone = phonenumbers.parse(phone, "US")
    else:
        try:
            phone = phonenumbers.parse(phone, None)
        except:
            await sio.emit('exception', json.dumps({"error": 'Bad Information for new client (phone)'}))

    phone_number = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.E164)

    client = core.new_client(email, password, name, phone_number, hudl_email, hudl_pass, team_name)

    if client is not None:
        print(client)
        await sio.emit('reply', {'uid': client.uid, 'display_name': client.display_name,
                'phone_number': client.phone_number, email: client.email})
    else:
        await sio.emit('exception', json.dumps({"error": 'Unknown submission error'}))


async def emit(pct, msg):
    stat = {'pct': pct, 'msg': msg}
    print('Should emit: ', stat)
    await sio.emit('model_status', stat)
    await sio.sleep(0.5)



@sio.event
async def svr_gen_model(sid, game_id, test_film_index):
    print('New Model Creation Request:')

    try:
        test_film_index = int(test_film_index)
    except:
        await sio.emit('exception', 'invalid film index')

    print('\tGame ID            :', game_id)
    print('\tIndex of test film :', test_film_index, '(', type(test_film_index), ')')

    await core.generate_model(game_id, test_film_index, emit)


@sio.event
async def svr_perform_qa(sid, name, names, data, headers, client_id):
    print('New QA Analysis Request:')
    print('\tName:       ', type(name), name)
    print('\tFilm names: ', type(names), names)
    print('\tData:       ', type(data), data)
    print('\tHeaders:    ', type(headers), headers)
    print('\tClient ID:  ', type(client_id), client_id)

    # Check valid
    if not (name and names and data and headers and client_id):
        print('\tFailed!')
        await sio.emit('exception', 'invalid input')

    datum = data.split(TKN_3)

    await sio.emit('reply', core.qa(name, names, datum, headers, client_id))



@sio.event
async def connect(sid, environ):
    print("connect ", sid)

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)

@sio.event
async def chat_message(sid, data):
    print("message ", data)
    await sio.emit('reply', {'hi':'there'})


def start():
    web.run_app(app, port=PORT)
    print("Received PORT", env.get("PORT"), "Will run on port", PORT)


if __name__ == '__main__':
    start()










# async def index(request):
#     """Serve the client-side application."""
#     with open('index.html') as f:
#         return web.Response(text=f.read(), content_type='text/html')
#
# app.router.add_get('/', index)







# from flask import Flask, request, flash, json
# from flask_cors import CORS
# from flask_socketio import SocketIO
# from svr.core_functions import user_info, new_client, get_clients, qa, get_games, generate_model
# from constants import TKN_3
# import phonenumbers
# import json
#
# app = Flask(__name__)
#
# socketio = SocketIO(app)
#
# if __name__ == '__main__':
#     app.run(threaded=False, processes=2)
#
#
# @socketio.on('message')
# def handle_message(message):
#     print('received message: ' + message)
#
#
# # @app.route('/')
# # def hello_world():
# #     # tell_huncho()
# #     return 'Hello, World!'
#
#
# @app.route('/userinfo')
# def svr_user_info():
#     uid = request.args.get('uid')
#     # return user_info(uid)
#     return {'name': 'Quavo Bossman', 'games': ['a', 'b'], 'isAdmin': True, 'subscription': 'A'}
#
#
# @app.route('/clients')
# def svr_clients():
#     # clients = get_clients()
#     clients = [{'name': 'Ryan Cocuzzo', 'games': ['e27e4763-834a-4594-b78f-219d271a0b62'],
#                 'hudl_email': 'rcocuzzo@u.rochester.edu', 'hudl_pass': 'Pablothepreacher71',
#                 'id': 'giP0g470mUeEe76VXCwlrCpT1xt2'}]
#     print('Found Clients: \n', clients)
#     return {'data': clients}
#
#
# @app.route('/games')
# def svr_games():
#     uid = request.args.get('client_id')
#     games = get_games(uid)
#     print('Serving Games for uid (', uid, ')..  :\n', games)
#     return {'data': games}
#
#
# @app.route('/newclient')
# def svr_new_client():
#     email = request.args.get('email')
#     password = request.args.get('password')
#     name = request.args.get('name')
#     phone_number = request.args.get('phone')
#     hudl_email = request.args.get('hudl_email')
#     hudl_pass = request.args.get('hudl_pass')
#     if not (email and password and name and phone_number and hudl_email and hudl_pass):
#         return json.dumps({"error": 'Bad Information for new client (null)'}), 500
#
#     if len(phone_number) == 10:
#         phone_number = phonenumbers.parse(phone_number, "US")
#     else:
#         try:
#             phone_number = phonenumbers.parse(phone_number, None)
#         except:
#             return json.dumps({"error": 'Bad Information for new client (phone)'}), 500
#
#     phone_number = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.E164)
#
#     client = new_client(email, password, name, phone_number, hudl_email, hudl_pass)
#
#     if client is not None:
#         print(client)
#         return {'uid': client.uid, 'display_name': client.display_name,
#                 'phone_number': client.phone_number, email: client.email}
#     else:
#         return json.dumps({"error": 'Unknown submission error'}), 500
#
#
# @app.route('/new_model', methods=['GET', 'POST'])
# def svr_gen_model():
#     if request.method == 'POST':
#         print('New Model Creation Request:')
#
#         game_id = request.form['game_id']
#         test_film_index = request.form['test_film_index']
#
#         try:
#             test_film_index = int(test_film_index)
#         except:
#             return 'invalid film index', 405
#
#         print('\tGame ID            :', game_id)
#         print('\tIndex of test film :', test_film_index, '(', type(test_film_index), ')')
#
#         return generate_model(game_id, test_film_index)
#
#
# @app.route('/new_qa', methods=['GET', 'POST'])
# def svr_perform_qa():
#     if request.method == 'POST':
#         print('New QA Analysis Request:')
#
#         name = request.form['name']
#         names = request.form['names']
#         datum_ = request.form['data']
#         headers = request.form['headers']
#         client_id = request.form['client_id']
#
#         # Check valid
#         if not (name and names and datum_ and headers and client_id):
#             print('\tFailed!')
#             return 'invalid input'
#
#         # print(datum_)
#         datum = datum_.split(TKN_3)
#         names = names.split(',')
#         headers = headers.split(',')
#
#         return qa(name, names, datum, headers, client_id)
#
#
# # def filedata(agg_data_str):
# #     '''
# #         Parses string matrix data
# #
# #         Ex. "[['1,2,3','4,5,6'], ['1,2,3','4,5,6']]" (str)
# #         represents:
# #
# #             ['1,2,3','4,5,6']
# #             ['1,2,3','4,5,6']
# #
# #         This converts to:
# #             [['1,2,3', '4,5,6'], ['1,2,3', '4,5,6']] (matrix)
# #
# #     '''
# #
# #     # break out of enclosing brackets
# #     st = agg_data_str.index('[') + 1
# #     end = agg_data_str.rindex(']')
# #     agg_data_str = agg_data_str[st:end]
# #     full = []
# #
# #     while (len(agg_data_str) > 0):
# #
# #         try:
# #             st = agg_data_str.index('[') + 1
# #             end = agg_data_str.index(']')
# #             ds = agg_data_str[st:end]
# #         except:
# #             break
# #
# #         try:
# #             if (ds.index("'") > -1):
# #                 full.append([item.split(',') for item in ds.split("'")[1::2]])
# #         except:
# #             full.append([item.split(',') for item in ds.split('"')[1::2]])
# #
# #         agg_data_str = agg_data_str[end + 1:]
# #
# #     return full
