import hudl_server.core_functions as core
from constants import TKN_3
from aiohttp import web
import phonenumbers
import socketio
import json

sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

@sio.event
async def svr_test(sid, testA, atestB, btestC):
    print('testA=', testA)
    print('atestB=', atestB)
    print('btestC=', btestC)
    await sio.emit('reply',' [testA, atestB, btestC]')

@sio.event
async def svr_user_info(sid, uid):
    # sio.emit('reply', core.user_info(uid))
    print('Getting user info')
    await sio.emit('reply', {'data': {'name': 'Quavo Bossman', 'games': ['a', 'b'],
                       'isAdmin': True, 'subscription': 'A'}})

@sio.event
async def svr_clients(sid):
    # clients = core.get_clients()
    clients = [{'name': 'Ryan Cocuzzo', 'games': ['e27e4763-834a-4594-b78f-219d271a0b62'],
                'hudl_email': 'rcocuzzo@u.rochester.edu', 'hudl_pass': 'Pablothepreacher71',
                'id': 'giP0g470mUeEe76VXCwlrCpT1xt2'}]
    print('Found Clients: \n', clients)
    await sio.emit('reply', {'data': clients}) # Note omitting await


@sio.event
async def svr_games(sid, client_id):
    # games = core.get_games(client_id)
    games = [{'created': '2011-10-05T14:48:00.000Z', 'films': [{'name': 'RPI vs. ROCHESTER FGIC 10/05/2019',
                                                        'missing': {'PASS ZONE': 1.0, 'PLAY TYPE': 0.06,
                                                                    'OFF PLAY': 0.34, 'RESULT': 1.0, 'GN/LS': 0.33,
                                                                    'OFF STR': 1.0, 'COVERAGE': 0.59, 'DEF FRONT': 0.58,
                                                                    'PLAY DIR': 1.0, 'BLITZ': 0.93, 'QTR': 1.0,
                                                                    'DN': 0.03, 'OFF FORM': 0.27, 'GAP': 1.0,
                                                                    'YARD LN': 0.04, 'DIST': 0.07, 'HASH': 0.01},
                                                        'quality_evaluations': [
                                                            {'quality': 0.32, 'name': 'pre_align_form'},
                                                            {'name': 'post_align_pt', 'quality': 0.32},
                                                            {'quality': 0.21, 'name': 'post_align_play'}]}, {
                                                           'missing': {'DIST': 0.09, 'HASH': 0.03, 'PASS ZONE': 1.0,
                                                                       'PLAY TYPE': 0.01, 'OFF PLAY': 0.33,
                                                                       'GN/LS': 0.2, 'RESULT': 0.94, 'OFF STR': 1.0,
                                                                       'DEF FRONT': 0.84, 'COVERAGE': 0.72,
                                                                       'BLITZ': 0.97, 'PLAY DIR': 1.0, 'QTR': 1.0,
                                                                       'DN': 0.08, 'OFF FORM': 0.27, 'GAP': 1.0,
                                                                       'YARD LN': 0.07}, 'quality_evaluations': [
            {'name': 'pre_align_form', 'quality': 0.44}, {'name': 'post_align_pt', 'quality': 0.44},
            {'quality': 0.33, 'name': 'post_align_play'}], 'name': 'UR vs. Union College 10/12/2019'}],
      'owner': '7oJZBvgcLZXzu1HRviScQFpalyn2', 'name': 'HUNCHO', 'id': '0079b5ffad324d38bd47d79569c3b2a0'},
     {'created': '2011-10-05T14:48:00.000Z', 'owner': '7oJZBvgcLZXzu1HRviScQFpalyn2', 'films': [{'missing': {'QTR': 1.0,
                                                                                                             'DN': 0.15,
                                                                                                             'OFF FORM': 0.27,
                                                                                                             'GAP': 1.0,
                                                                                                             'YARD LN': 0.2,
                                                                                                             'HASH': 0.33,
                                                                                                             'DIST': 0.15,
                                                                                                             'PASS ZONE': 1.0,
                                                                                                             'PLAY TYPE': 0.39,
                                                                                                             'OFF PLAY': 0.43,
                                                                                                             'GN/LS': 0.36,
                                                                                                             'RESULT': 1.0,
                                                                                                             'COVERAGE': 0.66,
                                                                                                             'DEF FRONT': 0.86,
                                                                                                             'OFF STR': 1.0,
                                                                                                             'BLITZ': 0.91,
                                                                                                             'PLAY DIR': 1.0},
                                                                                                 'quality_evaluations': [
                                                                                                     {'quality': 0.89,
                                                                                                      'name': 'pre_align_form'},
                                                                                                     {'quality': 0.89,
                                                                                                      'name': 'post_align_pt'},
                                                                                                     {'quality': 0.45,
                                                                                                      'name': 'post_align_play'}],
                                                                                                 'name': 'UR vs. Hobart College 11/16/2019'},
                                                                                                {'missing': {'DN': 0.57,
                                                                                                             'OFF FORM': 0.66,
                                                                                                             'GAP': 1.0,
                                                                                                             'YARD LN': 0.57,
                                                                                                             'HASH': 0.56,
                                                                                                             'DIST': 0.57,
                                                                                                             'PASS ZONE': 1.0,
                                                                                                             'PLAY TYPE': 0.52,
                                                                                                             'OFF PLAY': 0.67,
                                                                                                             'RESULT': 1.0,
                                                                                                             'GN/LS': 0.61,
                                                                                                             'OFF STR': 1.0,
                                                                                                             'COVERAGE': 0.79,
                                                                                                             'DEF FRONT': 0.82,
                                                                                                             'BLITZ': 0.96,
                                                                                                             'PLAY DIR': 1.0,
                                                                                                             'QTR': 1.0},
                                                                                                 'quality_evaluations': [
                                                                                                     {
                                                                                                         'name': 'pre_align_form',
                                                                                                         'quality': 0.73},
                                                                                                     {
                                                                                                         'name': 'post_align_pt',
                                                                                                         'quality': 0.73},
                                                                                                     {'quality': 0.42,
                                                                                                      'name': 'post_align_play'}],
                                                                                                 'name': 'UR vs AU 09/28/2019 FGIC'}],
      'name': 'Some Game title', 'id': '14403d510da9496182ef4b5019cd1c5f'}, {'created': '2011-10-05T14:48:00.000Z',
                                                                             'films': [{'missing': {'OFF STR': 1.0,
                                                                                                    'COVERAGE': 0.61,
                                                                                                    'DEF FRONT': 0.81,
                                                                                                    'PLAY DIR': 1.0,
                                                                                                    'BLITZ': 0.93,
                                                                                                    'QTR': 1.0,
                                                                                                    'DN': 0.07,
                                                                                                    'OFF FORM': 0.23,
                                                                                                    'GAP': 1.0,
                                                                                                    'DIST': 0.07,
                                                                                                    'PASS ZONE': 1.0,
                                                                                                    'PLAY TYPE': 0.02,
                                                                                                    'OFF PLAY': 0.38,
                                                                                                    'RESULT': 1.0,
                                                                                                    'GN/LS': 0.18},
                                                                                        'quality_evaluations': [
                                                                                            {'name': 'pre_align_form',
                                                                                             'quality': 0.9},
                                                                                            {'name': 'post_align_pt',
                                                                                             'quality': 0.9},
                                                                                            {'quality': 0.44,
                                                                                             'name': 'post_align_play'}],
                                                                                        'name': 'UR vs SLU FGIC'}, {
                                                                                           'missing': {
                                                                                               'PLAY TYPE': 0.39,
                                                                                               'OFF PLAY': 0.43,
                                                                                               'RESULT': 1.0,
                                                                                               'GN/LS': 0.36,
                                                                                               'OFF STR': 1.0,
                                                                                               'COVERAGE': 0.66,
                                                                                               'DEF FRONT': 0.86,
                                                                                               'BLITZ': 0.91,
                                                                                               'PLAY DIR': 1.0,
                                                                                               'QTR': 1.0, 'DN': 0.15,
                                                                                               'OFF FORM': 0.27,
                                                                                               'GAP': 1.0,
                                                                                               'YARD LN': 0.2,
                                                                                               'HASH': 0.33,
                                                                                               'DIST': 0.15,
                                                                                               'PASS ZONE': 1.0},
                                                                                           'quality_evaluations': [
                                                                                               {'quality': 0.89,
                                                                                                'name': 'pre_align_form'},
                                                                                               {'quality': 0.89,
                                                                                                'name': 'post_align_pt'},
                                                                                               {
                                                                                                   'name': 'post_align_play',
                                                                                                   'quality': 0.45}],
                                                                                           'name': 'UR vs. Hobart College 11/16/2019'},
                                                                                       {'missing': {'OFF PLAY': 0.49,
                                                                                                    'RESULT': 1.0,
                                                                                                    'GN/LS': 0.14,
                                                                                                    'OFF STR': 1.0,
                                                                                                    'COVERAGE': 0.56,
                                                                                                    'DEF FRONT': 0.58,
                                                                                                    'PLAY DIR': 1.0,
                                                                                                    'BLITZ': 1.0,
                                                                                                    'QTR': 1.0,
                                                                                                    'OFF FORM': 0.39,
                                                                                                    'GAP': 1.0,
                                                                                                    'PASS ZONE': 1.0,
                                                                                                    'PLAY TYPE': 0.18},
                                                                                        'quality_evaluations': [
                                                                                            {'quality': 0.8,
                                                                                             'name': 'pre_align_form'},
                                                                                            {'quality': 0.8,
                                                                                             'name': 'post_align_pt'},
                                                                                            {'name': 'post_align_play',
                                                                                             'quality': 0.41}],
                                                                                        'name': 'UR vs. ASC FGIC 09/21/2019 (Fixed)'}],
                                                                             'owner': '7oJZBvgcLZXzu1HRviScQFpalyn2',
                                                                             'name': 'Game game game', 'dictionary': {
            'OFF_FORM': ['BIG', 'TRIO', 'TRIPS', 'DAGGER', 'TREY', 'DUCK', 'DEUCE', 'EMPTY', 'DOUBLES', 'TRUCK', 'BELL',
                         'BANG', 'BOX', 'DUDE', 'TRAP', 'EXTRA', 'DAB', 'TRASH', 'DUO', 'ROLL'],
            'PLAY_TYPE': ['run', 'pass'],
            'OFF_PLAY': ['NY', 'GREEN BAY', 'WASHINGTON', 'HOUSTON', 'NASA', 'ATLANTA', 'AKRON', 'JOKER', 'DENVER',
                         'LACES', 'DETROIT', 'MINNESOTA', 'NEW ENGLAND', 'KC']}, 'training_info': {'postalignplay': {
            'categorical_accuracy': [0.1666666716337204, 0.4615384638309479, 0.25, 0.3333333432674408,
                                     0.5833333134651184], 'epoch': [20.0, 40.0, 60.0, 80.0, 100.0]}, 'postalignpt': {
            'categorical_accuracy': [0.5833333134651184, 0.5600000023841858, 0.5416666865348816, 0.7200000286102295,
                                     0.5], 'epoch': [20.0, 40.0, 60.0, 80.0, 100.0]}, 'prealignform': {
            'accuracy': [0.0, 0.0, 0.0, 0.0, 0.0],
            'categorical_accuracy': [0.0833333358168602, 0.1599999964237213, 0.0416666679084301, 0.07999999821186066,
                                     0.1666666716337204], 'epoch': [20.0, 40.0, 60.0, 80.0, 100.0]}},
                                                                             'id': '2ceaf5db41b740108cde15a6c6f36d33'}]

    print('Serving Games for uid (', client_id, ')..  :\n', games)
    await sio.emit('games', {'data': games})

@sio.event
async def svr_new_client(sid, email, password, name, phone, hudl_email, hudl_pass):

    if not (email and password and name and phone and hudl_email and hudl_pass):
        await sio.emit('exception', json.dumps({"error": 'Bad Information for new client (null)'}))

    if len(phone) == 10:
        phone = phonenumbers.parse(phone, "US")
    else:
        try:
            phone = phonenumbers.parse(phone, None)
        except:
            await sio.emit('exception', json.dumps({"error": 'Bad Information for new client (phone)'}))

    phone_number = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.E164)

    client = core.new_client(email, password, name, phone_number, hudl_email, hudl_pass)

    if client is not None:
        print(client)
        await sio.emit('reply', {'uid': client.uid, 'display_name': client.display_name,
                'phone_number': client.phone_number, email: client.email})
    else:
        await sio.emit('exception', json.dumps({"error": 'Unknown submission error'}))

async def emit(pct, msg):
    await sio.emit('model_status', {'pct': pct, 'msg': msg})

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

    # Check valid
    if not (name and names and data and headers and client_id):
        print('\tFailed!')
        await sio.emit('exception', 'invalid input')

    datum = data.split(TKN_3)
    names = names.split(',')
    headers = headers.split(',')

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

if __name__ == '__main__':
    web.run_app(app)







# async def index(request):
#     """Serve the client-side application."""
#     with open('index.html') as f:
#         return web.Response(text=f.read(), content_type='text/html')
#
# app.router.add_get('/', index)

