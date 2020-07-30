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
    games = core.get_games(client_id)
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

    core.generate_model(game_id, test_film_index, emit)


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

