

from aiohttp import web
import socketio
from os import environ as env

PORT = env.get("PORT") or 8080
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

def start():
    web.run_app(app, port=PORT)
    print("Received PORT", env.get("PORT"), "Will run on port", PORT)


if __name__ == '__main__':
    start()

