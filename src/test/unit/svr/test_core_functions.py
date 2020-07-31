import hudl_server.core_functions as core
from unittest import TestCase as Test
from src.main.util.io import info, ok
import asyncio
def updated(pct, msg):
    note = '\nReceived Update Notification (' + str(round(pct))+'%' + ') ' + msg
    ok(note)



class Tester(Test):

    def test_generate_model(self):
        # game_id = 'id1243onjun29'
        game_id = '2ceaf5db-41b7-4010-8cde-15a6c6f36d33'
        nfilms = 3
        try:  # Local
            import json
            with open('../../../../hudl_server/dbdata.json') as f:
                data = json.load(f)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    core.generate_model(game_id, nfilms - 1, updated, data_override=data, nodeploy=False))

                print('Done')
        except:  # Unit
            import json
            with open('hudl_server/dbdata.json') as f:
                data = json.load(f)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    core.generate_model(game_id, nfilms - 1, updated, data_override=data, nodeploy=True))
                print('Done')


# async def test_generate_model():
#     game_id = 'id1243onjun29'
#     # game_id = '2ceaf5db-41b7-4010-8cde-15a6c6f36d33'
#     nfilms = 3
#     try: # Local
#         import json
#         with open('../../../../hudl_server/dbdata.json') as f:
#             data = json.load(f)
#             await core.generate_model(game_id, nfilms-1,updated, data_override=data, nodeploy=False)
#     except: # Unit
#         import json
#         with open('hudl_server/dbdata.json') as f:
#             data = json.load(f)
#             await core.generate_model(game_id, nfilms - 1, updated, data_override=data, nodeploy=False)


# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
# result = loop.run_until_complete(test_generate_model())
# print('Done')







#
# import tensorflowjs as tfjs
# model = tfjs.converters.load_keras_model('postalignplay/model.json')
# print(model.summary())
# for layer in model.layers:
#     print('Layer: ', layer.name)
#     print('Weights: ', layer.weights)

