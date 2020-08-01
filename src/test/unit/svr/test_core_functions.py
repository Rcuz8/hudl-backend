import hudl_server.core_functions as core
from unittest import TestCase as Test
from src.main.util.io import info, ok, set_log_level
import asyncio
def updated(pct, msg):
    note = ' (' + str(round(pct))+'%' + ') ' + msg
    ok(note)

# Set Log Level
set_log_level(0)

class Tester(Test):

    # def test_generate_model(self):
    #     game_id = '2ceaf5db41b740108cde15a6c6f36d33'
    #     nfilms = 3
    #
    #     try:  # Local
    #         f = open('../../../../hudl_server/dbdata.json')
    #     except:  # Unit
    #         try:
    #             f = open('hudl_server/dbdata.json')
    #         except:
    #             raise Exception('core test : Could not open test data file!')
    #
    #     import json
    #     data = json.load(f)
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     loop.run_until_complete(
    #         core.generate_model(game_id, nfilms - 1, updated, data_override=data, nodeploy=True))
    #     f.close()
    #     loop.close()
    #
    #     ok('\nDone')


    def test_live_generate_model(self):
        game_id = '2ceaf5db41b740108cde15a6c6f36d33'
        nfilms = 3

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            core.generate_model(game_id, nfilms - 1, updated, data_override=None, nodeploy=False))
        loop.close()

        ok('\nDone')


    # def test_strip_game_hyphens(self):
    #     id = '2ceaf5db-41b7-4010-8cde-15a6c6f36d33'
    #     core.strip_game_hyphens(id)

