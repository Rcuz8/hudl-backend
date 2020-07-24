import hudl_server.core_functions as core
from unittest import TestCase as Test
from src.main.util.io import info, ok

def updated(pct, msg):
    note = '\nReceived Update Notification (' + str(round(pct))+'%' + ') ' + msg
    ok(note)

class Tester(Test):

    def test_generate_model(self):
        game_id = '2ceaf5db-41b7-4010-8cde-15a6c6f36d33'
        nfilms = 3
        try: # Local
            import json
            with open('../../../../hudl_server/dbdata.json') as f:
                data = json.load(f)
                core.generate_model(game_id, nfilms-1,updated, data_override=data, nodeploy=True)
                self.assertTrue(True)
        except: # Unit
            import json
            with open('hudl_server/dbdata.json') as f:
                data = json.load(f)
                core.generate_model(game_id, nfilms - 1, updated, data_override=data, nodeploy=True)
                self.assertTrue(True)


