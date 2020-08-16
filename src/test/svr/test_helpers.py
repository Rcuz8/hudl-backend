import svr.helpers as fn
from unittest import TestCase as Test
from constants import sample_data
from src.main.util.io import info, ok

def updated(pct, msg):
    note = '\nReceived Update Notification (' + str(round(pct))+'%' + ') ' + msg
    ok(note)

data = sample_data * 5

class Tester(Test):

    async def test_tri_build(self):
        models = await fn.tri_build(data, data, updated, 20, 40)
        print(models)
        self.assertEqual(len(models), 3, 'Builds all models.')

