import svr.helpers as fn
import svr.core_functions as core
from unittest import TestCase as Test
from constants import sample_data
from src.main.util.io import info, ok, set_log_level
import asyncio
''' data = sample_data * 5 '''


async def updated(pct, msg):
    note = 'Received Update Notification (' + str(round(pct))+'%' + ') ' + msg
    ok(note)

def get_col(arr, col):
    return [x[col] for x in arr]

def freqs(arr):
    import numpy as np

    x = np.array(arr)
    unique, counts = np.unique(x, return_counts=True)

    temp = np.array((unique, counts)).T
    temp = sorted(temp, key=lambda tup: int(tup[1]))
    temp = [list(a) for a in temp]
    temp = [(a,int(b)) for a,b in temp]
    temp.reverse()
    return temp

class Tester(Test):

    def test_tri_build(self):
        set_log_level(0)
        # Load Data
        f = open('../../../svr/dbdata.json')
        import json
        data = json.load(f)
        data = core.unbox(data)
        f.close()
        flatten = lambda l: [item for sublist in l for item in sublist]
        combined = flatten(data)
        fr = freqs(get_col(combined, 9))
        print(len(combined))
        for item in fr:
            print(item[0].ljust(26, ' '), ':', round(round(item[1] / len(combined), 3) * 100, 1), '%\t\t',
                  '(',item[1],'/',len(combined),')' )


        # Async Handling & Execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(fn.tri_build(data, data, updated, 1, 99, dual_build=False, skip=[2,3]))
        self.assertEqual(len(models), 3, 'Builds all models.')

        # Built, now handle..

        predictors = [model.build_predictor().predictor for model in models if model is not None]
        prediction_ins = models[0].data()[0] # Training X Data
        prediction_outs = models[0].data()[1] # Training Y Data

        print(prediction_ins)
        print(prediction_outs)
        # print('Prediction Data:\n', prediction_data)
        formation_predictions = predictors[0].predict_bulk(prediction_ins, encode=False, waterfall_size=3)
        for i in range(len(formation_predictions)):
            print('#' + str(i), [round(item, 2) for item in prediction_ins[i].tolist()], '=>', formation_predictions[i][0])



