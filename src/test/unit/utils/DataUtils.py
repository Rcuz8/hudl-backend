import src.main.core.ai.utils.data.Utils as DataUtils
import pandas as pd
import numpy as np
import unittest


class Aux:
    def __init__(self):
        self.instance = None


aux = Aux()

tr = pd.DataFrame(
    [[1, 2, np.nan, 4],
     [4, np.nan, 'hello', 7],
     [8, 9, 'huncho', np.nan],
     ], columns=['A', 'B', 'C', 'D'])

in_params = [('A', 'Col A', 'int')]
out_params = [('B', 'Col B', 'int'), ('C', 'Col C', 'one-hot')]

class Tester(unittest.TestCase):

    def test_missing_data_splits(self):
        # *array of splits* [  [[inp_params],[output_param]], [[inp_params],[output_param]],... ]
        splits = DataUtils.missing_data_splits(tr, in_params, out_params)
        self.assertEqual(len(splits), 2, 'There are 2 in/out split sets for 2 missing-data columns')
        
        ins, outs = splits[0]
        self.assertEqual(len(ins), 2, 'First split has 2 inputs')
        self.assertEqual(len(outs), 1, 'First split has 1 output')

        ins, outs = splits[1]
        self.assertEqual(len(ins), 2, '2nd split has 2 inputs')
        self.assertEqual(len(outs), 1, '2nd split has 1 output')
