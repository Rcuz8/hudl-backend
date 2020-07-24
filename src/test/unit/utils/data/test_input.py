from src.main.core.ai.utils.data.Input import Input as inp
import unittest as u
import src.test.utils as test
import pandas as pd
import numpy as np


class Aux:
    def __init__(self):
        self.instance = None

aux = Aux()

class Tester(u.TestCase):

    def test_gen_agg_data(self):
        sources = [test.data.copy(), test.data.copy()]
        actual = inp.generate_aggregate_dataframe(
            sources=sources,
            input_params=test.inputs,
            output_params=test.outputs,
            addtl_heuristic_fn=test.double,
            data_format='raw',
            per_file_heuristic_fn=test.filter,
            injected_headers=test.columns,
            relevent_columns_override=test.columns,
            drop_scarce_columns=False
        )

        expected = pd.DataFrame([test.f4, test.f4], columns=test.columns)

        self.assertIsNotNone(actual, 'Didnt generate DataFrame.')
        self.assertGreater(len(actual.index), 0, 'Didnt generate non-empty DataFrame.')

        expected.fillna('', inplace=True)
        actual.fillna('', inplace=True)

        self.assertTrue(expected.equals(actual), 'DataFrames are not the same.')

    def test_model_params(self):
        io = test.IOCombinations.alt
        actual = inp.model_params(io[0], io[1], test.dictionary)
        expected = {
            'inputs': 5,
            'outputs': [('C', 1, 'linear', 'mean_squared_error'), ('D', 1, 'linear', 'mean_squared_error')]
        }

        self.assertEqual(actual, expected, 'Model params are NOT the same.')
