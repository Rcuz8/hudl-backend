from src.main.core.ai.utils.data.Input import InputChef
from src.main.core.ai.utils.data.Utils import build_dictionary_for
import pandas as pd
import unittest

class Aux:
    def __init__(self):
        self.in_p = None
        self.out_p = None
        self.dictionary = None


aux = Aux()

files = ['../../../../data/nn/hudl-hobart_vs_rpi.json', '../../../../data/nn/hudl-hobart_vs_union.json']

# Define Input & output representations
input_params = [('PLAY_NUM', 'Play Num', 'int'), ('ODK', 'ODK', 'one-hot'), ('DN', 'Down', 'int'),
                  ('DIST', 'Distance', 'int'), ('HASH', 'Hash Line', 'one-hot'), ('YARD_LN', 'Yard Line', 'int')]
output_params = [('PLAY_TYPE', 'Play Type', 'one-hot'), ('OFF_FORM', 'Offensive Formation', 'one-hot'),
                   ('OFF_PLAY', 'Offensive Play', 'one-hot')]

true_test_mp = {
    'inputs': 2,
    'outputs': [ ('B', 1, 'linear', 'mean_squared_error'), ('C', 3, 'softmax', 'categorical_crossentropy')  ]
}

def apply_heuristic(df):
    df.replace('iz', 'izr', inplace=True)
    return df

class Tester(unittest.TestCase):

    def test_gen_agg_data(self):
        df, in_p, out_p = InputChef.generate_aggregate_dataframe_from_json(files,
                            input_params,output_params,apply_heuristic)
        self.assertIsNotNone(df, 'generated dataframe')
        print(df)
        aux.in_p = in_p
        aux.out_p = out_p
        aux.dictionary = build_dictionary_for(df)

    def test_model_params_1(self):
        df = pd.DataFrame(
            [[1, 2, 'hi', 4],
             [4, 5, 'hello', 7],
             [8, 9, 'huncho', 11],
             ], columns=['A', 'B', 'C', 'D'])
        inp = [('A', 'Col A', 'int'), ('D', 'Col D', 'int')]
        outp = [('C', 'Col C', 'one-hot'), ('B', 'Col B', 'int')]
        dictionary = build_dictionary_for(df)
        model_params = InputChef.model_params(inp,outp, dictionary)
        self.assertIsNotNone(model_params, 'Model params exists')
        self.assertEqual(true_test_mp, model_params, 'Model params is correct')

    def test_model_params_2(self):
        model_params = InputChef.model_params(aux.in_p,aux.out_p,aux.dictionary)
        self.assertIsNotNone(model_params, 'Model params exists')
        print(model_params)