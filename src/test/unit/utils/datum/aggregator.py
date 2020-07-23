import src.main.core.ai.utils.data.Aggregator as DataHandler
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def sc(x, _min, _max):
    return (x - 1) * (1 / (_max - _min))


c1_min = 1
c1_max = 8
c2_min = 2
c2_max = 9
c4_min = 4
c4_max = 12
c1mm = MinMaxScaler()
c1mm.fit([[c1_min], [c1_max]])
c2mm = MinMaxScaler()
c2mm.fit([[c2_min], [c2_max]])
c4mm = MinMaxScaler()
c4mm.fit([[c4_min], [c4_max]])


def custom_scaler(_min, _max):
    scaler = MinMaxScaler()
    scaler.fit([[_min], [_max]])
    return scaler


def c1_sc(x):
    return c1mm.transform([[x]])[0][0]


def c2_sc(x):
    return c2mm.transform([[x]])[0][0]


def c4_sc(x):
    return c4mm.transform([[x]])[0][0]


tr = pd.DataFrame(
    [[1, 2, 'hi', 4],
     [4, 5, 'hello', 7],
     [8, 9, 'huncho', 11],
     ], columns=['A', 'B', 'C', 'D'])

tst = pd.DataFrame(
    [[3, 2, 'quavo', 7],
     [4, 5, 'big dog', 12],
     [8, 9, 'icon', 11],
     [8, 9, 'huncho', 11],
     ], columns=['A', 'B', 'C', 'D'])

input_params = [('A', 'Col A', 'int'), ('D', 'Col D', 'int')]
output_params = [('C', 'Col C', 'one-hot'), ('B', 'Col B', 'int')]

c_enc = {
    'big dog': np.asarray([1, 0, 0, 0, 0, 0]),
    'hello': np.asarray([0, 1, 0, 0, 0, 0]),
    'hi': np.asarray([0, 0, 1, 0, 0, 0]),
    'huncho': np.asarray([0, 0, 0, 1, 0, 0]),
    'icon': np.asarray([0, 0, 0, 0, 1, 0]),
    'quavo': np.asarray([0, 0, 0, 0, 0, 1])
}

# Note: label encoders as array is a bad idea
exp_test_res = {
    'data': ([[c1_sc(3), c4_sc(7)],
              [c1_sc(4), c4_sc(12)],
              [c1_sc(8), c4_sc(11)],
              [c1_sc(8), c4_sc(11)]],
             {'C': [c_enc['quavo'], c_enc['big dog'], c_enc['icon'], c_enc['huncho']],
              'B': np.asarray([c2_sc(2), c2_sc(5), c2_sc(9), c2_sc(9)])}),
    'label_encoders_null': ({'A': True, 'D': True}, {'B': True, 'C': False}),
# Notice not exact same, but cannot recreate exact dictionary
    'scalers_null': ({'A': False, 'D': False}, {'B': False, 'C': True}),
# Notice not exact same, but cannot recreate exact scalers
    'shapes': (2, 7),  # IMPORTANT, Should be this exactly to reflect ENTIRE dictionary
    'df_index_lists': ([0, 3], [2, 1]),
    'column_names': (['A', 'D'], ['C', 'B'])
}

exp_tr_res = {
    'data': ([[c1_sc(1), c4_sc(4)],
              [c1_sc(4), c4_sc(7)],
              [c1_sc(8), c4_sc(11)]],
             {'C': [c_enc['hi'], c_enc['hello'],
                    c_enc['huncho']], 'B': np.asarray([c2_sc(2), c2_sc(5), c2_sc(9)])}),
    'label_encoders_null': ({'A': True, 'D': True}, {'B': True, 'C': False}),
# Notice not exact same, but cannot recreate exact dictionary
    'scalers_null': ({'A': False, 'D': False}, {'B': False, 'C': True}),
# Notice not exact same, but cannot recreate exact scalers
    'shapes': (2, 7),  # IMPORTANT, Should be this exactly to reflect ENTIRE dictionary
    'df_index_lists': ([0, 3], [2, 1]),
    'column_names': (['A', 'D'], ['C', 'B'])
}


class Aux:
    def __init__(self):
        self.instance = None


aux = Aux()


class TestAggregator(unittest.TestCase):

    def test_smtg(self):
        start = 4
        st_scaled = c4_sc(start)
        st_unscaled = c4mm.inverse_transform([[st_scaled]])[0][0]
        self.assertEqual(start, st_unscaled, 'Unscaling is fine.')

    def test_builds_aggregate_dictionary(self):
        dictionary = DataHandler.DataHandler.build_aggregate_dictionary(tr, tst)
        dictionary['C'].sort()
        should_be = {'A': None, 'B': None, 'C': ['big dog', 'hello', 'hi', 'huncho', 'icon', 'quavo'], 'D': None}
        self.assertEqual(dictionary, should_be, 'Dictionary is correct')
        aux.instance = dictionary

    def test_split_and_encode(self):
        # Run Split / Encode
        tr_res, tst_res, _dict = DataHandler.DataHandler.split_and_encode(tr, tst, input_params, output_params)

        # Setup data  (for verification)
        exp_tr_data_in = np.array(exp_tr_res['data'][0])
        exp_tst_data_in = np.array(exp_test_res['data'][0])
        exp_tr_data_out_c = np.array(exp_tr_res['data'][1]['C'])
        exp_tst_data_out_c = np.array(exp_test_res['data'][1]['C'])
        exp_tr_data_out_b = np.array(exp_tr_res['data'][1]['B'])
        exp_tst_data_out_b = np.array(exp_test_res['data'][1]['B'])

        print(tr_res['data'][0])
        print(exp_tr_data_in)

        # Verify data is correct (ref. above for data format)
        self.assertTrue((exp_tr_data_in == tr_res['data'][0]).all(), 'Training data (X) is equal')
        self.assertTrue((exp_tst_data_in == tst_res['data'][0]).all(), 'Test data (X) is equal')
        self.assertTrue((exp_tr_data_out_c == tr_res['data'][1]['C']).all(), 'Training data (Y, C) is equal')
        self.assertTrue((exp_tst_data_out_c == tst_res['data'][1]['C']).all(), 'Training data (Y, C) is equal')
        self.assertTrue((exp_tr_data_out_b == tr_res['data'][1]['B']).all(), 'Training data (Y, B) is equal')
        self.assertTrue((exp_tst_data_out_b == tst_res['data'][1]['B']).all(), 'Training data (Y, B) is equal')

        # Verify encoders OK
        self.assertEqual(tr_res['encoders'][0]['A'] is None, exp_tr_res['label_encoders_null'][0]['A']
                         , 'Same train label encoders (A)')
        self.assertEqual(tr_res['encoders'][0]['D'] is None, exp_tr_res['label_encoders_null'][0]['D']
                         , 'Same train label encoders (D)')
        self.assertEqual(tst_res['encoders'][1]['B'] is None, exp_test_res['label_encoders_null'][1]['B']
                         , 'Same test label encoders (B)')
        self.assertEqual(tst_res['encoders'][1]['C'] is None, exp_test_res['label_encoders_null'][1]['C']
                         , 'Same test label encoders (C)')

        # Verify index lists OK
        self.assertEqual(tr_res['df_index_lists'], exp_tr_res['df_index_lists'], 'DF Index lists (tr) are equal')
        self.assertEqual(tst_res['df_index_lists'], exp_test_res['df_index_lists'], 'DF Index lists (tst) are equal')

        # Verify column names OK
        self.assertEqual(tr_res['column_names'], exp_tr_res['column_names'], 'DF Col. Names (tr) are equal')
        self.assertEqual(tst_res['column_names'], exp_test_res['column_names'], 'DF Col. Names (tst) are equal')
