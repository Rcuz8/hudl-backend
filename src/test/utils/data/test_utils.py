from constants import sample_data as data, model_gen_configs as configs, data_headers
import src.main.ai.data.Tiny_utils as tiny
import src.test.utils as test
import unittest as u
import pandas as pd
import numpy as np

config = configs.post_align_play.io
frame = pd.DataFrame(data, columns=data_headers)
data_dict = Utils.build_dictionary_for(frame)
data_bounds = Utils.build_boundaries_for(frame)

class TestUtils(u.TestCase):

    def test_missing_data_splits(self):
        tr = pd.DataFrame(
            [[1, 2, np.nan, 4],
             [4, np.nan, 'hello', 7],
             [8, 9, 'huncho', np.nan],
             ], columns=['A', 'B', 'C', 'D'])

        in_params = [('A', 'Col A', 'int')]
        out_params = [('B', 'Col B', 'int'), ('C', 'Col C', 'one-hot')]

        splits = Utils.missing_data_splits(tr, in_params, out_params)
        self.assertEqual(len(splits), 2, 'There are 2 in/out split sets for 2 missing-data columns')

        ins, outs = splits[0]
        self.assertEqual(len(ins), 2, 'First split has 2 inputs')
        self.assertEqual(len(outs), 1, 'First split has 1 output')

        ins, outs = splits[1]
        self.assertEqual(len(ins), 2, '2nd split has 2 inputs')
        self.assertEqual(len(outs), 1, '2nd split has 1 output')

    def test_df_power_wash(self):

        # Drop Scarce Columns

        actual = Utils.df_power_wash(test.frame.copy(), test.inputs.copy(), test.outputs.copy(), test.double,
                                     relevent_columns_override=test.columns.copy(), drop_scarce_columns=False)

        expected = pd.DataFrame ([['aa',    2.0,       4.0,     6.0]], columns=test.columns)

        self.assertTrue(expected.equals(actual), 'DataFrames are not the same.')

        # Don't Drop Scarce Columns

        actual = Utils.df_power_wash(test.frame, test.inputs, test.outputs, test.double,
                                     relevent_columns_override=test.columns, drop_scarce_columns=True)

        expected_columns = test.columns.copy()
        expected_columns.remove('D')
        expected_columns.remove('C')

        expected = pd.DataFrame([
            ['aa', 2.0],
            ['bb', 8.0],
            ['cc', 12.0]
        ], columns=expected_columns)

        self.assertTrue(expected.equals(actual), 'DataFrames are not the same.')

    def test_analyze_data_quality(self):

        # Dropping columns

        actual = Utils.analyze_data_quality(
            sources=[test.data.copy()],
            injected_headers=test.columns,
            preprocessing_fn=test.double,
            injected_filenames=['Source 1'],
            data_format='raw',
            relevent_columns_configs=test.configs,
            drop_scarce_columns=True
        )

        expected = [
            { 'name': 'Source 1',
              'quality_evaluations': [{'name': 'Config 1', 'quality': 0.6}, {'name': 'Config 2', 'quality': 0.6}],
              'missing': {'A': 0.2, 'B': 0.4, 'C': 0.6, 'D': 0.8},
              'data': tiny.nans_to_nones(test.data.copy())
             }
        ]

        for i in range(len(actual)):
            actual[i]['data'] = tiny.nans_to_nones(actual[i]['data'])

        self.assertEqual(actual, expected)

         # Not dropping columns

        actual = Utils.analyze_data_quality(
            sources=[test.data.copy()],
            injected_headers=test.columns,
            preprocessing_fn=test.double,
            injected_filenames=['Source 1'],
            data_format='raw',
            relevent_columns_configs=test.configs,
            drop_scarce_columns=False
        )

        expected = [
            {'name': 'Source 1',
             'quality_evaluations': [{'name': 'Config 1', 'quality': 0.2}, {'name': 'Config 2', 'quality': 0.4}],
             'missing': {'A': 0.2, 'B': 0.4, 'C': 0.6, 'D': 0.8},
             'data': tiny.nans_to_nones(test.data.copy())
             }
        ]

        for i in range(len(actual)):
            actual[i]['data'] = tiny.nans_to_nones(actual[i]['data'])

        self.assertEqual(actual, expected)










