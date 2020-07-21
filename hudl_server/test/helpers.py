import pandas as pd
import hudl_server.helpers as helpers
import constants as con
import unittest
import numpy as np
from keras.models import Sequential as seq
import keras.layers as layers

from firebase_admin import initialize_app, credentials
cred = credentials.Certificate('../fbpk.json')
initialize_app(cred, {
    'storageBucket': 'hudpred.appspot.com'
})


sample_data_testing_conversion = [
    [
        {'0': [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None, None, None]},
        {'0': [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None, None, None]},
{'0': [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None, None, None]},
        {'0': [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None, None, None]},
{'0': [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None, None, None]},
        {'0': [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None, None, None]},
{'0': [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None, None, None]},
        {'0': [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None, None, None]},
{'0': [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None, None, None]},
        {'0': [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None, None]},
        {'0': [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None, None, None]}
    ]
]

sample_data = [
    [
        {'0': [1, 'K', np.nan, np.nan, 'L', -35, 'KO Rec', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]},
        {'0': [2, 'O', 1.0, 10.0, 'R', -23, 'Run', np.nan, 5.0, 'BIG', 'NY', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]},
        {'0': [3, 'O', 2.0, 5.0, 'R', -28, 'Run', np.nan, 1.0, 'TRIO', 'NY', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]},
        {'0': [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', np.nan, 7.0, np.nan, 'SEAGULL', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}
    ]
]

class db_helpers_tester(unittest.TestCase):

    # def test_extract_data_from_fb(self):
    #     some_data = helpers.fb_data_to_matrix(sample_data_testing_conversion.copy())
    #     expected = [
    #         [
    #             [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None, None, None],
    #             [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None, None],
    #             [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None, None],
    #             [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None, None, None],
    #             [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None,
    #              None, None],
    #             [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None,
    #              None, None],
    #             [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None,
    #              None, None],
    #             [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None,
    #              None, None],
    #             [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None,
    #              None, None],
    #             [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None,
    #              None, None],
    #             [1, 'K', None, None, 'L', -35, 'KO Rec', None, None, None, None, None, None, None, None, None, None,
    #              None, None],
    #             [2, 'O', 1.0, 10.0, 'R', -23, 'Run', None, 5.0, 'BIG', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [3, 'O', 2.0, 5.0, 'R', -28, 'Run', None, 1.0, 'TRIO', 'NY', None, None, None, None, None, None, None,
    #              None],
    #             [4, 'O', 3.0, 4.0, 'R', -29, 'Pass', None, 7.0, None, 'SEAGULL', None, None, None, None, None, None,
    #              None, None],
    #         ]
    #     ]
    #     self.assertEqual(some_data, expected, 'Successfully converts firebase data to matrix')
    #
    # def test_bldr_make(self):
    #     cfgs = con.model_gen_configs
    #     config = cfgs.post_align_play
    #     data = helpers.fb_data_to_matrix(sample_data_testing_conversion.copy())
    #     model = helpers.bldr_make(config, train=data, test=data)
    #     self.assertIsNotNone(model, 'Model is OK')

    def test_deploy_model(self):
        import keras
        import numpy as np
        from keras.layers import ActivityRegularization

        inputs = keras.Input(shape=(3,))
        outputs = ActivityRegularization()(inputs)
        model = keras.Model(inputs, outputs)

        # If there is a loss passed in `compile`, thee regularization
        # losses get added to it
        model.compile(optimizer="adam", loss="mse")
        model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

        game_id = '14403d51-0da9-4961-82ef-4b5019cd1c5f'
        game_id = ''.join(game_id.split('-')) # Happens in fb file
        helpers.deploy_model(model, game_id)




