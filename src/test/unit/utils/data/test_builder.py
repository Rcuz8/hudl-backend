from src.main.core.ai.utils.data.Input import Input as inp
from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
import unittest as u
import src.test.utils as test
import pandas as pd
import numpy as np


def matrix_to_keras_output(matrix, outputs):
    """Convert raw output data into keras-formatted output data

    Parameters
    ----------
    matrix : list(list)
        2D Array
    outputs : list(str)
        Model output list

    Returns
    -------
    Keras-style output

    Examples
    --------

        mx = [[1,2],[3,4]]
        outputs = ['A','B']

    >>> matrix_to_keras_output(mx, outputs)
    {'A': [1, 2], 'B': [3, 4]}

    """
    return {item[0]: item[1] for item in list(zip(outputs, matrix))}

def x_assert(case, expected, actual,training=True):
    description = ('Training' if training else 'Test') + ' data (X) checks out'
    case.assertEqual(expected, actual.tolist(), description)

def y_assert(case, expected, actual,training=True):
    description = ('Training' if training else 'Test') + ' data (Y) checks out'
    ok = expected == {key: val.tolist() for key, val in actual.items()}
    case.assertTrue(ok, description)


class Tester(u.TestCase):

    def test_prepare_raw_perfect(self):
        sources = [test.data.copy(), test.data.copy()]
        outputs = test.outputs
        output_column_names = [col[0] for col in outputs]
        trainx, trainy, testx, testy = bldr(test.inputs, test.outputs) \
            .with_type('raw') \
            .with_filenames(['Source 1', 'Source 2']) \
            .with_heads(test.columns) \
            .with_important_columns(test.columns) \
            .with_iterating_adjuster(test.filter) \
            .with_batch_adjuster(test.double) \
            .and_train(sources) \
            .and_eval(sources) \
            .prepare() \
            .data()

        exp_trainx = exp_testx = [
            [1],
            [1]
        ]

        exp_y = [
            [0, 0],
            [0, 0]
        ]

        exp_trainy = exp_testy = matrix_to_keras_output(exp_y, output_column_names)

        x_assert(self, exp_trainx, trainx, training=True)
        x_assert(self, exp_testx, testx, training=False)
        y_assert(self, exp_trainy, trainy, training=True)
        y_assert(self, exp_testy, testy, training=True)

