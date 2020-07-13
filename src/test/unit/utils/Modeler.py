from src.main.core.ai.utils.model.EZModel import Modeler
import unittest
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Aux:
    def __init__(self):
        self.instance = None


aux = Aux()

files = ['../../../../data/nn/hudl-hobart_vs_rpi.json', '../../../../data/nn/hudl-hobart_vs_slu.json','../../../../data/nn/hudl-hobart_vs_union.json']

# Define Input & output representations
input_params = [('PLAY_NUM', 'Play Num', 'int'), ('ODK', 'ODK', 'one-hot'), ('DN', 'Down', 'int'),
                  ('DIST', 'Distance', 'int'), ('HASH', 'Hash Line', 'one-hot'), ('YARD_LN', 'Yard Line', 'int')]
output_params = [('PLAY_TYPE', 'Play Type', 'one-hot'), ('OFF_FORM', 'Offensive Formation', 'one-hot'),
                   ('OFF_PLAY', 'Offensive Play', 'one-hot')]



class Tester(unittest.TestCase):

    def test_creates(self):
        modelr = Modeler(input_params, output_params) \
            .add_training_json(files[0]) \
            .add_training_json(files[1]) \
            .add_test_json(files[2]) \
            .build(optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),custom=True) \
            .summarize() \
            .train(1000)\
            .plot()\
            .eval()

        self.assertIsNotNone(modelr, 'Builds Modeler')
