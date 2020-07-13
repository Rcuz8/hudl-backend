import src.main.core.ai.utils.model.CustomCallbacks as CustomCallbacks
import unittest


class Aux:
    def __init__(self):
        self.calls = 0
        self.instance = None


    def someCallback(self, epoch):
        self.calls += 1

aux = Aux()

class Tester(unittest.TestCase):

    def test_creates(self):
        aux.instance = CustomCallbacks.ProgressCallback(20, aux.someCallback, 2)
        self.assertIsNotNone(aux.instance, 'Instantiated')

    def test_updates_progress(self):
        for i in range(20):
            aux.instance.on_train_batch_begin(i)
            aux.instance.on_train_batch_end(i)
            aux.instance.on_test_batch_begin(i)
            aux.instance.on_test_batch_end(i)
        self.assertEqual(aux.calls, 10)
