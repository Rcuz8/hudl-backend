import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from constants import experiment

class ProgressCallback(tf.keras.callbacks.Callback):

    def __init__(self, epochs, cb=None, k=20, is_auxilary=False):
        self.epochs = epochs
        self.n = 0
        self.pbar = tqdm(total=epochs) if not is_auxilary else None
        if self.pbar is not None:
            self.pbar.set_description('\nTraining Model.       ')
        self.k = k
        self.cb = cb
        self.is_auxilary = is_auxilary
        self.test_x = None
        self.test_y = None
        self.model_ = None
        self.eval_results = []
        self.evals:pd.DataFrame = None
        self.use_k = True

    def add_test_info(self, model, test_x, test_y):
        self.model_ = model
        self.test_x = test_x
        self.test_y = test_y
        return self

    def add_col(self,col):
        if self.evals is None:
            self.evals = pd.DataFrame(columns=self.model_.metrics_names)
            print('Created callbacks attribute "evals" = (DataFrame)\n ', self.evals)
        self.evals.loc[len(self.evals.index)] = col

    def dismiss_k(self):
        self.use_k = False
        return self

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        self.n += 1
        if self.use_k:

            if (self.n % self.k == 0 or self.n == self.epochs):
                if self.pbar is not None:
                    self.pbar.update(self.k)
                if (self.cb is not None and not self.is_auxilary):
                    self.cb(self.n)
                if self.test_x is not None:
                    evaluation = self.model_.evaluate(self.test_x,self.test_y,verbose=2)
                    self.eval_results.append(evaluation)

                    self.add_col(evaluation)
                    if (self.n >= self.epochs):
                        self.model.stop_training = True

        else:
            if self.pbar is not None:
                self.pbar.update(self.k)
            if (self.cb is not None and not self.is_auxilary):
                self.cb(self.n)
            if self.test_x is not None:
                evaluation = self.model_.evaluate(self.test_x, self.test_y,verbose=0)
                if (experiment is not None):
                    experiment.log_metrics(evaluation,epoch=self.n)
                    self.eval_results.append(evaluation)
        if (self.n == self.epochs):
            if self.pbar is not None:
                self.pbar.update(self.k)
                self.pbar.set_description('\nCompleted Training.   ')
                self.pbar.close()





