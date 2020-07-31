import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from constants import experiment
from src.main.util.io import info, ok, err

class ProgressCallback(tf.keras.callbacks.Callback):

    def __init__(self, epochs, cb=None, k=20, is_auxilary=False, batch_size=8):
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
        self.progress_update_fn = None
        self.batch_size = batch_size

    def add_progress_update_fn(self, fn):
        self.progress_update_fn = fn
        return self

    def add_test_info(self, model, test_x, test_y):
        self.model_ = model
        self.test_x = test_x
        self.test_y = test_y
        return self

    def add_col(self,col): # Add ROW
        if self.evals is None:
            self.evals = pd.DataFrame(columns=['epoch'] + self.model_.metrics_names)
            info('Created callbacks attribute "evals" = (DataFrame)')
        self.evals.loc[len(self.evals.index)] = [self.n] + col

    def accuracies(self):
        """Gets the accuracies.

        Example/Test:

            import pandas as pd

            class X:
                def __init__(self):
                    self.evals = pd.DataFrame({'epoch': [1,2,3], 'test_accuracy': [4,5,6], 'shouldNotInclude': [7,8,9]})
                def accuracies(self):
                    acc_cols = [col for col in self.evals.columns if col.lower().find('accuracy') >= 0]

                    accs = {'epoch': self.evals['epoch'].values.tolist()}
                    for col in acc_cols:
                        accs[col] = self.evals[col].values.tolist()

                    return accs

            print(X().accuracies()) # {'epoch': [1, 2, 3], 'test_accuracy': [4, 5, 6]}

        """
        try:
            acc_cols = [col for col in self.evals.columns if col.lower().find('accuracy') >= 0]
        except:
            return None # No test data

        accs = {'epoch': self.evals['epoch']}
        for col in acc_cols:
            accs[col] = self.evals[col]

        return accs

    def __label_for(self, col):
        cl = col.lower()
        if cl == 'accuracy':
            return 'Total Accuracy'
        if cl == 'loss':
            return 'Total Loss'
        if cl.find('_accuracy') >= 0:
            return cl[0:cl.find('_accuracy')]
        if cl.find('_loss') >= 0:
            return cl[0:cl.find('_loss')]
        raise Exception('Modeler label_for() : Cannot find label for ', col)

    def dismiss_k(self):
        self.use_k = False
        return self

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def trigger_update_fn(self, done=False):
        if self.progress_update_fn:
            if not done:
                self.progress_update_fn(self.n / self.epochs)
            else:
                self.progress_update_fn(1.0)

    def on_epoch_end(self, epoch, logs=None):
        self.n += 1
        epoch = self.n
        if self.use_k:

            if (epoch % self.k == 0 or epoch == self.epochs):
                if self.pbar is not None:
                    self.pbar.update(self.k)
                    self.trigger_update_fn()
                if (self.cb is not None and not self.is_auxilary):
                    self.cb(epoch)
                if self.test_x is not None:
                    evaluation = self.model_.evaluate(self.test_x,self.test_y,verbose=2)
                    self.eval_results.append(evaluation)

                    self.add_col(evaluation)
                    if (epoch >= self.epochs):
                        self.model.stop_training = True

        else:
            if self.pbar is not None:
                self.pbar.update(self.k)
                self.trigger_update_fn()
            if (self.cb is not None and not self.is_auxilary):
                self.cb(epoch)
            if self.test_x is not None:
                evaluation = self.model_.evaluate(self.test_x, self.test_y, verbose=0)
                if (experiment is not None):
                    experiment.log_metrics(evaluation,epoch=epoch)
                    self.eval_results.append(evaluation)
        if abs(self.epochs - epoch) <= 2:
            if self.pbar is not None:
                self.pbar.update(self.k)
                self.trigger_update_fn(done=True)
                self.pbar.set_description('\nCompleted Training.   ')
                self.pbar.close()
                self.model.stop_training = True




import keras

def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_dim=784))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
#
# # Limit the data to 1000 samples
# x_train = x_train[:1000]
# y_train = y_train[:1000]
# x_test = x_test[:1000]
# y_test = y_test[:1000]
#
#
# model = get_model()
# EPOCHS = 25
# def cb(x):
#     print(x)
# model.fit(
#     x_train,
#     y_train,
#     batch_size=128,
#     epochs=EPOCHS,
#     verbose=0,
#     validation_split=0.5,
#     callbacks=[ProgressCallback(epochs=EPOCHS,k=4,batch_size=8)],
# )
# #
# # res = model.evaluate(
# #     x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]
# # )
# #
# # res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])

