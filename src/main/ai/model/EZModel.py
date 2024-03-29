from src.main.ai.model import CustomCallbacks
from src.main.ai.data.Builder import huncho_data_bldr
from src.main.ai.model.Builder import MB
import matplotlib.pyplot as pyplot
from src.main.util.io import info, ok, white
import optuna
import tensorflow.keras as keras
import numpy as np
from src.main.ai.model.Predictor import Predictor


class EZModel:

    def __init__(self, db: huncho_data_bldr):
        self.output_column_names = db.output_column_names
        self.output_plaintext_names = db.output_plaintext_names
        self.ptxt_for = dict(zip(db.output_column_names, db.output_plaintext_names))
        self.scalers = db.scalers()['train']  # should be the same for both
        self.encoders = db.encoders()['train']  # should be the same for both
        self.df_indices = db.relevent_dataframe_indices()['train']  # should be the same for both
        self.df_columns = db.columns()
        self.dictionary = db.dictionary
        self.mp = db.model_params()
        self.train_x, self.train_y, self.test_x, self.test_y = db.data()
        self.training = None
        self.test = None
        self.training_result = None
        self.test_result = None
        self.mb = None
        self.training_files = []
        self.test_files = []
        self.didCompile = False
        self.modpar = None
        self.optimal_layers = None
        self.optimal_lr = None
        self.predictor = None
        self.__training_accuracies = None
        self.__training_report = None
        self.isSequential = False

    def build(self, use_optimized_dimensions=True, nlayers=7, lr=None, activation='relu', custom=False,
              custom_layers=[(128, 'relu', 0.25), (64, 'relu', 0.25)],
              optimizer=keras.optimizers.Adam(learning_rate=0.002), forceSequential=False,
              metrics=['categorical_crossentropy', 'categorical_accuracy'], autobuild_predictor=True
              ):
        if (lr is not None):
            optimizer.learning_rate = lr
        # Insert optimal dimensions
        if (use_optimized_dimensions and self.optimal_layers is not None):
            custom_layers = self.optimal_layers
            optimizer.learning_rate = self.optimal_lr
        # Generate model params
        mp = self.mp.copy()
        # Build model
        self.mb = MB(mp) \
            .construct(nlayers=nlayers, activation=activation, custom=custom,
                       custom_layers=custom_layers, optimizer=optimizer, sequentialOverride=forceSequential,
                       metrics=metrics)
        self.isSequential = forceSequential
        if autobuild_predictor:
            self.build_predictor()
        return self

    def optimize(self, minHiddens=1, maxHiddens=3, hidden_width_min=10, hidden_width_max=120,
                 dropout_min=0.2, dropout_max=0.5,
                 lr_min=1e-4, lr_max=1e-2, ntrials=40, epochs=60, pruner=optuna.pruners.HyperbandPruner(),
                 metrics=['categorical_crossentropy', 'categorical_accuracy']):
        # Generate model params
        mp = self.mp.copy()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(Objective(self.train_x, self.train_y, self.test_x, self.test_y,
                                 self.output_plaintext_names,
                                 mp, minHiddens=minHiddens,
                                 maxHiddens=maxHiddens, hidden_width_min=hidden_width_min,
                                 hidden_width_max=hidden_width_max,
                                 dropout_min=dropout_min, dropout_max=dropout_max, lr_min=lr_min, lr_max=lr_max,
                                 epochs=epochs, metrics=metrics
                                 ), n_trials=ntrials)
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        opt = list(trial.params.items())
        n_layers = opt.pop(0)[1]
        self.optimal_lr = opt.pop(-1)[1]
        self.optimal_layers = []
        print(opt)

        for i in range(n_layers):
            num_hiddens = int(opt.pop(0)[1])
            dropout = 0
            if (len(opt) > 1):
                dropout = int(opt.pop(0)[1])
            self.optimal_layers.append((num_hiddens, 'relu', dropout))
        print('Generated optimal layers: ', self.optimal_layers)
        print('Generated optimal lr: ', self.optimal_lr)
        return self

    def summarize(self):
        print(self.mb.model.summary())
        return self

    def get_keras_model(self):
        return self.mb.model

    def output_dictionary(self):
        """Dictionary for each output. For numerical, would need a similar output_scalers() function.
        """
        return {col: self.dictionary[col] for col in self.output_column_names}

    def output_value_at(self, loc: int):
        """Get some item from the vocabulary of the output variable.
        This is useful when comparing actual plaintext outputs (actual vs expected).
        Parameters
        ----------
        loc : int
            The location of the item in the dictionary of the output variable

        Returns
        -------
            The plaintext output value
        """
        if not self.isSequential:
            raise ValueError("Cuz-handled error. Unimplemented dictionary get for multiple outputs.")

        outputdict = self.dictionary[self.output_column_names[0]]
        return outputdict[loc]

    def __reformat_results(self, res: list):
        s = "  "
        # tl = [(ax[i], ay[i]) for i in range(min(len(ax), len(ay)))]
        for item in res:
            s += item[0].ljust(10) + " (" + str(int(item[1]*100)) + "%)   "
        return s

    def show_model_predictions(self, test_X: list, test_Y: list, encode=False, waterfall_size=2):
        if waterfall_size < 2:
            raise ValueError("Unimplemented : waterfall_size < 2 in model_predictions() helper.")
        """Get bulk model predictions for some test data in a human-readable way.
        """



        try:
            test_Y[0]
        except:
            test_Y = test_Y[list(dict(test_Y))[0]]  # Get first item

        if type(test_Y[0]) != np.ndarray and type(test_Y[0]) != list:
            raise ValueError("Unimplemented : model_predictions for non-categorical outputs.")

        for index, predicted in enumerate(self.predictor.predict_bulk(test_X, encode, waterfall_size)):
            '''[[('HOUSTON', 0.43867397), ('MINNESOTA', 0.09643049), ('DENVER', 0.08414613)]]'''

            # Get the max of the list (WARNING: Intended for categorical output lists)
            y: list = test_Y[index]
            actual = int(np.argmax(y))

            # Now let's get their plaintext versions
            actual_pxt: str = self.output_value_at(actual)
            white("Actual: ", actual_pxt.ljust(15), "\t\tPredicted: ", self.__reformat_results(predicted[0]).ljust(35),
                  "CORRECT" if str(predicted[0][0][0]) == actual_pxt else " ")




    def data(self):
        return self.train_x, self.train_y, self.test_x, self.test_y

    @classmethod
    def __k_for(cls, epochs, num_notifications):
        return epochs / num_notifications

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

    def train(self, epochs=1000, plot=False, batch_size=8,
              notif_every=50, on_update=None):
        info('\nBegin Training.')

        callback = CustomCallbacks.ProgressCallback(epochs, k=notif_every, batch_size=batch_size) \
            .add_test_info(self.mb.model, self.test_x, self.test_y, self.isSequential) \
            .add_progress_update_fn(on_update)

        repeats = 3
        folds = 5

        self.mb.fit(self.train_x, self.train_y, callback=callback, epochs=epochs,
                    repeats=repeats, folds=folds, batch_size=batch_size, isSequential=self.isSequential)

        self.__training_accuracies = callback.accuracies()
        self.__training_report = callback.training_report()

        if (plot):

            X = self.__training_report["epoch"].tolist()

            loss_cols = [col for col in callback.evals.columns if col.lower().find('loss') >= 0]
            acc_cols = [col for col in callback.evals.columns if col.lower().find('accuracy') >= 0]

            print('Here are the columns: \n\tLoss:', loss_cols, '\n\tAcc :', acc_cols)

            fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1)
            ax1.set_title('Average Accuracy')
            ax2.set_title('Max Accuracy')
            ax3.set_title('Loss')

            for col in acc_cols:
                ptxt = self.__label_for(col)
                ax1.plot(X, callback.evals[col].rolling(3, min_periods=1).mean(), label=ptxt)
                ax2.plot(X, callback.evals[col].rolling(len(X), min_periods=1).max(), label=ptxt)

            ax1.legend(loc="upper right")
            ax2.legend(loc="upper right")

            for col in loss_cols:
                ptxt = self.__label_for(col)
                ax3.plot(X, callback.evals[col].values, label=ptxt)

            ax3.legend(loc="upper right")
            pyplot.show()

        return self

    def build_predictor(self):
        model = self.mb.model
        input_encoders = self.encoders[0]
        output_encoders = self.encoders[1]
        input_scalers = self.scalers[0]
        output_scalers = self.scalers[1]
        in_col_df_indices = self.df_indices[0]
        out_col_df_indices = self.df_indices[1]
        dictionary = self.dictionary
        columns = self.df_columns
        self.predictor = Predictor(model, input_encoders, output_encoders, input_scalers, output_scalers,
                                   in_col_df_indices, out_col_df_indices, dictionary, columns=columns)
        return self

    def training_accuracies(self):
        return self.__training_accuracies

    def training_report(self):
        return self.__training_report

    def set_training_accuracies(self, to):
        self.__training_accuracies = to
        return self

    def eval(self):
        print('Testing')
        return self.mb.evaluate(self.test_x, self.test_y, self.output_plaintext_names)

    def test_predictions(self):
        pred = self.predictor.predict_bulk(self.test_x, encode=False, waterfall_size=3)
        return pred

    def plot(self):
        self.mb.plot(self.output_column_names, self.output_plaintext_names)
        return self

    def training_heatmap(self):
        self.mb.heatmap(self.training)
        return self

    def test_heatmap(self):
        self.mb.heatmap(self.test)
        return self


class Objective(object):

    def __init__(self, train_x, train_y, test_x, test_y, output_plaintext_names, model_params,
                 minHiddens=1, maxHiddens=3, hidden_width_min=10, hidden_width_max=120,
                 dropout_min=0.2, dropout_max=0.5,
                 lr_min=1e-4, lr_max=1e-2, epochs=35, metrics=['categorical_crossentropy', 'categorical_accuracy']
                 ):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model_params = model_params
        self.output_plaintext_names = output_plaintext_names
        self.minHiddens = minHiddens
        self.maxHiddens = maxHiddens
        self.hidden_width_min = hidden_width_min
        self.hidden_width_max = hidden_width_max
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epochs = epochs
        self.metrics = metrics

    def __call__(self, trial):
        # Clear clutter from previous session graphs.
        keras.backend.clear_session()

        # Generate our trial model.
        mb = MB(self.model_params, True).construct(trial=trial,
                                                   minHiddens=self.minHiddens,
                                                   maxHiddens=self.maxHiddens, hidden_width_min=self.hidden_width_min,
                                                   hidden_width_max=self.hidden_width_max,
                                                   dropout_min=self.dropout_min, dropout_max=self.dropout_max,
                                                   lr_min=self.lr_min,
                                                   lr_max=self.lr_max, metrics=self.metrics) \
            .fit(self.train_x, self.train_y, None, batch_size=5, epochs=self.epochs)

        _, evaluation = mb.evaluate(self.test_x, self.test_y, self.output_plaintext_names)

        acc = mb.model.metrics_names.index('accuracy')

        return evaluation[acc]

# # 300 represents number of points to make between T.min and T.max
# xnew = np.linspace(min(X), max(X), 900)
# spl1 = make_interp_spline(X, roll_avg, k=5)  # type: BSpline
# spl2 = make_interp_spline(X, roll_max, k=1)  # type: BSpline
# avg_smooth = spl1(xnew)
# max_smooth = spl2(xnew)
# pyplot.plot(xnew, avg_smooth, label='Avg Accuracy')
# pyplot.plot(xnew, max_smooth, label='Max Accuracy')
