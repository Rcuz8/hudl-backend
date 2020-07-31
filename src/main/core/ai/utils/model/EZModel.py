from src.main.core.ai.utils.model import CustomCallbacks
from src.main.core.ai.utils.data.Builder import huncho_data_bldr
from src.main.core.ai.utils.model.Builder import MB
import matplotlib.pyplot as pyplot
import numpy as np
import optuna
import keras

from src.main.core.ai.utils.model.Predictor import Predictor


class EZModel:

    def __init__(self, db:huncho_data_bldr):
        self.output_column_names = db.output_column_names
        self.output_plaintext_names = db.output_plaintext_names
        self.ptxt_for = dict(zip(db.output_column_names, db.output_plaintext_names))
        self.scalers = db.scalers()['train']                        # should be the same for both
        self.encoders = db.encoders()['train']                      # should be the same for both
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


    def build(self, use_optimized_dimensions=True, nlayers=7, lr=None, activation='relu', custom=False,
              custom_layers=[(128, 'relu', 0.25),(64, 'relu', 0.25)],
              optimizer=keras.optimizers.Adam(learning_rate=0.002), forceSequential=False
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
                       custom_layers=custom_layers, optimizer=optimizer, sequentialOverride=forceSequential)

        return self

    def optimize(self, minHiddens= 1, maxHiddens= 3, hidden_width_min=10,hidden_width_max=120,
                 dropout_min=0.2,dropout_max=0.5,
                 lr_min=1e-4, lr_max=1e-2, ntrials=40, epochs=60, pruner=optuna.pruners.HyperbandPruner()):
        # Generate model params
        mp = self.mp.copy()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(Objective(self.train_x, self.train_y, self.test_x, self.test_y,
                                 self.output_plaintext_names,
                                 mp, minHiddens=minHiddens,
                                 maxHiddens=maxHiddens, hidden_width_min=hidden_width_min,
                                 hidden_width_max=hidden_width_max,
                                 dropout_min=dropout_min, dropout_max=dropout_max, lr_min=lr_min, lr_max=lr_max,
                                 epochs=epochs
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
        print('Outputs: ', self.output_column_names)
        print('Dict   : ', self.dictionary)
        return {col: self.dictionary[col] for col in self.output_column_names}

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
        print('BEGIN TRAINING:')
        print()

        callback = CustomCallbacks.ProgressCallback(epochs, k=notif_every, batch_size=batch_size) \
            .add_test_info(self.mb.model, self.test_x, self.test_y)\
            .add_progress_update_fn(on_update)

        repeats = 3
        folds = 5


        self.mb.fit(self.train_x, self.train_y,callback=callback,epochs=epochs,
                    repeats=repeats, folds=folds, batch_size=batch_size)

        self.__training_accuracies = callback.accuracies()

        if (plot):

            # quick maths
            epf = round(round(epochs / folds) / repeats)
            rem = epochs - (folds * repeats * epf)
            epochs_for_fold = [epf for _ in range(folds)]
            epochs_for_fold[-1] += int(rem / repeats)

            X = [(1+i) * callback.k for i in range(len(callback.eval_results))]

            loss_cols = [col for col in callback.evals.columns if col.lower().find('loss') >= 0]
            acc_cols = [col for col in callback.evals.columns if col.lower().find('accuracy') >= 0]

            print('Here are the columns: \n\tLoss:',loss_cols,'\n\tAcc :', acc_cols)

            fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1)
            ax1.set_title('Average Accuracy')
            ax2.set_title('Max Accuracy')
            ax3.set_title('Loss')

            for col in acc_cols:
                ptxt = self.__label_for(col)
                ax1.plot(X, callback.evals[col].rolling(5, min_periods=1).mean(), label=ptxt)
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
        model               = self.mb.model
        input_encoders      = self.encoders[0]
        output_encoders     = self.encoders[1]
        input_scalers       = self.scalers[0]
        output_scalers      = self.scalers[1]
        in_col_df_indices   = self.df_indices[0]
        out_col_df_indices  = self.df_indices[1]
        dictionary          = self.dictionary
        columns             = self.df_columns
        self.predictor = Predictor(model, input_encoders, output_encoders, input_scalers, output_scalers,
                                   in_col_df_indices, out_col_df_indices, dictionary, columns=columns)
        return self

    def training_accuracies(self):
        return self.__training_accuracies

    def set_training_accuracies(self, to):
        self.__training_accuracies = to
        return self

    def eval(self):
        print('Testing')
        return self.mb.evaluate(self.test_x, self.test_y,self.output_plaintext_names)

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
                 lr_min=1e-4, lr_max=1e-2, epochs=35
                 ):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model_params = model_params
        self.output_plaintext_names = output_plaintext_names
        self.minHiddens=minHiddens
        self.maxHiddens=maxHiddens
        self.hidden_width_min=hidden_width_min
        self.hidden_width_max=hidden_width_max
        self.dropout_min=dropout_min
        self.dropout_max=dropout_max
        self.lr_min=lr_min
        self.lr_max=lr_max
        self.epochs = epochs

    def __call__(self, trial):
        # Clear clutter from previous session graphs.
        keras.backend.clear_session()

        # Generate our trial model.
        mb = MB(self.model_params, True).construct(trial=trial,
                   minHiddens=self.minHiddens,
                   maxHiddens=self.maxHiddens, hidden_width_min=self.hidden_width_min,
                   hidden_width_max=self.hidden_width_max,
                   dropout_min=self.dropout_min, dropout_max=self.dropout_max, lr_min=self.lr_min,
                   lr_max=self.lr_max) \
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