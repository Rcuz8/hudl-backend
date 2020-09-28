
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, ReLU, Input, Activation
from tensorflow.keras.models import Sequential
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.regularizers import l2

from src.main.util.io import warn, info
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import tensorflow.keras as keras

matplotlib.rcParams['figure.figsize'] = (12, 8)
pyplot.style.use('ggplot')

class MB:

    def __init__(self, model_params:dict, is_auxilary_process=False):
        self.model: keras.Model = None
        self.is_auxilary_process = is_auxilary_process
        self.history = None
        self.model_params = model_params

    # construct the model
    def construct(self, trial=None, nlayers=7, activation='relu',
                  custom=False, custom_layers=[(128, 'relu', 0.25) for i in range(4)],
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.002),
                  metrics=['accuracy'],
                  minHiddens= 1, maxHiddens= 3, hidden_width_min=12,hidden_width_max=120,
                  dropout_min=0.1,dropout_max=0.3,
                  lr_min = 2e-4, lr_max=1e-2, sequentialOverride=False
                  ):
        """
            Construct a keras model
        :param model_params: { 'inputs': 10 *inputs*, 'outputs': [ (2 *values*, 'softmax', 'categorical_cross..'), .. }
        :param nlayers: Number of neural net layers (can be swapped for custom 'custom_layers')
        :return: the model
        """

        model = None

        if not sequentialOverride:

            # Form input layer
            inp = keras.Input(shape=(self.model_params['inputs'],))

            if trial is not None:

                nlayers = trial.suggest_int("n_layers", minHiddens, maxHiddens)

                num_hidden = int(trial.suggest_loguniform("n_units_l0", hidden_width_min, hidden_width_max))
                hidden = Dense(num_hidden,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(inp)
                hidden = ReLU()(hidden)
                if (nlayers - 1 > 0):
                    for i in range(nlayers-1):
                        # DEPRE: Restrict # hiddens to max of prev # hiddens
                        dropout = trial.suggest_uniform("dropout_l{}".format(i), dropout_min, dropout_max)
                        num_hidden = int(trial.suggest_loguniform("n_units_l{}".format(i+1), hidden_width_min, hidden_width_max))
                        hidden = Dropout(dropout)(hidden)
                        hidden = Dense(num_hidden,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(hidden)
                        hidden = ReLU()(hidden)

                # We compile our model with a sampled learning rate.
                optimizer.learning_rate = trial.suggest_loguniform("lr", lr_min, lr_max)

            else:
                # Form hiddens
                if custom:
                    if len(custom_layers) == 0:
                        raise ValueError('Cannot contruct a custom model with no layers!')
                    act = custom_layers[0][1]
                    hidden = Dense(custom_layers[0][0],
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(inp)
                    if act == 'lrelu':
                        hidden = LeakyReLU()(hidden)
                    else:
                        hidden = ReLU()(hidden)
                    for layer in custom_layers[1:]:
                        act = layer[1]
                        hidden = Dropout(layer[2])(hidden)
                        hidden = Dense(layer[0],
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(hidden)
                        if act == 'lrelu':
                            hidden = LeakyReLU()(hidden)
                        else:
                            hidden = ReLU()(hidden)
                else:
                    hidden = Dense(128, activation=activation,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(inp)
                    while (nlayers > 1):
                        hidden = Dropout(dropout_min)(hidden)
                        hidden = Dense(128, activation=activation,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))(hidden)
                        nlayers -= 1

            # Form output layers (independent)
            outlayers = [Dense(layer[1], activation=layer[2], name=layer[0])(hidden) for layer in self.model_params['outputs']]

            # build model
            model = keras.Model(inputs=inp, outputs=outlayers)

            # Get loss names & weights
            lossnames = {name: item_entropy for name, _, _, item_entropy in self.model_params['outputs']}
            lossweights = {name: 1.0 for name, _, _, _ in self.model_params['outputs']}

            # Compile
            model.compile(optimizer=optimizer, loss=lossnames, loss_weights=lossweights,
                          metrics=metrics)
        else:

            if trial is not None:

                nlayers = trial.suggest_int("n_layers", minHiddens, maxHiddens)
                num_hidden = int(trial.suggest_loguniform("n_units_l0", hidden_width_min, hidden_width_max))
                model_layers = [
                    Input(shape=(self.model_params['inputs'],)),
                    Dense(num_hidden,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01)),
                    ReLU()
                ]
                if nlayers - 1 > 0:
                    for i in range(nlayers - 1):
                        dropout = trial.suggest_uniform("dropout_l{}".format(i), dropout_min, dropout_max)
                        num_hidden = int(
                            trial.suggest_loguniform("n_units_l{}".format(i + 1), hidden_width_min, hidden_width_max))
                        model_layers.append(Dropout(dropout))
                        model_layers.append(Dense(num_hidden,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01)))
                        model_layers.append(ReLU())

                # We compile our model with a sampled learning rate.
                optimizer.learning_rate = trial.suggest_loguniform("lr", lr_min, lr_max)

                if len(self.model_params['outputs']) > 1:
                    warn('Building Sequential model for multi-output problem. '
                         'Will likely have undefined behavior.')

                output = self.model_params['outputs'][0]

                # Add output layer
                outlayer = Dense(output[1], activation=output[2], name=output[0])
                model_layers.append(outlayer)
                # Loss
                lossnames = output[3]

                # build model
                model = Sequential(model_layers)

                # Compile
                model.compile(optimizer=optimizer, loss=lossnames,  metrics=metrics)


            else:

                # Form hiddens
                if custom:
                    if len(custom_layers) == 0:
                        raise ValueError('Cannot contruct a custom model with no layers!')
                    act = custom_layers[0][1]
                    model_layers = [
                        Input(shape=(self.model_params['inputs'],)),
                        Dense(custom_layers[0][0],
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))
                    ]
                    if act == 'lrelu':
                        model_layers.append(LeakyReLU())
                    elif act == 'relu':
                        model_layers.append(ReLU())
                    else:
                        model_layers.append(Activation(act))
                    for layer in custom_layers[1:]:
                        act = layer[1]
                        model_layers.append(Dropout(layer[2]))
                        model_layers.append(Dense(layer[0],
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01)))
                        if act == 'lrelu':
                            model_layers.append(LeakyReLU())
                        elif act == 'relu':
                            model_layers.append(ReLU())
                        else:
                            model_layers.append(Activation(act))
                else:
                    model_layers = [
                        Input(shape=(self.model_params['inputs'],)),
                        Dense(128,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))
                    ]
                    while nlayers > 1:
                        model_layers.append(Dropout(dropout_min))
                        model_layers.append(
                            Dense(128, activation=activation,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01)))
                        nlayers -= 1


                if len(self.model_params['outputs']) > 1:
                    warn('Building Sequential model for multi-output problem. '
                         'Will likely have undefined behavior.')

                output = self.model_params['outputs'][0]

                # Add output layer
                outlayer = Dense(output[1], activation=output[2], name=output[0])
                model_layers.append(outlayer)
                # Loss
                lossnames = output[3]

                # build model
                model = Sequential(model_layers)

                # Compile
                model.compile(optimizer=optimizer, loss=lossnames, metrics=metrics)

        info('Builder generated model.')

        model.summary()

        self.model = model
        return self

    # train the model
    def fit(self, X, Y, callback, epochs=100, batch_size=8, folds=6, repeats=3, isSequential=False):

        # quick maths
        epf = round(round(epochs / folds) / repeats)
        rem = epochs - (folds * repeats * epf)
        epochs_for_fold = [epf for _ in range(folds)]
        epochs_for_fold[-1] += int(rem / repeats)

        callbacks = None
        if callback is not None:
            callbacks = [callback]

        # create repeated k fold
        kf = RepeatedKFold(n_splits=folds, n_repeats=repeats)

        j = 0

        for train_index, test_index in kf.split(X):
            # Get x data
            X_train, X_test = X[train_index], X[test_index]
            # .. y is a bit more involved bc it's a dictionary
            y_train = {}
            y_test = {}
            for output_name, output_data in Y.items():
                try:
                    y_train[output_name] = output_data[train_index, :]
                    y_test[output_name] = output_data[test_index, :]
                except:
                    y_train[output_name] = output_data[train_index]
                    y_test[output_name] = output_data[test_index]

            if isSequential:
                y_train = list(y_train.items())[0][1]
                y_test = list(y_test.items())[0][1]

            val_data = (X_test,y_test)

            self.model.fit(X_train, y_train, validation_data=val_data, batch_size=batch_size,
                           epochs=epochs_for_fold[j % folds], verbose=0, callbacks=callbacks
                           )

            j += 1

        return self

    # evaluate the model
    def evaluate(self, X, Y, output_plaintext_names):
        # print('Evaluating')
        evaluation = self.model.evaluate(X, Y, verbose=0)
        s = ''
        if (len(output_plaintext_names) > 1):
            # print('Loss ', round(evaluation[0], 2))
            at_i = 1
            # for i in range(len(output_plaintext_names)):
                # print(output_plaintext_names[i] + ' Loss ', round(evaluation[at_i + i], 2))
            at_i += len(output_plaintext_names)
            for i in range(len(output_plaintext_names)):
                acc = output_plaintext_names[i] + ' Accuracy : ' + "{0:.0%}".format(evaluation[at_i + i])
                s += acc + '\n'
                # print(acc)
        return s, evaluation

    # plot the model
    def plot(self, output_column_names, output_plaintext_names):
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        if (len(output_column_names) == 1):
            pyplot.plot(self.history.history["val_loss"], label=output_plaintext_names[0])
        else:
            for colname, plaintextname in np.column_stack((output_column_names, output_plaintext_names)):
                pyplot.plot(self.history.history[colname + "_loss"], label=plaintextname)
        pyplot.legend()
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        if (len(output_column_names) == 1):
            pyplot.plot(self.history.history["val_accuracy"], label=output_plaintext_names[0])
        else:
            for colname, plaintextname in np.column_stack((output_column_names, output_plaintext_names)):
                pyplot.plot(self.history.history[colname + "_accuracy"], label=plaintextname)
        pyplot.legend()
        pyplot.show()
        return self

    def heatmap(self, df):
        cols = df.columns
        colours = ['#baed9a', '#591615']  # specify the colours - red is missing. green is not missing.
        sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
        pd.options.mode.chained_assignment = None
        pyplot.show()
        # print(self.data_presence())
        return self
