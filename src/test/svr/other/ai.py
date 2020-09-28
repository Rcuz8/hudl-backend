# # multi-class classification with Keras
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import datetime
# # from tensorflow import keras
# import keras
# from keras.layers import Dense
# from keras.utils import np_utils
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from tensorflow.python.keras.callbacks import TensorBoard
#
# # load dataset
# # get csv
# df = pd.read_csv('../../data/nnabrev/WillWait.txt')
# dataset = df.values
#
#
# # list = [( min, max, encode_as )]
# def encoded(ds, _list):
#     X = None
#     for _min, _max, encode_as in _list:
#         # Encode G
#         if encode_as == 'float':
#             # [:,...] is the 2D splice where you take an entire column, for ... columns
#             G = ds[:, _min: _max].astype(float)
#         elif encode_as == 'int':
#             # [:,...] is the 2D splice where you take an entire column, for ... columns
#             G = ds[:, _min: _max].astype(int)
#         elif encode_as == 'one-hot':
#             # get column
#             G = ds[:, _min: _max]
#             # reshape column as array
#             G = G.ravel()
#             # encode class values as integers
#             encoder = LabelEncoder()
#             encoder.fit(G)
#             encoded_G = encoder.transform(G)
#             # convert integers to dummy variables (i.e. one hot encoded)
#             dummy_y = np_utils.to_categorical(encoded_G)
#             G = dummy_y
#             # elif encode_as == 'embedded':
#             # 	G = ds[:, _min: _max].astype(int)
#             # Append G
#         if X is None:
#             X = G
#         else:
#             X = np.column_stack((X, G))
#     return X
#
#
# # Get Inputs
# X = encoded(dataset, [(0, 1, 'int'), (1, 2, 'one-hot'), (2, 3, 'float')])
# print('Inputs: ', X[0:4])
# # Store # inputs
# input_shape = X.shape[1]
#
# # Get Outputs
# out_restname = Y = encoded(dataset, [(3, 4, 'one-hot')])
# out_frustration = encoded(dataset, [(4, 5, 'float')])
# # Store # outputs
# restaurant_neurons = out_restname.shape[1]
# frustration_neurons = out_frustration.shape[1]
#
# batch_size = 8
# ep = 200
#
# # Form input layer
# inp = keras.Input(shape=(input_shape,))
#
# hidden = Dense(20, activation='relu')(inp)
# hidden = Dense(128, activation='relu')(hidden)
# hidden = Dense(128, activation='relu')(hidden)
#
# # Form output layer
# branchA = Dense(restaurant_neurons, activation="softmax", name='restaurant_output')(hidden)
# branchB = Dense(frustration_neurons, activation="linear", name='frustration_output')(hidden)
# # out = layers.concatenate([branchA, branchB])
#
# _model = keras.Model(
#     inputs=inp,
#     outputs=[branchA, branchB]
# )
#
# # define two dictionaries: one that specifies the loss method for
# # each output of the network along with a second dictionary that
# # specifies the weight per loss
# losses = {
# 	"restaurant_output": "categorical_crossentropy",
# 	"frustration_output": "mse",
# }
# lossWeights = {"restaurant_output": 1.0, "frustration_output": 1.0}
#
# # compile the model
# _model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
#
# # summarize the model
# print(_model.summary())
# print('Input data (X) sample: \n', X[0:4])
# print('Output data (out_restname) sample: \n', out_restname[0:4])
# print('Output data (out_frustration) sample: \n', out_frustration[0:4])
#
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# # train the network to perform multi-output classification
# H = _model.fit(X, {"restaurant_output": out_restname, "frustration_output": out_frustration}, validation_split=0.2, epochs=ep, verbose=1, callbacks=[tb])
# # evaluate the model
# loss, lossr, lossf, accuracyr, accuracyf = _model.evaluate(X, {"restaurant_output": out_restname, "frustration_output": out_frustration}, verbose=0)
# print('Epochs: %f' % (ep))
# print('Loss: %f' % (loss))
# print('Restaurant Loss: %f' % (lossr))
# print('Frustration Loss: %f' % (lossf))
# print("Restaurant Accuracy: {:.1%}".format(accuracyr))
# print("Frustration Accuracy: {:.1%}".format(accuracyf))