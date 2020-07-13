# import pandas as pd
# import keras
# from keras.layers import Dense
# import src.main.core.ai.utils.DataUtils as du
# from matplotlib import pyplot
# import tensorflow as tf
# import numpy as np
# from keras.utils import np_utils
# from array import array
#
# from src.main.core.ai.utils.Predictor import Predictor
#
# BATCH_SIZE = 3
# EPOCHS = 100
#
#
#
#
# # load & handle dataset
#
# df = pd.read_csv('../../data/nnabrev/dinner.txt')                # get csv
#
# dataset = df.values                                              # grab values
# # Define Input & output representations
# in_struct =  [ ('FOOD', 'one-hot'), ('DRINKS', 'int') ]
# out_struct = [ ('PLACE', 'one-hot'), ('TTL_COST', 'int') ]
#
# # encode a list of columns
# X, input_encoders = du.col_encode(df, in_struct)
# # Store # inputs
# input_shape = X.shape[1]
#
# # Get properly encoded outputs columns
# Y_place, place_enc_asList = du.col_encode(df, [out_struct[0]])
# Y_ttlCost, ttlcost_enc_asList = du.col_encode(df, [out_struct[1]])
# output_encoders = []
# output_encoders.extend(place_enc_asList)
# output_encoders.extend(ttlcost_enc_asList)
#
# Out = {
#     'names': { "place": "categorical_crossentropy", "ttlCost": "mean_squared_error" },
#     'weights': { "place": 1.0, "ttlCost": 1.0 },
#     'neurons': { 'place': Y_place.shape[1], 'ttlCost':  1 },
#     'data': { "place": Y_place, "ttlCost": Y_ttlCost }
# }
#
# # Form input layer
# inp = keras.Input(shape=(input_shape,))
# #Form Hidden
# hidden = Dense(128, activation='relu')(inp)
# hidden = Dense(128, activation='relu')(hidden)
# hidden = Dense(128, activation='relu')(hidden)
# hidden = Dense(128, activation='relu')(hidden)
# hidden = Dense(128, activation='relu')(hidden)
# # Form output layers
# branchA = Dense(Out['neurons']['place'], activation="softmax", name='place')(hidden)
# branchB = Dense(Out['neurons']['ttlCost'], activation='linear', name='ttlCost')(hidden)
#
# # build model
# _model = keras.Model( inputs=inp, outputs=[branchA, branchB] )
# # compile the model
# _model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0015), loss=Out['names'], loss_weights=Out['weights'], metrics=['accuracy'])
#
# # summarize the model
# print(_model.summary())
#
# # train the network to perform multi-output classification
# history = _model.fit(X, Out['data'], validation_split=0.2, epochs=EPOCHS, verbose=1)
#
# in_col_df_indices = [df.columns.get_loc(colname) for colname, _ in in_struct]
# out_col_df_indices = [df.columns.get_loc(colname) for colname, _ in out_struct]
#
# print('Formal Prediction: ', Predictor(_model, input_encoders, output_encoders,df, in_col_df_indices, out_col_df_indices).predict(['wings', 3]))
# # _maxes = maxes(prediction, _out_struct=out_struct)
# # print('Matrix maxes: ', _maxes)
# # invert(_maxes, output_encoders, prediction)
# # print('Which translated to: ',  _maxes)
# # print('Restaurant: ', _maxes['PLACE'], 'Total: ', _maxes['TTL_COST'])
# #
#
#
#
#
#
#
# # # evaluate the model
# # res = _model.evaluate(X, Out['data'], verbose=0)
#
# # # plot loss during training
# # pyplot.subplot(211)
# # pyplot.title('Loss')
# # pyplot.plot(history.history['place_loss'], label='place')
# # pyplot.plot(history.history['ttlCost_loss'], label='total cost')
# # pyplot.legend()
# # # plot accuracy during training
# # pyplot.subplot(212)
# # pyplot.title('Accuracy')
# # pyplot.plot(history.history['place_accuracy'], label='place')
# # pyplot.plot(history.history['ttlCost_accuracy'], label='total cost')
# # pyplot.legend()
# # pyplot.show()