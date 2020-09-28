import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

ins = pd.read_csv('testfiles/form_out_inputs.csv', header=None)
outs = pd.read_csv('testfiles/form_out_outputs.csv', header=None)

X = ins.values.tolist()
Y = outs.values.tolist()

print(X)
print(Y)

# Globals
EPOCHS = 900
width = 20


def load_model():
    # Fit Data Specs
    input_length = len(X[0])
    output_length = len(Y[0])
    # Model Layers
    inp = lambda: Input(input_length)
    out = lambda: Dense(output_length, 'softmax')
    dense = lambda: Dense(width, 'relu',
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          activity_regularizer=l2(0.01))
    drp = lambda: Dropout(0.1)

    # Create Model
    model = Sequential([
        inp(), dense(), drp(), dense(), drp(), out()
    ])

    model.compile(loss='categorical_crossentropy', optimizer=SGD(0.005, 0.4, True, clipvalue=0.2),
                  metrics=['categorical_crossentropy', 'categorical_accuracy'])

    model.summary()

    return model


model = load_model()

history = model.fit(X, Y, batch_size=6, epochs=EPOCHS, verbose=0)


def pl():
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


pl()

for pred in model.predict(X[0:20]):
    print('Max Value:  [', np.argmax(pred), '] : ', pred[np.argmax(pred)])

# tm = Sequential([Input(2), Dense(5), Dense(2, activation='softmax')])
# tm.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# data = [[1,0], [0,1],[0,0]]
# print(tm.predict(data))
