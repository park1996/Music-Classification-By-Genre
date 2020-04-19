
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam

import os
# use plaidml lib as the background support for AMD
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

os.environ["RUNFILES_DIR"] = "/usr/local/share/plaidml"

os.environ["PLAIDML_NATIVE_PATH"] = "/usr/local/lib/libplaidml.dylib"

class RNN_MODEL:
    def __init__(self, input_shape, dim1, dim2, output_size):
        
        print("init LSTM RNN model ...")    
        self.model = Sequential()

        self.model.add(LSTM(units=dim1, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
        print("init the first layer of the RNN")
        self.model.add(LSTM(units=dim2,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        print("init the second layer of the RNN")

        self.model.add(Dense(units=output_size, activation="softmax"))

        # code modified and refered from https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification.git
        # Keras optimizer defaults:
        # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
        # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
        # SGD    : lr=0.01,  momentum=0.,                             decay=0.
        
        opt = Adam()
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.model.summary()

        print("finsish initialization")
    
    def train(self, features, target_labels, batch_size, epochs):
        history = self.model.fit(x=features, y=target_labels, batch_size = batch_size, epochs = epochs)
        return history

    def test(self, features, target_labels):
        (score, eval_accuracy) = self.model.evaluate(x=features, y=target_labels, batch_size=35, verbose=1)
        return (score, eval_accuracy)
        
    def save(self, model_path):
        self.model.save(model_path)




