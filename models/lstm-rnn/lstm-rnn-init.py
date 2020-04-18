
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam


class rnn:
    def __init__(self, input_shape, dim1, dim2):
        
        print("init LSTM RNN model ...")
        model = Sequential()

        model.add(LSTM(units=dim1, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
        print("init the first layer of the RNN")
        model.add(LSTM(units=dim2,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        print("init the second layer of the RNN")
        model.add(Dense(units=input_shape, activation="softmax"))
        
        print("Compiling ...")
        # Keras optimizer defaults:
        # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
        # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
        # SGD    : lr=0.01,  momentum=0.,                             decay=0.
        opt = Adam()
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()

        print("finish initialization")
    
    def train(self, features, target_labels, batch_size, epochs):
        history = self.model.fit(x=features, y=target_labels, batch_size = 35, epochs = 10)
        return history

    def validate(self, features, target_labels, batch_size):
        (score, eval_accuracy) = self.model.evaluate(x=features, y=target_labels, batch_size=batch_size, verbose=1)
        return (score, eval_accuracy)

    def test(self, features, target_labels, batch_size):
        (score, eval_accuracy) = self.model.evaluate(x=features, y=target_labels, batch_size=batch_size, verbose=1)
        return (score, eval_accuracy)
        
    def save(self, model_path):
        self.model.save(model_path)




