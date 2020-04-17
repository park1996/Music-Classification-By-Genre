from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.regularizers import l2
import tensorflow as tf

class cnn:
    def __init__(self, input_shape):
        #Model construction code based on the following links:
        #https://engmrk.com/module-22-implementation-of-cnn-using-keras/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
        #https://levelup.gitconnected.com/audio-data-analysis-using-deep-learning-with-python-part-2-4a1f40d3708d
        self.MODEL = Sequential()
        self.MODEL.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        self.MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        self.MODEL.add(Conv2D(64, kernel_size=(5,5), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        self.MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        self.MODEL.add(Conv2D(64, kernel_size=(5,5), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        self.MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        self.MODEL.add(Conv2D(64, kernel_size=(5,5), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        self.MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        self.MODEL.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        self.MODEL.add(Dense(1024, activation=tf.nn.relu))
        self.MODEL.add(Dropout(0.2))
        self.MODEL.add(Dense(10,activation=tf.nn.softmax))
        self.MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def train(self, features, target_labels):
        history=self.MODEL.fit(x=features, y=target_labels, epochs=10)
        return history
    def test(self, features, target_labels):
        (eval_loss, eval_accuracy) = self.MODEL.evaluate(x=features, y=target_labels)
        return (eval_loss, eval_accuracy)
    def predict(self, features):
        return self.MODEL.predict_classes(features)
    def save(self, model_path):
        self.MODEL.save(model_path)
    def load(self, model_path):
        self.MODEL = load_model(model_path)