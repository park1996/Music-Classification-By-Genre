from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf

class cnn:
    def __init__(self):
        #Model construction code based on the following links:
        #https://engmrk.com/module-22-implementation-of-cnn-using-keras/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
        #https://levelup.gitconnected.com/audio-data-analysis-using-deep-learning-with-python-part-2-4a1f40d3708d
        self.MODEL = Sequential()
        input_shape=(64, 64, 3)
        self.MODEL.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
        self.MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        self.MODEL.add(Conv2D(64, kernel_size=(5,5), input_shape=input_shape))
        self.MODEL.add(MaxPooling2D(pool_size=(2, 2)))
        self.MODEL.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        self.MODEL.add(Dense(1024, activation=tf.nn.relu))
        self.MODEL.add(Dropout(0.2))
        self.MODEL.add(Dense(10,activation=tf.nn.softmax))
        self.MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def train(self, features, target_labels):
        self.MODEL.fit(x=features, y=target_labels, epochs=10)