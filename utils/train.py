from utils import loaddb, utils, config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.optimizers import RMSprop

def mpii_model():  
    model = Sequential([
      #layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.1),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.3),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(11)
    ])
    return model

def train_mpii():
    x_train, y_train = loaddb.load_mpii(config.dbpath_mpii)
    model = mpii_model()
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss=utils.degrees_mean_error, metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=16)
    
    return x_train, y_train