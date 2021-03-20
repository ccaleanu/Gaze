from utils import loaddb, utils, config, models
from tensorflow.keras.optimizers import RMSprop
import numpy as np

def train_mpii():
    x_train, y_train = loaddb.load_mpii(config.dbpath_mpii)
    model = models.mpii_model()
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss=utils.degrees_mean_error, metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size)
    return 0
    
    