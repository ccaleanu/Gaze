from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#MPII Model
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
      layers.Dense(2)
    ])
    return model
   