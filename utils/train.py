import config
from utils import loaddb, utils, models
from tensorflow.keras.optimizers import RMSprop
import numpy as np

import autokeras as ak
import tensorflow as tf

def train_mpii():
    x_train, y_train = loaddb.load_mpii(config.dbpath_mpii)
    # model = models.mpii_model()
    # model.compile(optimizer=RMSprop(learning_rate=0.0001), loss=utils.degrees_mean_error, metrics=['acc'])
    
    #Try AllClassic on
    num_classes = 2
    
    myModel = __import__(config.myModelType + '.' + 'AllClassic', fromlist=['AllClassic'])
    myClassifier = getattr(myModel, 'AllClassic')
    
    model = myClassifier.build(num_classes)
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy'])
    #---------------------------------------------------------------------------------------  
    history = model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size)
    
    return 0
    
def train_ak():
    image_count = len(list(config.database_path.glob('**/*.jpg')))
    print("# of images found:", image_count)

    list_ds = tf.data.Dataset.list_files(str(config.database_path/'*/*.jpg'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = list_ds.map(utils.process_path, num_parallel_calls=AUTOTUNE)
    
    features = np.array([list(x[0].numpy()) for x in list(train_ds)])
    labels = np.array([x[1].numpy() for x in list(train_ds)])

    input_node = ak.ImageInput()
    output_node = ak.Normalization()(input_node)
    output_node = ak.ImageAugmentation(horizontal_flip=False, vertical_flip=False, rotation_factor=False, zoom_factor=False)(output_node)
    output_node = ak.ClassificationHead()(output_node)
    
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=config.max_trials,
        directory=config.outpath_mpii)
    # Feed the tensorflow Dataset to the classifier.

    split = config.split
    x_val = features[split:]
    y_val = labels[split:]
    x_train = features[:split]
    y_train = labels[:split]

    clf.fit(
        x_train,
        y_train,
        # Use your own validation set.
        validation_data=(x_val, y_val),
        epochs=config.epochs,
    )

    # Predict with the best model.
    #predicted_y = clf.predict(x_val)
    #print(predicted_y)


    # Evaluate the best model with testing data.
    print(clf.evaluate(x_val, y_val))

    # Export as a Keras Model.
    model = clf.export_model()

    print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>
    model.save(config.output_path + "model_autokeras.h5")
    
    return 0