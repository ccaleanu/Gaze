import numpy as np

#Columbia dataset
def load_columbia():
    print("Columbia dataset loading method")

#MPII Gaze dataset
def load_mpii(path):
    print("Loading MPII dataset...")
    data = np.load(path)
    gazes = data['gaze']
    images = data['image']
    index = int(len(gazes) * 0.75)
    train_images = images[0:index]
    train_gazes = gazes[0:index]
    test_images = images[index:len(images)]
    test_gazes = gazes[index:len(gazes)]
    train_images = np.reshape(train_images, (len(train_images), 36, 60, 1))
    train_images = train_images.astype('float16') / 255
    test_images = np.reshape(test_images, (len(test_images), 36, 60, 1))
    test_images = test_images.astype('float16') / 255
    print('Dataset loaded')
    return train_images, train_gazes

# TBD dataset
def load_tbd_dataset():
    Print("TBD dataset loading method")