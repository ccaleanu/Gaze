import numpy as np
import os
import scipy.io as sio
from tensorflow.keras.backend import cos, sin, sqrt, mean
from tensorflow import acos

#MPII utils
def convert_gaze_3d_2d(vect):
    x, y, z = vect
    phi = np.arctan2(-x, -z)
    theta = np.arcsin(-y)
    return np.array([theta, phi])
 
def convert_gaze_2d_3d(angles):
    x = -cos(angles[:, 0]) * sin(angles[:, 1])
    y = -sin(angles[:, 0])
    z = -cos(angles[:, 1]) * cos(angles[:, 1])
    norm = sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

def degrees_mean_error(y_true, y_pred):
    x_p, y_p, z_p = convert_gaze_2d_3d(y_pred)
    x_t, y_t, z_t = convert_gaze_2d_3d(y_true)
    angles = mean(x_p * x_t + y_p * y_t + z_p * z_t)
    error_value = (acos(angles) * 180 / np.pi)
    return error_value
    
#Preprocess MPII
def read_mat(path_mat):
    content = sio.loadmat(path_mat, struct_as_record=False, squeeze_me=True)
    data = content['data']
    return data
    
def get_data(path_data):
    images = []
    gazes = []

    for patient in os.listdir(path_data):
        full_path_patient = os.path.join(path_data, patient)
        for day_name in os.listdir(full_path_patient):
            full_day_path = os.path.join(full_path_patient, day_name)
            #print('Read data from: ', full_day_path)

            content = read_mat(full_day_path)

            left_images = content.left.image
            left_gazes = content.left.gaze

            right_images = content.right.image
            right_gazes = content.right.gaze

            if left_images.shape == (36, 60):
                left_images = left_images[np.newaxis, :, :]
                left_gazes = left_gazes[np.newaxis, :]

            if right_images.shape == (36, 60):
                right_images = right_images[np.newaxis, :, :]
                right_gazes = right_gazes[np.newaxis, :]

            for i in np.arange(0, len(left_gazes), 1):

                images.append(left_images[i])
                images.append(right_images[i])
                
                gazes.append(convert_gaze_3d_2d(left_gazes[i]))
                gazes.append(convert_gaze_3d_2d(right_gazes[i]))

    return images, gazes