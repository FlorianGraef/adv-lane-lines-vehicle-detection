from keras.models import load_model
import numpy as np
import matplotlib as plt
import cv2

from keras import backend as K

smooth = 1
def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

class VehicleInference():
    def __init__(self, model_path, nn_input_dim):
        self.model = load_model(model_path, custom_objects={'IOU_calc_loss': IOU_calc_loss, 'IOU_calc':IOU_calc})
        #self.model.load_weights('sem_seg_unet30e_w.h5')
        self.nn_input_dims = nn_input_dim

    def get_mask(self, img):
        batch_wrapped = np.empty([1, self.nn_input_dims[1], self.nn_input_dims[0], 3])

        batch_wrapped[0] = cv2.resize(img, self.nn_input_dims)
        pred_mask = (self.model.predict(batch_wrapped))[0]

        return pred_mask
