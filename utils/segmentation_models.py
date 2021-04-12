import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss
sm.set_framework('tf.keras')
K.set_image_data_format('channels_last')

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true = K.flatten(y_true)
    unlabeled = 0.5 * (tf.sign(y_true + 0.5) + 1)
    y_true = unlabeled * y_true

    y_pred = K.flatten(y_pred)
    y_pred = unlabeled * y_pred

    intersection = K.sum(y_true * y_pred)

    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def build_model(image_size,  class_count=1, BACKBONE = 'resnet18'):
    

        input_data = Input(image_size+(4,), dtype='float32')
        inp_3C = Conv2D(3, (1, 1))(input_data) 
      #  inp_3C= tf.keras.applications.resnet.preprocess_input(temp)
        
        base_model = sm.Unet(backbone_name=BACKBONE,encoder_weights='imagenet',input_shape=image_size + (3,))
             
        out_data = base_model(inp_3C)
        
        model = Model(inputs=input_data, outputs=out_data)
        
        loss = "binary_crossentropy" 

        dice =  dice_coefficient #sm.metrics.IOUScore() 
        
        opt = Adam(lr = .0001)
        
        model.compile(optimizer=opt, loss=loss, metrics=[dice])

        return model


