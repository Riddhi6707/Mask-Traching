import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss
sm.set_framework('tf.keras')
K.set_image_data_format('channels_last')

def build_model(image_size,  class_count=1, BACKBONE = 'resnet18'):
    

        input_data = Input(image_size+(4,), dtype='float32')
        inp_3C = Conv2D(3, (1, 1))(input_data) 
        
        base_model = sm.Unet(backbone_name=BACKBONE,encoder_weights='imagenet',input_shape=image_size + (3,))
             
        out_data = base_model(inp_3C)
        
        model = Model(inputs=input_data, outputs=out_data)
        
        loss = sm.losses.JaccardLoss() +  sm.losses.BinaryFocalLoss()  #

        dice =  sm.metrics.IOUScore() 
        
        opt = Adam(lr = .0002)
        
        model.compile(optimizer=opt, loss=loss, metrics=[dice])

        return model


