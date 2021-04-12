
import os
import numpy as np
import config
import utils
from utils.DataGenerator_Online import DataGen

from tensorflow.python.keras.callbacks import  ModelCheckpoint,CSVLogger
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true = K.flatten(y_true)
    unlabeled = 0.5 * (tf.sign(y_true + 0.5) + 1)
    y_true = unlabeled * y_true

    y_pred = K.flatten(y_pred)
    y_pred = unlabeled * y_pred

    intersection = K.sum(y_true * y_pred)

    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)



if __name__ == '__main__':
    
    
    batch_size = config.batch_size    
    batch_count = config.batch_count    
    image_dir = config.image_dir    
    mask_dir = config.mask_dir    
    image_height = config.image_height    
    image_width = config.image_width    
    nepochs = config.epochs_no
    
    test_class = config.test_class
    test_vid = config.test_dir
    test_mask = config.test_mask_dir
    mode = config.mode
    Results = config.Results
    
    filename = r'C:\Users\antsexp\Desktop\Training-Trial\MaskTrack_Solution\models-1-sm-resnet-imagenet-valid-10\models-valid-4-bcatch_count-500\model-ep068-loss0.010-val_loss0.238.h5'
       #filename = r".\models_1000_1\model- ep001-loss0.051-val_loss0.051.h5"
    model = load_model(filename,custom_objects={'dice_coefficient': dice_coefficient})
    #model = load_model(filename,custom_objects={'iou_score': sm.metrics.IOUScore()})
   # model =  tf.saved_model.load("models/")
    
    frame_ids = os.listdir(test_vid)
    
    if config.mode == 'online':
    
        online_train_gen = DataGen(100, image_dir, mask_dir, test_class, image_height, image_width,8)
        
        filepath = r'models/online/model-ep{epoch:03d}-loss{loss:.3f}.h5'        
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger('training_online.csv', separator = ',')
        
        
        online_model = model
        online_model.fit(online_train_gen ,batch_size=batch_size,callbacks=[checkpoint,csv_logger],
                      epochs=nepochs, verbose=1)
    
    accs = []
    for i in range(1,len(frame_ids)):
        
            im_org = cv2.imread(os.path.join(test_vid,frame_ids[i]))            
            im_org = cv2.resize(im_org, (image_width,image_height), interpolation = cv2.INTER_AREA)
            im = np.array(im_org, dtype="float32") / 255.0 
            
            label_path = os.path.join(test_mask,frame_ids[0])
            label_path = os.path.splitext(label_path)[0]
            label_path = label_path + ".png"
            label_org = cv2.imread(label_path)
          #print(label_path)
            label_org = cv2.resize(label_org, (image_width,image_height), interpolation = cv2.INTER_AREA)
        
            label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)   
        # k = (np.unique(label.flatten()))[0]
            _,im_label = cv2.threshold(label,0,255,cv2.THRESH_BINARY) 
            im_label = np.array(im_label,dtype = "float32")/255.0   
           
            # if i == 1:
            #         label_path = os.path.join(test_mask,frame_ids[0])
            #         label_path = os.path.splitext(label_path)[0]
            #         label_path = label_path + ".png"
            #         # print(label_path)
            #         label_org = cv2.imread(label_path)
            #         label_org = cv2.resize(label_org, (image_width,image_height), interpolation = cv2.INTER_AREA)
            #         label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)  
            #         # print(np.unique(label.flatten()))
            #         # k = (np.unique(label.flatten()))[0]
            #         _,im_label = cv2.threshold(label,0,255,cv2.THRESH_BINARY) 
            #         im_label = np.array(im_label,dtype = "float32")/255.0 

            # else :
            #         label_path = os.path.join(Results, test_class) + "/offline/" + str(i-1) + ".png" 
                    
            #        # print(label_path)
            #         label_org = cv2.imread(label_path)
            #         label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY) 
            #         _,im_label = cv2.threshold(label,0,255,cv2.THRESH_BINARY) 
            #         im_label = np.array(label,dtype = "float32")/255.0 
               
            
            
            new = np.dstack((im, im_label))
            new = new.reshape(1,image_height, image_width,-1)
           # print(new.shape)
            
            gt_label_path = os.path.join(test_mask,frame_ids[i])
            gt_label_path = os.path.splitext(gt_label_path)[0]
            gt_label_path = gt_label_path + ".png"
            gt_label_org = cv2.imread(gt_label_path)
            gt_label_org = cv2.resize(gt_label_org, (image_width,image_height), interpolation = cv2.INTER_AREA)      
            gt_label = cv2.cvtColor(gt_label_org, cv2.COLOR_BGR2GRAY)   
           # k = (np.unique(gt_label.flatten()))[0]
            _,gt_im_label = cv2.threshold(gt_label,0,255,cv2.THRESH_BINARY) 
            gt_im_label = np.array(gt_im_label,dtype = "float32")/255.0 
            
            if mode == 'offline':
                
               # print(" new shape : ",new.shape)
                pred = model.predict(new)
                finalPred = (pred>0.5).astype(np.uint8)
                path = os.path.join(Results, test_class) + "/offline/" + str(i) + ".png"
                
                cv2.imwrite(path,np.squeeze(finalPred)*255)
                
                
                evalResult = model.evaluate(new,gt_im_label[np.newaxis, :, :,np.newaxis],batch_size = 1,verbose = 0)
                print("acc :", evalResult[1])
                accs.append(evalResult[1])
                
            
                
            elif mode == 'online':
                
                pred = online_model.predict(new)
                finalPred = (pred>0.5).astype(np.uint8)
                path = os.path.join(Results, test_class) + "/online/" + str(i) + ".png"
                cv2.imwrite(path,np.squeeze(finalPred)*255)
                
                evalResult = online_model.evaluate(new,gt_im_label[np.newaxis, :, :,np.newaxis],verbose = 0,batch_size = 1)
               # print("acc :", evalResult[1])
                accs.append(evalResult[1])
                
    fig, ax = plt.subplots()
    ax.boxplot(accs, meanline=True, showmeans=True)
    plt.xticks([1],[test_class])
   # plt.yticks([1],["Frame Dice-Coefficient"])
    plt.savefig(os.path.join(Results, test_class) +'/' +  mode +'/frame_accs.png')
        
        
