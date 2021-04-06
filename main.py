import config
import utils
from utils.DataGenerator import DataGen
from tensorflow.python.keras.callbacks import  ModelCheckpoint
from utils.segmentation_models import build_model
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
# Hyper parameters from config file

    batch_size = config.batch_size    
    batch_count = config.batch_count    
    image_dir = config.image_dir    
    mask_dir = config.mask_dir    
    image_height = config.image_height    
    image_width = config.image_width    
    nepochs = config.epochs_no

#DataGenerator class instantiation for training and validation

    train_gen = DataGen(batch_count, image_dir, mask_dir, image_height, image_width,batch_size,mode = "Train")    
    valid_gen = DataGen(100, image_dir, mask_dir, image_height, image_width,2,mode = "Valid")
    
#Model save checkpoint
    filepath = r'C:/Users/RRay Cha/source/repos/MaskNet-Code/models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'        
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#Build and Train Model
    model = build_model((image_height,image_width))
    model.summary()
    H = model.fit( train_gen ,batch_size=batch_size,callbacks=[checkpoint],
                  epochs=nepochs, validation_data=valid_gen,verbose=1 )
    
#Generate the training logs
    
    print(H.history.keys())
    lossNames = ["loss","loss"]
    N = np.arange(0,nepochs)
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(2, 1, figsize=(13, 13))

        # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(N, H.history[l], label=l)
        ax[i].plot(N, H.history["val_" + l], label="val_" + l)
        ax[i].legend()

    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plotPath =r'C:/Users/RRay Cha/source/repos/MaskNet-Code/models/losses.png' 
    plt.savefig(plotPath)
    plt.close()

    # create a new figure for the accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["iou_score"],
        label="detection_train_acc")
    plt.plot(N, H.history["val_iou_score"],
        label="val_detection_acc")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    # save the accuracies plot
    plotPath=r'models/accs.png'
        
    plt.savefig(plotPath)