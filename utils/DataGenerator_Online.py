import os
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import albumentations as A





class DataGen(Sequence):
    def __init__(self,  batch_count,image_dir, mask_dir, test_class, img_height, img_width, batch_size):
       
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size        
        self.batch_x =  None 
        self.batch_y = None         
        #self.mode = mode       
        self.batch_count = batch_count
        #self.class_count = np.zeros(len(os.listdir(self.image_dir))).tolist()
        #self.classes = os.listdir(self.image_dir)
        self.test_class = test_class

    def __len__(self):
              
# The batch_count should be atleast equal to the number of the video classes. Preferably multiple times of the no of classes.
        return  self.batch_count #if self.batch_count >= len(os.listdir(self.image_dir)) else len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        
        
#Training consists of randomly chosen frames of given  batch size from any of the video group. There will be 
#deisgnated no of batches as per batch count. 

      #  if self.mode == "Train" :
            
            self.batch_x =  np.zeros((self.batch_size, self.img_height, self.img_width, 4))
            self.batch_y = np.zeros((self.batch_size, self.img_height, self.img_width, 1))
            
           # idx = np.random.randint(0,len(self.classes))
           # self.class_count[idx] += 1
            
            # if self.class_count[idx] == np.floor(self.batch_count/len(os.listdir(self.image_dir))):
            #     (self.classes).remove(str(self.classes[idx]))
            #     self.class_count.pop(idx)
            
            selected_class_video = os.path.join(self.image_dir,self.test_class)
            selected_class_mask =  os.path.join(self.mask_dir,self.test_class)
            
           
            frame_ids = os.listdir(selected_class_video)
            ids = np.random.randint(1, len(frame_ids) - 2, self.batch_size)
            ct = 0
            for i in range(1,self.batch_size + 1):



                    im_org = cv2.imread(os.path.join(selected_class_video,frame_ids[i]))  
                   # im_org.resize((self.img_height, self.img_width,im_org.shape[2]))
                    im_org = cv2.resize(im_org, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                    
                    
                    label_path = os.path.join(selected_class_mask,frame_ids[0])
                    label_path = os.path.splitext(label_path)[0]
                    label_path = label_path + ".png"
                    label_org = cv2.imread(label_path)
                   # label_org.resize((self.img_height, self.img_width,label_org.shape[2])) 
                    label_org = cv2.resize(label_org, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                    label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)   
                    k = (np.unique(label.flatten()))[0]
                    _,im_label = cv2.threshold(label,(k+10),255,cv2.THRESH_BINARY)  
                    
                    
#                     aug = A.Compose([
#                         A.VerticalFlip(p=0.5,height=self.img_height, width=self.img_width),              
#                         A.RandomRotate90(p=0.5)])
                        
#                     np.random.seed(7)
#                     augmented = aug(image=im_org, mask=im_label)
                    
#                     im = augmented['image']
#                     im_label = augmented['mask']
                    im_label = np.array(im_label,dtype = "float32")/255.0
                    
                    x = np.random.randint(0,5,1)
                    y = np.random.randint(0,5,1)
                    M = np.float32([[1,0,x],[0,1,y]])
                    trans_thresh = cv2.warpAffine(im_label,M,( self.img_width,self.img_height))
                    trans_thresh =  np.array(trans_thresh,dtype = "float32")/255.0                 
                    aug = A.ElasticTransform(p=1, alpha=20, sigma=30 * 0.05, alpha_affine=30 * 0.03)
                    np.random.seed(7)
                    augmented = aug(image=im_org, mask= trans_thresh)
                    mask_final = augmented['mask']
                    
                    im = np.array(im_org, dtype="float32") / 255.0 
                    new = np.dstack((im, mask_final))
        
                        
                   
                                    
                    self.batch_y[ct, :, :, 0] = im_label[:, :]
                    self.batch_x[ct, :, :, :] = new[:, :, :]
                    ct += 1
           
           
            return self.batch_x, self.batch_y
        
    def get_label(self):
        return self.batch_x, self.batch_y
