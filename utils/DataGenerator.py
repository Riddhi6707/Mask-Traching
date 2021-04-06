
import os
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import albumentations as A
#np.random.seed(np.random.randint(5000))




class DataGen(Sequence):
    def __init__(self,  batch_count,image_dir, mask_dir, img_height, img_width, batch_size, mode = 'Train'):
       
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size        
        self.batch_x =  None 
        self.batch_y = None         
        self.mode = mode       
        self.batch_count = batch_count
        self.class_count = np.zeros(len(os.listdir(self.image_dir))).tolist()
        self.classes = os.listdir(self.image_dir)

    def __len__(self):
              
# The batch_count should be atleast equal to the number of the video classes. Preferably multiple times of the no of classes.
        return  self.batch_count if self.batch_count >= len(os.listdir(self.image_dir)) else len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        
        
#Training consists of randomly chosen frames of given  batch size from any of the video group. There will be 
#deisgnated no of batches as per batch count. 

        if self.mode == "Train" :
            
            self.batch_x =  np.zeros((self.batch_size, self.img_height, self.img_width, 4))
            self.batch_y = np.zeros((self.batch_size, self.img_height, self.img_width, 1))
          #  print(len(self.classes)-1)
            ll = np.random.randint(0,(len(self.classes)-1),10)
            idt =ll[0] 
          #  print("idt:" ,ll)
            self.class_count[idt] += 1
            
           # print(self.batch_count)
           # print(len(os.listdir(self.image_dir)))
           # print(np.floor(self.batch_count/len(os.listdir(self.image_dir))))
            if self.class_count[idt] == np.floor(self.batch_count/len(os.listdir(self.image_dir))):
                #print("hello")
                (self.classes).remove(str(self.classes[idt]))
                self.class_count.pop(idt)
                
            if len(self.classes) == 1:
                self.classes = os.listdir(self.image_dir)
                self.class_count = np.zeros(len(os.listdir(self.image_dir))).tolist()
            
            selected_class_video = os.path.join(self.image_dir,str(self.classes[idt]))
            selected_class_mask =  os.path.join(self.mask_dir,str(self.classes[idt]))
            
           
            frame_ids = os.listdir(selected_class_video)
            ids = np.random.randint(1, len(frame_ids) - 10, self.batch_size)
            #print(ids)
            ct = 0
            for i in ids:
                     
                    im_org = cv2.imread(os.path.join(selected_class_video,frame_ids[i]))                    
                    im = np.array(im_org, dtype="float32") / 255.0 
                    #im.resize((self.img_height, self.img_width,im.shape[2]))
                    im = cv2.resize(im, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                   
    
                    mask_path = os.path.join(selected_class_mask,frame_ids[i-1])
                    mask_path = os.path.splitext(mask_path)[0]
                    mask_path = mask_path + ".png"                    
                    mask_org = cv2.imread(mask_path)
                    mask_org = cv2.resize(mask_org, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                   # mask_org.resize((self.img_height, self.img_width,mask_org.shape[2]))                                     
                    mask = cv2.cvtColor(mask_org, cv2.COLOR_BGR2GRAY)                                       
                    k = (np.unique(mask.flatten()))[0]
                    _,thresh = cv2.threshold(mask,(k+10),255,cv2.THRESH_BINARY)
                    x = np.random.randint(0,10,1)
                    y = np.random.randint(0,10,1)
                    M = np.float32([[1,0,x],[0,1,y]])
                    trans_thresh = cv2.warpAffine(thresh,M,( self.img_width,self.img_height))
                    trans_thresh =  np.array(trans_thresh,dtype = "float32")/255.0 #np.asarray(trans_thresh / 255.0, dtype=np.float32)                   
                    aug = A.ElasticTransform(p=1, alpha=40, sigma=40 * 0.05, alpha_affine=40 * 0.03)
                    np.random.seed(7)
                    augmented = aug(image=im, mask= trans_thresh)
                    mask_final = augmented['mask']
                   # mask_final = thresh
                    new = np.dstack((im, mask_final))
                   
                     
                    label_path = os.path.join(selected_class_mask,frame_ids[i])
                    label_path = os.path.splitext(mask_path)[0]
                    label_path = label_path + ".png"
                    label_org = cv2.imread(label_path)
                    label_org = cv2.resize(label_org, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                  #  label_org.resize((self.img_height, self.img_width,mask_org.shape[2]))                                     
                    label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)   
                    k = (np.unique(label.flatten()))[0]
                    _,im_label = cv2.threshold(label,(k+10),255,cv2.THRESH_BINARY) 
                    im_label = np.array(im_label,dtype = "float32")/255.0
                   
                                    
                    self.batch_y[ct, :, :, 0] = im_label[:, :]
                    self.batch_x[ct, :, :, :] = new[:, :, :]
                    ct += 1
           
          #  print("Train_done")
            return self.batch_x, self.batch_y
         
        
        elif   self.mode == "Valid" :
        
#Each Validation batch consists of 2nd last and third last frames of  from any of the video group. There will be 
#deisgnated no of batches as per batch count.     
       
            self.batch_x =  np.zeros((10, self.img_height, self.img_width, 4))
            self.batch_y = np.zeros((10, self.img_height, self.img_width, 1))
        
            idx = np.random.randint(0,len(self.classes)-1)
           
            
            selected_class_video = os.path.join(self.image_dir,str(self.classes[idx]))
            selected_class_mask =  os.path.join(self.mask_dir,str(self.classes[idx]))
            
            (self.classes).remove(str(self.classes[idx]))
            
            if len(self.classes) == 1:
                self.classes = os.listdir(self.image_dir)
           
            frame_ids = os.listdir(selected_class_video)
            ids = np.random.randint(1, len(frame_ids) - 3, 2)
            ct = 0
            for i in range(0,10):
                    
                    im_org = cv2.imread(os.path.join(selected_class_video,frame_ids[len(frame_ids)-10+i]))                    
                    im = np.array(im_org, dtype="float32") / 255.0 
                   # im.resize((self.img_height, self.img_width,im.shape[2]))
                    im = cv2.resize(im, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                    
                    mask_path = os.path.join(selected_class_mask,frame_ids[len(frame_ids)-11+i])
                    mask_path = os.path.splitext(mask_path)[0]
                    mask_path = mask_path + ".png"
                    mask_org = cv2.imread(mask_path)
                   # mask_org.resize((self.img_height, self.img_width,mask_org.shape[2]))  
                    mask_org = cv2.resize(mask_org, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                    mask = cv2.cvtColor(mask_org, cv2.COLOR_BGR2GRAY) 
                    k = (np.unique(mask.flatten()))[0]
                    _,thresh = cv2.threshold(mask,(k+10),255,cv2.THRESH_BINARY)
                    thresh =  np.array(thresh,dtype = "float32")/255.0
                    new = np.dstack((im, thresh))
                    
                    label_path = os.path.join(selected_class_mask,frame_ids[len(frame_ids)-10+i])
                    label_path = os.path.splitext(mask_path)[0]
                    label_path = label_path + ".png"
                    label_org = cv2.imread(label_path)
                    label_org = cv2.resize(label_org, (self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
                   # label_org.resize((self.img_height, self.img_width,mask_org.shape[2]))                                     
                    label = cv2.cvtColor(label_org, cv2.COLOR_BGR2GRAY)  
                    k = (np.unique(label.flatten()))[0]
                    _,im_label = cv2.threshold(label,(k+10),255,cv2.THRESH_BINARY)
                    im_label = np.array(im_label,dtype = "float32")/255.0
                    
                    self.batch_y[ct, :, :, 0] = im_label[:, :]
                    self.batch_x[ct, :, :, :] = new[:, :, :]
                    ct +=1
           
          #  print('Valid_done')
            return self.batch_x, self.batch_y
                

    def get_label(self):
        return self.batch_x, self.batch_y