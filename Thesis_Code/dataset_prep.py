# Import needed packages
import gc; gc.enable() # memory is tight
import cv2#
import numpy as np#
from pandas import read_csv,DataFrame
from os import listdir#
from os.path import isfile, join#
from sklearn.model_selection import train_test_split#
from tensorflow.random import set_seed 
from itertools import chain
"""
   Input Image Ready
"""

def set_seed(SEED):
    np.random.seed(SEED)
    set_seed(SEED)
    return SEED

def get_labels(label_filename):
    data =  read_csv(join(train_image_dir_l,label_filename), sep=" ", 
                       names=['label', 'truncated', 'occluded', 'alpha', 
                              'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 
                              'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 
                              'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score'])
    
    return data

def create_mask(mask_dir, img_shape):
    mask = np.zeros(shape=(img_shape[0], img_shape[1], 1))
    with open(mask_dir) as f:
        content = f.readlines()
    content = [x.split() for x in content] 
    for item in content:
        if (item[0]=='Pedestrian') | (item[0] =='Person_sitting') | ( item[0] =='Cyclist'):
            ul_col, ul_row = int(float(item[4])), int(float(item[5]))
            lr_col, lr_row = int(float(item[6])), int(float(item[7]))            
            mask[ul_row:lr_row, ul_col:lr_col, 0] = 1 
    return mask

'''make batchs of images
'''
def create_images_generator(df_in, resized_shape):
    """
    Creates imput image dataset
    
    Input: df_train of shape (#,m,n,c)
    
    Output: (output batch image of shape (batch size,256,256,3), output masks of shape (batch size,256,256,1))
    
    """
    batch_image = []
    batch_mask = []
    df_in_list = (df_in).values.tolist()
    np.random.shuffle(df_in_list)  
    
    while True:
        for image_path, mask_path in df_in_list:
            image_r = cv2.imread(image_path)
            mask_r = create_mask(mask_path, image_r.shape)
            
            image_r = cv2.resize(image_r,(resized_shape[1], resized_shape[0]))
            mask_r = cv2.resize(mask_r,(resized_shape[1], resized_shape[0]))
            
            batch_image.append(image_r)
            batch_mask.append(mask_r)
        return np.array(batch_image), np.expand_dims(np.array(batch_mask),axis=-1)  
    
def train_test_index(mask,test_size):
    m,H,W,_=mask.shape
    V=np.sum(mask.reshape(m,H*W)/(H*W),axis=1);
    np.random.shuffle(V)
    Seg=np.linspace(0,np.max(V),3);
    V1=np.where(V==0)[0];
    V2=np.where(((V>0) & (V<=Seg[1])))[0];
    V3=np.where(((V>Seg[1]) & (V<=Seg[2])))[0];

    V1_test=V1[:int(len(V1)*test_size)]
    V1_train=V1[int(len(V1)*test_size):]

    V2_test=V2[:int(len(V2)*test_size)]
    V2_train=V2[int(len(V2)*test_size):]

    V3_test=V3[:int(len(V3)*test_size)]
    V3_train=V3[int(len(V3)*test_size):]

    V_train=[V1_train, V2_train, V3_train];
    V_test=[V1_test, V2_test, V3_test];

    Train=list(chain.from_iterable(V_train));
    Test=list(chain.from_iterable(V_test));
    return Train, Test

def input_data_ready(train_image_dir_l='C:/Users/s_t392/Downloads/Thesis/New_Image/labels',
                     train_image_dir = 'C:/Users/s_t392/Downloads/Thesis/New_Image/images/',test_size=0.25, 
                     BATCH_SIZE=32, resized_shape=(224,224)):
    #Filename of the image
    images =  [(train_image_dir+f) for f in listdir(train_image_dir) if isfile(join(train_image_dir, f))]
    masks = [(train_image_dir_l+f) for f in listdir(train_image_dir_l) if isfile(join(train_image_dir_l, f))]
    df = DataFrame(np.column_stack([images, masks]), columns=['images', 'masks'])
    # Create Dataset for training
    image, mask  = create_images_generator(df, resized_shape=resized_shape);
    Train, Test = train_test_index(mask,test_size)
    Train_image, Train_mask=image[Train], mask[Train];
    Test_image, Test_mask=image[Test], mask[Test];
    return Train_image, Train_mask, Test_image, Test_mask    