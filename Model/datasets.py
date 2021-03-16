import os
import posixpath
import utils
import struct

from PIL import Image
import numpy as np
import tensorflow as tf

from config import Config

class yaleb(object):

    def __init__(self, config):
        self.batch_size = config.batch_size

    def read_yaleb(self):
        
        
        train_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_data_small.npy')
        train_label = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_label_small.npy')
        
        val_data = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/val_data_small.npy')
        val_label= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/val_label_small.npy')

        # train_sub_1_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_1_data.npy')
        # train_sub_1_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_1_label.npy')
        
        # train_sub_2_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_2_data.npy')
        # train_sub_2_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_2_label.npy')
        
        # train_sub_3_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_3_data.npy')
        # train_sub_3_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_3_label.npy')
        
        # train_sub_4_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_4_data.npy')
        # train_sub_4_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_4_label.npy')
        
        # train_sub_5_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_5_data.npy')
        # train_sub_5_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_5_label.npy')
        
        # train_sub_6_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_6_data.npy')
        # train_sub_6_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_6_label.npy')
        
        # train_sub_7_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_7_data.npy')
        # train_sub_7_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/train_sub_7_label.npy')
        
        
        # val_sub_1_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/val_sub_1_data.npy')
        # val_sub_1_labels = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/val_sub_1_label.npy') 
        
        # val_sub_2_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/val_sub_2_data.npy')
        # val_sub_2_labels = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/val_sub_2_label.npy')        
        
        test_set = np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/test_data_small.npy')
        test_labels= np.load('/content/gdrive/MyDrive/colab/Face_detection/Data/test_label_small.npy')
        
        
        # train_sub_1_labels = train_sub_1_labels.astype('float32')
        # train_sub_2_labels = train_sub_2_labels.astype('float32')
        # train_sub_3_labels = train_sub_3_labels.astype('float32')
        # train_sub_4_labels = train_sub_4_labels.astype('float32')
        # train_sub_5_labels = train_sub_5_labels.astype('float32')
        # train_sub_6_labels = train_sub_6_labels.astype('float32')
        # train_sub_7_labels = train_sub_7_labels.astype('float32')
        
        # val_sub_1_labels = val_sub_1_labels.astype('float32')
        # val_sub_2_labels = val_sub_2_labels.astype('float32')
        
        return (train_set, train_label),(val_data, val_label),(test_set,test_labels)


        # return (train_sub_1_set, train_sub_1_labels),(train_sub_2_set, train_sub_2_labels),(train_sub_3_set, train_sub_3_labels),(train_sub_4_set, train_sub_4_labels),(train_sub_5_set, train_sub_5_labels),(train_sub_6_set, train_sub_6_labels),(train_sub_7_set, train_sub_7_labels), (val_sub_1_set, val_sub_1_labels),(val_sub_2_set, val_sub_2_labels), (test_set,test_labels)


    def get_dataset(self):
        
        train,val,test = self.read_yaleb()
        
        train_data = utils.convert_to_dataset(train, self.batch_size)
        test_data =  utils.convert_to_dataset(val, self.batch_size)

        # train1,val1,test = self.read_yaleb()
#       ,train4,train5,train6,train7,val2, 
        # Create tf Datasets for each.
        
        
        
        # train_data_1= utils.convert_to_dataset(train1, self.batch_size)
        # train_data_2= utils.convert_to_dataset(train2, self.batch_size)
        # train_data_3= utils.convert_to_dataset(train3, self.batch_size)
        # train_data_4= utils.convert_to_dataset(train4, self.batch_size)
        # train_data_5= utils.convert_to_dataset(train5, self.batch_size)
        # train_data_6= utils.convert_to_dataset(train6, self.batch_size)
        # train_data_7= utils.convert_to_dataset(train7, self.batch_size)

        # test_data_1 = utils.convert_to_dataset(val1, self.batch_size)
        # test_data_2 = utils.convert_to_dataset(val2, self.batch_size)
        
        # train_data= train_data_1.concatenate(train_data_2)
        # train_data= train_data.concatenate(train_data_3)
        # train_data= train_data.concatenate(train_data_4)
        # train_data= train_data.concatenate(train_data_5)
        # train_data= train_data.concatenate(train_data_6)
        # train_data= train_data.concatenate(train_data_7)
        
    
        # test_data= test_data_1.concatenate(test_data_2)
        
        # train_data = tf.concat([train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6, train_data_7], 0)
        
        # test_data = tf.concat([test_data_1, test_data_2], 0)
        
        # train_data = train_data.batch(self.batch_size, drop_remainder=True)
        # test_data = test_data.batch(self.batch_size, drop_remainder=True)

        return train_data, test_data
