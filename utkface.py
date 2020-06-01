import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder 
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

utk_face = glob.glob('C:\\Users\\Nata\\Desktop\\dataset1\\UTKFace\\*.JPG')
print(len(utk_face))

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

#utk_img = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in utk_face]
utk_img=[]
for img in utk_face:
    img_open=load_img(img, target_size=IMG_DIM)
    utk_img.append(img_to_array(img_open))
    img_open.close()

print('UTK dataset shape:', utk_img.shape)

