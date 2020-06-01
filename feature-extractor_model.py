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

man_train = glob.glob('C:\\Users\\Nata\\Desktop\\dataset1\\dataset1\\train\\man\\*.JPG')
woman_train = glob.glob('C:\\Users\\Nata\\Desktop\\dataset1\\dataset1\\train\\woman\\*.JPG')

man_val=glob.glob('C:\\Users\\Nata\\Desktop\\dataset1\\dataset1\\valid\\man\\*.JPG')
woman_val=glob.glob('C:\\Users\\Nata\\Desktop\\dataset1\\dataset1\\valid\\woman\\*.JPG')

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

#Images-train
men_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in man_train]
women_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in woman_train]
men_imgs = np.array(men_imgs)
women_imgs = np.array(women_imgs)
train_imgs=np.append(men_imgs,women_imgs,axis = 0)
print('Men dataset shape:', men_imgs.shape)
print('Women dataset shape:', women_imgs.shape)
print('Train dataset shape:', train_imgs.shape)

#Images-val
men_val_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in man_val]
women_val_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in woman_val]
men_val_imgs = np.array(men_val_imgs)
women_val_imgs = np.array(women_val_imgs)
val_imgs=np.append(men_val_imgs,women_val_imgs,axis = 0)
print('Val Men dataset shape:', men_val_imgs.shape)
print('Val Women dataset shape:', women_val_imgs.shape)
print('Validation dataset shape:', val_imgs.shape)

#Labels-train
man_labels = ['men' for fn in man_train]
women_labels=['women' for wn in woman_train]
train_labels=man_labels+women_labels
print('Train labels dataset shape:', len(train_labels))

#Labels-val
man_val_labels = ['men' for fn in man_val]
women_val_labels=['women' for wn in woman_val]
val_labels=man_val_labels+women_val_labels
print('Validation labels dataset shape:', val_labels)

train_imgs_scaled = train_imgs.astype('float32') 
validation_imgs_scaled = val_imgs.astype('float32') 
train_imgs_scaled /= 255 
validation_imgs_scaled /= 255 
 
# visualize a sample image 
print(men_imgs[0].shape) 

# encode text category labels 
le = LabelEncoder() 
le.fit(train_labels)
# le.fit(val_labels)
train_labels_enc = le.transform(train_labels) 
validation_labels_enc = le.transform(val_labels) 
 
print(train_labels[1:50], train_labels_enc[1:50])

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')

img_id = 50
men_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1], batch_size=1) 
men = [next(men_generator) for i in range(0,5)] 
fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in men]) 
l = [ax[i].imshow(men[i][0][0]) for i in range(0,5)]

img_id = 1000 
women_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1], batch_size=1) 
women = [next(women_generator) for i in range(0,5)] 
fig, ax = plt.subplots(1,5, figsize=(15, 6)) 
print('Labels:', [item[1][0] for item in women]) 
l = [ax[i].imshow(women[i][0][0]) for i in range(0,5)]

train_generator = train_datagen.flow(train_imgs, train_labels_enc,batch_size=30)
val_generator = val_datagen.flow(val_imgs, validation_labels_enc, batch_size=30)


restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(300,300,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()


model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(300,300,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, verbose=1)

model.save('men_women_tlearn_img_aug_cnn_restnet50.h5')                             