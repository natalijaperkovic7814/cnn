# load_model_sample.py
from keras.models import load_model
from keras.preprocessing.image import load_img
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import ExifTags
import cv2
import argparse
from tkinter import *
import glob
from decimal import Decimal



def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(300, 300))
    img_tensor = image.img_to_array(img)
    img_tensor = np.array(img_tensor)
    # img_tensor_scaled = img_tensor.astype('float32')                     # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    # img_tensor_scaled /= 255.
    # cv2.imshow('img_tensor', img_tensor)
    # cv2.waitKey()                                     # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def predict(folder_path):

    dictImage={}
    for image in os.listdir(folder_path):

        img_path = folder_path+image
        # print(img_path)
        new_image = load_image(img_path)
        pred = model.predict(new_image)
        model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        dictImage[image]=pred.item(0)

    return dictImage

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def detectCropAndSave(imageName):

    #detect face, crop and save images
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = cv2.imread('GroupImages/'+imageName)
    # img = cv2.resize(img, (1366, 868))
    filelist = glob.glob(os.path.join("CroppedImages/", "*.jpg"))
    for f in filelist:
        os.remove(f)
    clone = img.copy() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-8, y-10), (x+w, y+h), (255, 0, 0), 2)
        crop_img = clone[y-10:y+10+h, x-8:x+8+w] 
        name = str(x-8) + "_" + str(y-10) + ".jpg"
        cv2.imwrite("CroppedImages/" +name, crop_img)
 

    cv2.namedWindow('Amanda', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Amanda', img)
    cv2.waitKey()
    return img



if __name__ == "__main__":

    img=detectCropAndSave('people.jpg')

    # load model
    model = tf.keras.models.load_model('men_women_tlearn_img_aug_cnn_restnet50.h5', compile=False)

    path='CroppedImages/'
    dict_im=predict(path)


    for key in dict_im:
        loc=key.split("_")

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(loc[0]),int(loc[1][:-4]))
        fontScale              = 0.6
        fontColor              = (0,0,240)
        lineType               = 2

        gender=''
        if dict_im[key]<=0.5:
            gender='men -> '+str(round(dict_im[key],4))
        else:
            gender='women -> '+str(round(dict_im[key],4))

        cv2.putText(img,gender, bottomLeftCornerOfText, font, fontScale,fontColor, lineType)

    cv2.namedWindow('Amanda', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Amanda', img)
    cv2.waitKey()



    # image path
    # img_path = 'CroppedImages/121_355.jpg'    # men
    #img_path = 'dataset1/test/woman/face_2.jpg'   # women
    
    # load a single image
    # new_image = load_image(img_path)

    # check prediction 
    # pred = model.predict(new_image)

    # print(pred)