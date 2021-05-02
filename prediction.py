import argparse
import os
from zipfile import ZipFile
from urllib.request import urlopen
import shutil
import pandas as pd
from time import time
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.optimizers import Adam
import csv
from keras.models import Model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras import backend as K
from skimage.io import imread
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
import requests
import tensorflow as tf
import os
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
import sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tkinter as tk
import tkinter.messagebox as msg
import pickle

from PIL import Image,ImageTk

CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']

CLASS_NAMES1 = ['滇刺枣',
               '马缨丹属',
               '扁轴木属',
               '银胶菊',
               '金合欢',
               '橡胶紫茉莉',
               '飞机草',
               '蛇杂草',
               '非本模型能识别的杂草']
model1 =tf.keras.models.load_model('D:\\biyesheji\\DeepWeeds-master\\models\\resnet.hdf5')
model2 =tf.keras.models.load_model('D:\\biyesheji\\DeepWeeds-master\\models\\inception.hdf5')
window = tk.Tk()
window.title('深度学习杂草识别-毕业论文设计')
window.geometry('600x400')

#D:\biyesheji\DeepWeeds-master\yanzhengtupian\20161207-142702-0.jpg


def resnet_prediction():
       address=address_name.get()
       img=imread(address)
       img = resize(img, (224, 224))
       img = np.expand_dims(img, axis=0)
       prediction = model1.predict(img,  batch_size=32, verbose=1)
       y_pred = np.argmax(prediction, axis=1) 
       msg.showinfo('Python', '预测的结果为:'+CLASS_NAMES1[y_pred[0]])

def inception_prediction():
       address=address_name.get()
       img=imread(address)
       img = resize(img, (224, 224))
       img = np.expand_dims(img, axis=0)
       prediction = model2.predict(img,  batch_size=32, verbose=1)
       y_pred = np.argmax(prediction, axis=1) 
       msg.showinfo('Python', '预测的结果为:'+CLASS_NAMES1[y_pred[0]])





canvas = tk.Canvas(window, width=600, height=200, bg='green')
image_file = ImageTk.PhotoImage(Image.open('D:\\biyesheji\\DeepWeeds-master\\download.jpg'))
image = canvas.create_image(300, 0, anchor='n', image=image_file)
canvas.pack(side='top')

tk.Label(window, text='欢迎来到杂草识别系统，请给出图片地址，并选择深度学习模式',font=('宋体', 16)).pack()

tk.Label(window, text='请输入图片地址:', font=('黑体', 16)).place(x=10, y=250)

address_name = tk.StringVar()
entry_address_name=tk.Entry(window, textvariable=address_name, font=('Arial', 14))
entry_address_name.place(x=200,y=250)
l1= tk.Button(window, text='Resnet50模式预测', font=('黑体', 12), width=20, height=1,activebackground='green',command=resnet_prediction).place(x=10,y=280)

l2= tk.Button(window, text='Inception-V3模式预测', font=('黑体', 12), width=20, height=1,activebackground='green',command=inception_prediction).place(x=200,y=280)



window.mainloop()