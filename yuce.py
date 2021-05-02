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
print(tf.__version__)

LABEL_DIRECTORY = "./labels/"
MODEL_DIRECTORY = "./models/"
MODEL_GD_ID = "1MRbN5hXOTYnw7-71K-2vjY01uJ9GkQM5"
IMG_GD_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"




# Global variables

CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']


IMG_DIRECTORY = r'D:\\biyesheji\\DeepWeeds-master\\yanzhengtupian'
OUTPUT_DIRECTORY = "./outputs/"

test_path = r'D:\\biyesheji\\DeepWeeds-master\\yanzhengtupian'

files = os.listdir(test_path)
os.chdir(test_path)


weight_path=r'D:\\biyesheji\\DeepWeeds-master\\models\\resnet.hdf5'
# Measure the speed of performing inference with the chosen model averaging over DeepWeeds images

model =tf.keras.models.load_model(weight_path)


for file in files:
        # Load image
        img = imread(file)
        img = resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)# Map  batch
        prediction = model.predict(img,  batch_size=32, verbose=0)
        y_pred = np.argmax(prediction, axis=1)      
        print(prediction)  
                # Scale from int to float
                #img = img * 1./255
                #preprocessing_time = time() - start_time
                #start_time = time()
                # Predict label
                #prediction = model.predict(img, batch_size=1, verbose=0)
        
                #y_pred = np.argmax(prediction, axis=1)
                #y_pred[np.max(prediction, axis=1) < 1/9] = 8
                #inference_time = time() - start_time
                # Append times to lists
                #preprocessing_times.append(preprocessing_time)
                #inference_times.append(inference_time)






