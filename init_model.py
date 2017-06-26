
# coding: utf-8

# In[1]:

import argparse
import colorsys
import imghdr
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head


# In[2]:

#test_path = os.path.expanduser('videos')
output_path = os.path.expanduser('out')
sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
model_path = os.path.expanduser('model_data/yolo.h5')
assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
anchors_path = os.path.expanduser('model_data/yolo_anchors.txt')
classes_path = os.path.expanduser('model_data/coco_classes.txt')

with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(anchors_path) as f:
    anchors = f.readline()
    try:
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    except ValueError:
        pass
    

yolo_model = load_model(model_path)

# Verify model, anchors, and classes are compatible
num_classes = len(class_names)
num_anchors = len(anchors)
# TODO: Assumes dim ordering is channel last
model_output_channels = yolo_model.layers[-1].output_shape[-1]
assert model_output_channels == num_anchors * (num_classes + 5),     'Mismatch between model and given anchor and class sizes. '     'Specify matching anchors and classes with --anchors_path and '     '--classes_path flags.'
print('{} model, anchors, and classes loaded.'.format(model_path))


if not os.path.exists(output_path):
    print('Creating output path {}'.format(output_path))
    os.mkdir(output_path)


# In[ ]:



