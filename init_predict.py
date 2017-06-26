
# coding: utf-8

# In[1]:

from init_pipeline import *

import argparse
import colorsys
import imghdr
import os
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont


# In[6]:

def res(image_file):
    
    test_path = os.path.expanduser('images')
    output_path = os.path.expanduser('out')
    #for image_file in os.listdir(test_path):
        #try:
            #image_type = imghdr.what(os.path.join(test_path, image_file))
            #if not image_type:
                #continue
        #except IsADirectoryError:
            #continue
    
    image = Image.open(os.path.join(test_path, image_file))
    final = pipeline(image)
    final.save(os.path.join(output_path, image_file), quality=90)
    
    #sess.close()

# In[ ]:



