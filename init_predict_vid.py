
# coding: utf-8

# In[1]:

from init_pipeline_vid import *

import imageio
#imageio.plugins.ffmpeg.download()

import cv2
import argparse
import colorsys
import imghdr
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
#%matplotlib inline

from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont


# In[6]:

def res():
    
    #test_path = os.path.expanduser('images')
    #output_path = os.path.expanduser('out')
    #for image_file in os.listdir(test_path):
        #try:
            #image_type = imghdr.what(os.path.join(test_path, image_file))
            #if not image_type:
                #continue
        #except IsADirectoryError:
            #continue
    
    #image = Image.open(os.path.join(test_path, image_file))
    #final = pipeline(image)
    #final.save(os.path.join(output_path, image_file), quality=90)
    project_video_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    lane_clip = clip1.fl_image(pipeline)
    #%time 
    lane_clip.write_videofile(project_video_output, audio=False)
    
    sess.close()
if __name__ == '__main__':
    res()

# In[ ]:



