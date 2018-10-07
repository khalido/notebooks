from tqdm import tqdm, tqdm_notebook
import os
import numpy as np
from PIL import Image, ImageOps
from IPython.display import HTML, display, clear_output
from tqdm import tnrange, tqdm_notebook
import cv2


def make_square_by_padding(im, pad_color=(0,155,255)):
    """takes in a image, pads it to make it square, returns it"""
    if im.height == im.width:
        return im
    pad = abs(im.height - im.width)
    
    if im.width < im.height:
        padding = (0, 0, pad, 0) # padding is LTRB
    else:
        padding = (0, pad, 0, 0)
        
    return ImageOps.expand(im, padding, pad_color)


def pre_process_image(file_name):
    """takes in a file_name, opens it and returns the preprocessed image"""
    im = Image.open(file_name)
    
    # check that image is RGB
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    # pad the image so its 1:1
    im = make_square_by_padding(im, pad_color=(255,255,255))
    
    # reshape to 128x128x3
    im = im.resize((128,128))
    
    # contrast stretch
    im = ImageOps.autocontrast(im)

    return im

def preprocess_all_images_in_dir(img_dir="gear_images/", 
                              processed_dir="gear_images_processed/", 
                              show_images=False):
    """processes all the images in the subdirectories of the passed in img_dir
       and saves them in subdirectories of the same name in the processed_dir"""

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for sub_dir in tqdm_notebook(os.listdir(img_dir)):
        cur_dir = img_dir + sub_dir
        new_dir = processed_dir + sub_dir

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for img in os.listdir(cur_dir):
            im = pre_process_image(cur_dir +"/"+ img)
            if show_images:
                clear_output(wait=True)
                display(im)
            im.save(new_dir + "/" + img)
            
def read_all_images_in_dir(img_dir="gear_images_processed/"):
    """processes all the images in the subdirectories of the passed in img_dir
       and return a list of images and their labels."""

    images = []
    label_names = []
    labels = []
    
    for i, sub_dir in tqdm_notebook(enumerate(os.listdir(img_dir))):
        cur_dir = img_dir + sub_dir
        label = sub_dir
        label_names.append(label)
        
        for img in os.listdir(cur_dir):
            im = cv2.imread(cur_dir +"/"+ img)
            images.append(im.flatten())
            labels.append(i)
    
    return np.array(images), np.array(labels), label_names