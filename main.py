import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

data_dir = 'face_classification'

# Verify data directory
if not os.path.exists(data_dir):
    raise ValueError(f"Data directory {data_dir} does not exist")

# Image extensions
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Clean up invalid images
for image_class in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, image_class)
    if not os.path.isdir(class_dir):
        continue
    for image in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'Image not in ext list {image_path}')
                os.remove(image_path)
        except Exception as e:
            print(f'Issue with image {image_path}')
            # os.remove(image_path)

