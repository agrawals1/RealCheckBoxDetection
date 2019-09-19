import cv2
import os

#from PIL import Image

 

# Create an Image object from an Image

def cropMe(im):

 

# Crop the iceberg portion
    cropped = im[3:344,15:139].copy()
    #cropped = im.crop((166,84,322,439))(y, y+h, x, x+w)
    return cropped
 

for file in os.listdir(r"./data/splitImages/val/class-0"):
    
    cropped = cropMe(cv2.imread(r"./data/splitImages/val/class-0/{}".format(file)))
    cv2.imwrite(r"./data/splitImagescropped/val/class-0/{}".format(file), cropped)
    