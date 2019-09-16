import cv2

#from PIL import Image

 

# Create an Image object from an Image

def cropMe(im):

 

# Crop the iceberg portion
    cropped = im[84:439,166:322].copy()
    #cropped = im.crop((166,84,322,439))
    return cropped
 

# Display the cropped portion

