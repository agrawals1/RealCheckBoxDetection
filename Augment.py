import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import xml.etree.ElementTree as ET
import os
import glob
import pandas as pd

def parseXML(xmlfile):

    tree = ET.parse(r"./data/BB_data/"+xmlfile)
    root = tree.getroot()
    allchecks= []
    for item in root.findall('./object'):
        for child in item:
            if child.tag == 'bndbox':
                allchecks.append([int(child[0].text), int(child[1].text), int(child[2].text), int(child[3].text)])
       
    return allchecks


def convert_labels(image_aug, aug_coords):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (Xmin, Ymin, Xmax, Ymax) annotation format to
        (x, y, width, height) normalized YOLO format.
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    
    size = image_aug.shape
    if(size[1]==None):
        print(path, "not found")
        return '', '', '', ''
    
        
    f = open(r"./data/afterAugBB/yolo/{}.txt".format("image" + str(counter_for_image) + "iteration" + str(counter_for_iteration)), "a")            
    for element in aug_coords:
        x1 = element[0]
        y1 = element[1]
        x2 = element[2]
        y2 = element[3]
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1./size[1]
        dh = 1./size[0]
        x = (x1 + x2)/2.0
        y = (y1 + y2)/2.0
        w = x2 - x1
        h = y2 - y1
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        cls = 0
        f.write("{} {} {} {} {} \n".format(cls, x, y, w, h))
    f.close()
    
        

def Augmentors(image, bbs):
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                rotate=(-5, 5), # rotate by -45 to +45 degrees
                shear=(5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            #execute 0 to 5 of the following (less important) augmenters per image
            #don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 2),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.5)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 4)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 0.5), lightness=(0.15, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.5)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0))
                     ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    #iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-5, 5), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-5, 5)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    #iaa.OneOf([
                        # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        # iaa.FrequencyNoiseAlpha(
                            # exponent=(-4, 0),
                            # first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            # second=iaa.ContrastNormalization((0.5, 2.0))
                        # )
                    # ]),
                    #iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
   
    try:        
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        return image_aug, bbs_aug
    except:
        print("caught")  
        return "caught", "caught"



if __name__ == "__main__":
    counter_for_image = 0
    xml_files = []
    png_files = []
    image_list = os.listdir(r"./data/ScannedImages")
    bb_list = os.listdir(r"./data/BB_data")
    for bb_file in bb_list:
        if bb_file.endswith(".xml"):
            xml_files.append(bb_file[:-4])
    for filename in image_list:
        if filename.endswith(".png"):
            png_files.append(filename[:-4])           
    for pngfile in png_files:
        for xmlfile in xml_files:
            if (pngfile == xmlfile):
                checkboxes = parseXML(xmlfile + ".xml")
                counter_for_image +=1
                counter_for_iteration = 0
                for i in range(50):
                    counter_for_iteration +=1
                    image = imageio.imread("./data/ScannedImages/{}".format(pngfile + ".png"))
                    bblist = []                        
                    for element in checkboxes:
                        bblist.append(BoundingBox(x1=element[0], x2=element[2], y1=element[1], y2=element[3], label='0'))
                                
                            bbs = BoundingBoxesOnImage(bblist, shape=image.shape)
                            try:
                                imageaug, bbs_aug = Augmentors(image, bbs)
                                aug_coords = (bbs_aug.to_xyxy_array())
                            except:
                                continue    #csv files ka format is xmin ymin xmax ymax whereas txt file (yolo) ka format is x y width height                      
                            aug_coords_class = np.insert(aug_coords, 0, 0, axis=1)                          
                            aug_df = pd.DataFrame(data=aug_coords_class)
                            aug_df.to_csv(r"./data/afterAugBB/xminymin/{}.csv".format("image" + str(counter_for_image) + "iteration" + str(counter_for_iteration)), index=False, sep=" ", header=False)
                            convert_labels(imageaug, aug_coords)
                            if(imageaug == "caught"):
                                continue                            
                            
                            
                            try :                                
                                imageio.imwrite("./data/augImgs/" + "image" + str(counter_for_image) + "iteration" + str(counter_for_iteration) + ".png", (imageaug, size=2))
                            except:
                                print("skipping.....")
                                print(pngfile)
                                continue

                            
                        
                        

        
