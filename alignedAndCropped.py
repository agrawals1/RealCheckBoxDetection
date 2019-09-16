import cv2
import imutils
import Middlecrop as crop
import align_v1 as align
import os

ref_image = cv2.imread("scanned_final_2.png", cv2.IMREAD_COLOR)   
for i, file in enumerate(os.listdir(r"./data/ScannedImages")):

    orgimage = cv2.imread("./data/ScannedImages/{}".format(file), cv2.IMREAD_COLOR)
    orgimage = imutils.resize(orgimage, height= 700)
    aligned, h = align.alignImages((orgimage), (ref_image))
    alignedAndCropped = crop.cropMe(aligned)
    cv2.imwrite("./data/alignedAndCropped{}.png".format(i), alignedAndCropped)      