
import cv2
import numpy as np
import os, random, sys
import pickle
import scipy.ndimage.filters as filters
from random import randint

import time




image_dir = './images/'
weed_in = './check_yes/'
weed_out = './check_no/'


box_dim = 32




cv2.destroyAllWindows()
frname = 'wildebeest? y or n'
cv2.namedWindow(frname, flags =  cv2.WINDOW_NORMAL)
for imgName in os.listdir(image_dir):
    

    img = cv2.imread(image_dir + imgName)
    if img.size < box_dim*2:
        continue
    cv2.imshow(frname, img)        
    k = cv2.waitKey(1000)
    if k==27:    # Esc key to stop
        break
    elif k==ord('n'):
        os.rename(image_dir + imgName, weed_out + imgName)
    else:
        os.rename(image_dir + imgName, weed_in + imgName)

                



