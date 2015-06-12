
import cv2
import numpy as np
import os, random, sys
import pickle
import scipy.ndimage.filters as filters
from random import randint



def on_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        tmpImg = (img[y-box_dim/2:y+box_dim/2, x-box_dim/2:x+box_dim/2])
        cv2.imwrite(image_dir + imgName, tmpImg)
        os.rename(image_dir + imgName, final3000 + imgName)
        
                        
image_dir = './images/'
final3000 = './yes3000/'
no3000 = './no3000/'

box_dim = 32



cv2.destroyAllWindows()
frname = 'wildebeest? y or n'
cv2.namedWindow(frname, flags =  cv2.WINDOW_NORMAL)
cv2.setMouseCallback(frname,on_click)
for imgName in os.listdir(image_dir):
    

    imgName = random.choice(os.listdir(image_dir)) #change dir name to whatever
    img = cv2.imread(image_dir + imgName)
    x = int(img.shape[0]*0.5)
    y = int(img.shape[0]*0.5)
    img2 = np.copy(img)
    if img.size < box_dim*box_dim:
        continue
    cv2.circle(img2, (x,y),1,255,-1)
    cv2.imshow(frname, img2)        
    k = cv2.waitKey(2500)
    if k==27:    # Esc key to stop
        break
    elif k==ord('y'):
        tmpImg = (img[y-box_dim/2:y+box_dim/2, x-box_dim/2:x+box_dim/2])
        cv2.imwrite(image_dir + imgName, tmpImg)
        os.rename(image_dir + imgName, final3000 + imgName)
    elif k==ord('n'):
        tmpImg = (img[y-box_dim/2:y+box_dim/2, x-box_dim/2:x+box_dim/2])
        cv2.imwrite(image_dir + imgName, tmpImg)
        os.rename(image_dir + imgName, no3000 + imgName)
                
