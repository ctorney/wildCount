
import cv2
import numpy as np
import os, random, sys
import pickle
import scipy.ndimage.filters as filters
from random import randint

sys.path.append('../extractor/')

from circularHOGExtractor import circularHOGExtractor
import time

offset = str(randint(0,1000))
ch = circularHOGExtractor(8,2,4) 


def scanSave(x,y,save_path):
    scan = 4
    startx = max(x-scan,0)
    stopx = min(x+scan,Nx)
    starty = max(y-scan,0)
    stopy = min(y+scan,Ny)
    tmpImg = (cleanframe[y-box_dim/2:y+box_dim/2, x-box_dim/2:x+box_dim/2])
    if ((tmpImg.shape[0] + tmpImg.shape[1]) == 2*box_dim):
        clsImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
        res = fhgb.predict_proba(ch.extract(clsImg))
        maxVal = res[0,1]
        bx = x
        by = y
        for ix in range(startx,stopx):
            for iy in range(starty,stopy):
                tmpImg = (cleanframe[iy-box_dim/2:iy+box_dim/2, ix-box_dim/2:ix+box_dim/2])
                if ((tmpImg.shape[0] + tmpImg.shape[1]) == 2*box_dim):
                    clsImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
                    res = fhgb.predict_proba(ch.extract(clsImg))
                    thisVal = res[0,1]
                    if thisVal>maxVal:
                        maxVal = thisVal
                        bx = ix
                        by = iy
        tmpImg = (cleanframe[by-box_dim/2:by+box_dim/2, bx-box_dim/2:bx+box_dim/2])
        cv2.imwrite(save_path, tmpImg)
 
def is_wildebeest(event,x,y,flags,param):
    global counter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        save_path = "yes2/img-" + offset + str(counter) + ".png"
        scanSave(x,y,save_path)
        counter += 1
        
                        
def isnt_wildebeest(event,x,y,flags,param):
    global counter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        save_path = "no2/img-" + offset + str(counter) + ".png"
        tmpImg = (cleanframe[y-box_dim/2:y+box_dim/2, x-box_dim/2:x+box_dim/2])
        cv2.imwrite(save_path, tmpImg)
        counter += 1



photo_dir = '/home/ctorney/data/wildebeest_survey/2012/'
counted_dir = '../countedImages/'


box_dim = 32


#params = cv2.SimpleBlobDetector_Params()
#params.minDistBetweenBlobs = 20.0;
#params.filterByInertia = False;
#params.filterByConvexity = False;
#params.filterByColor = False;
#params.filterByCircularity = False;
#params.filterByArea = True;
#params.minArea = 25.0;
#params.maxArea = 50.0;
#params.blobColor = 0.0
#b = cv2.SimpleBlobDetector(params)


counter = 0
fhgb = pickle.load( open( "../boosters/fhgBooster.p", "rb" ) )

endClass = False
nos=True
yeses=False
cv2.destroyAllWindows()
if nos:
    noname = 'dbl click miss-identified wildebeest (ESC to quit, c to continue)'
    cv2.namedWindow(noname, flags =  cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(noname, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
for imgName in os.listdir(photo_dir):
    

#imgName = random.choice(os.listdir(photo_dir)) #change dir name to whatever
    cleanframe = cv2.imread(photo_dir + imgName)
    (Nx,Ny,Nz) = np.shape(cleanframe)
    print imgName
    if not os.path.isfile(counted_dir + 'c' + imgName):
        continue
    countframe = cv2.imread('../countedImages/c' + imgName)
    if yeses:
        yesname = imgName + ': dbl click wildebeest (ESC to quit, c to continue)'
        cv2.namedWindow(yesname, flags =  cv2.WINDOW_NORMAL)
        #    cv2.setWindowProperty(yesname, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        cv2.resizeWindow(yesname, 4*Nx,4*Ny);
        cv2.setMouseCallback(yesname,is_wildebeest)
        cv2.imshow(yesname,countframe)
        while(1):
            k = cv2.waitKey(0)
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
        cv2.destroyAllWindows()
        if endClass: break
    if nos:
        #    cv2.setWindowProperty(noname, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        cv2.setMouseCallback(noname,isnt_wildebeest)
        cv2.imshow(noname,countframe)
        while(1):
            k = cv2.waitKey(0)
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
        if endClass: break
                



