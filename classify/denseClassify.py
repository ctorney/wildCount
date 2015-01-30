
import cv2
import cv
import numpy as np
import os, random, sys
import pickle
import scipy.ndimage.filters as filters

sys.path.append('../extractor/')

from circularHOGExtractor import circularHOGExtractor
import time


ch = circularHOGExtractor(8,2,4) 


photo_dir = '/home/ctorney/data/wildebeest_survey/'
counted_dir = '../countedImages/'


box_dim = 32


params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 20.0;
params.filterByInertia = False;
params.filterByConvexity = False;
params.filterByColor = False;
params.filterByCircularity = False;
params.filterByArea = True;
params.minArea = 25.0;
params.maxArea = 50.0;
params.blobColor = 0.0
b = cv2.SimpleBlobDetector(params)


gnb = pickle.load( open( "../boosters/contrastBooster.p", "rb" ) )
fhgb = pickle.load( open( "../boosters/fhgBooster.p", "rb" ) )

for imgName in os.listdir(photo_dir):
    
#   imgName = 'gnucount00241.jpg'
    imgName = random.choice(os.listdir(photo_dir)) #change dir name to whatever
    if os.path.isfile(counted_dir + 'c' + imgName):
        continue

    openframe = cv2.imread(photo_dir + imgName)
     
    frame = cv2.cvtColor(openframe, cv2.COLOR_BGR2GRAY)
    cleanframe = cv2.imread(photo_dir + imgName)
    img = frame.astype(np.float32)/255.0
    sys.stdout.write("calculating contrast ...")

    sys.stdout.flush()
    maxVals = filters.maximum_filter(img, box_dim)
    minVals = filters.minimum_filter(img, box_dim)
    (Nx,Ny) = np.shape(frame)
    contrast = np.zeros((Nx,Ny,1))
    contrast[:,:,0] = np.divide((maxVals - minVals),(maxVals+ minVals))
    sys.stdout.write("done\nclassifying based on contrast ... ")
    sys.stdout.flush()
    res = gnb.predict(contrast.reshape(Nx*Ny,1)).reshape(Nx,Ny,1)
    output = np.zeros_like(frame)
    output[res[:,:,0]>0.5]=255.0
    sys.stdout.write("done\nclassifying based on fourier HOG ...\n ")
    sys.stdout.flush()

    output2 = np.zeros_like(frame)
    for px in range(box_dim/2,Nx-box_dim/2):
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*px/float(Nx)), int(100.0*px/float(Nx))))
        sys.stdout.flush()
        for py in range(box_dim/2,Ny-box_dim/2):

            if output[px, py]  == 255.0:#*res[0,0]
                tmpImg = cleanframe[int(px)-box_dim/2:int(px)+box_dim/2, int(py)-box_dim/2:int(py)+box_dim/2]
                clsImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
                res = fhgb.predict(ch.extract(clsImg))
                if res[0]>0.5:
                    output2[px,py]  = 255.0
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20), int(100)))
    sys.stdout.flush()

    sys.stdout.write("\ndone\ncounting objects ... ")
    sys.stdout.flush()
    output2 = cv2.GaussianBlur(output2,(7,7),0)
    blob = b.detect(output2)
    counter = 1
    for beest in blob:
        cv2.putText(openframe,str(counter) ,((int(beest.pt[0])+20, int(beest.pt[1])+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
        cv2.circle(openframe, ((int(beest.pt[0]), int(beest.pt[1]))),12,255,2)

        counter = counter + 1
    sys.stdout.write("done\n\n")
    sys.stdout.flush()
    sys.stdout.write(imgName + " has %d wildebeest\n" % (counter))
    sys.stdout.flush()
    cv2.imwrite(counted_dir + 'c' + imgName, openframe)
#   break




