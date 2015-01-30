
import cv2
import cv
import numpy as np
import os, random, sys
import pickle
import scipy.ndimage.filters as filters

import time
sys.path.append('../extractor/')

from cudaHOGExtractor import cudaHOGExtractor
import time


ch = cudaHOGExtractor(8,2,4) 


photo_dir = '/home/ctorney/data/wildebeest_survey/'
counted_dir = '../countedImages/'


box_dim = 32


params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 20.0;
params.filterByInertia = False;
params.filterByConvexity = False;
params.minThreshold = 5;
params.maxThreshold = 255;
params.thresholdStep = 5.0;
params.filterByColor = False;
params.filterByCircularity = False;
params.filterByArea = True;
params.minArea = 20.0;
params.maxArea = 50.0;
params.blobColor = 0.0

b = cv2.SimpleBlobDetector(params)


gnb = pickle.load( open( "../boosters/contrastBooster.p", "rb" ) )
fhgb = pickle.load( open( "../boosters/fhgBooster.p", "rb" ) )

for imgName in os.listdir(photo_dir):
    
    start_time = time.time()
    sys.stdout.write("Processing image " + imgName + " \n=====================================\n" )
    sys.stdout.flush()
#imgName = random.choice(os.listdir(photo_dir)) #change dir name to whatever
#    if os.path.isfile(counted_dir + 'c' + imgName):
#        continue

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
    check  = np.nonzero(output)
    positions=np.transpose(np.vstack((check[0],check[1])))
    plen = np.size(positions,0)

    chunkSize = 8192#65536#4096
    splitPos = np.array_split(positions,np.arange(chunkSize,plen,chunkSize))
    
    ch.prepareExtract(frame)    
    output2 = np.zeros_like(frame)
    nsplits = np.size(splitPos,0)
    for k in range(nsplits):
        thisPos = np.copy(splitPos[k],order='c')
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*k/float(nsplits)), int(100.0*k/float(nsplits))))
        sys.stdout.flush()
        N = np.size(thisPos,0)
        features = ch.denseExtract(frame, thisPos, N)    
        res = fhgb.predict(features)
#       output2[thisPos[res>0.5]]=255.0
        output2[thisPos[res>0.5,0],thisPos[res>0.5,1]]=255.0
#       break
#        for pxs in range(N):
#           print pxs
#            res = fhgb.predict(features[pxs])
#            if res[0]>0.5:
#                px = thisPos[pxs,0]
#                py = thisPos[pxs,1]
#                output2[px,py]  = 255.0

#    break
#   print thisPos[000,0], thisPos[000,1]
#    px = thisPos[000,0]+1
#    py = thisPos[000,1]+1
    
#   tmpImg = cleanframe[int(px)-box_dim/2:int(px)+box_dim/2, int(py)-box_dim/2:int(py)+box_dim/2]
#    clsImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
#    features2 = ch.extract(clsImg)
#    for i in range(np.size(features2,0)):
#        print features[000,i], features2[i]
        
#    break
#    for px in range(box_dim/2,Nx-box_dim/2):
#        sys.stdout.write('\r')
#        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*px/float(Nx)), int(100.0*px/float(Nx))))
#        sys.stdout.flush()
#        for py in range(box_dim/2,Ny-box_dim/2):
#
#            if output[px, py]  == 255.0:#*res[0,0]
#                tmpImg = cleanframe[int(px)-box_dim/2:int(px)+box_dim/2, int(py)-box_dim/2:int(py)+box_dim/2]
#                clsImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
#                res = fhgb.predict(ch.extract(clsImg))
#                if res[0]>0.5:
#                    output2[px,py]  = 255.0
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20), int(100)))
    sys.stdout.flush()

    sys.stdout.write("\ndone\ncounting objects ... ")
    sys.stdout.flush()
    output2 = cv2.GaussianBlur(output2,(7,7),0)
    blob = b.detect(output2)
    counter = 0
    for beest in blob:
        counter = counter + 1
        cv2.putText(openframe,str(counter) ,((int(beest.pt[0])+20, int(beest.pt[1])+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
        cv2.circle(openframe, ((int(beest.pt[0]), int(beest.pt[1]))),12,255,2)

    sys.stdout.write("done\n\n")
    sys.stdout.flush()
    sys.stdout.write(imgName + " has %d wildebeest\n" % (counter))
    sys.stdout.flush()
#   cv2.imwrite(counted_dir + '2c' + imgName, output2)
    cv2.putText(openframe,str(counter) + ' wildebeest found in this image in %ds' % (time.time() - start_time),((100, 200)), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(255,255,255),2)

    cv2.imwrite(counted_dir + 'c' + imgName, openframe)
#   break




