
import cv2
import cv
import numpy as np
import os, random, sys
import pickle

sys.path.append('./fhgClassifier/')

from circularHOGExtractor import circularHOGExtractor
import numpy as np
import time


ch = circularHOGExtractor(3,4,2) 
nFeats = ch.getNumFields()


photo_dir = '/home/ctorney/data/wildebeest_survey/'


box_dim = 32




gnb = pickle.load( open( "../adaBoostClass.p", "rb" ) )

for imgName in os.listdir(photo_dir):
    imgName = 'gnucount00235.jpg'

    openframe = cv2.imread(photo_dir + imgName)
     
    frame = cv2.cvtColor(openframe, cv2.COLOR_BGR2GRAY)

    split = 4
    (Nx,Ny) = np.shape(frame)
    sNx =int( Nx/float(split))
    sNy = int(Ny/float(split))
    output = np.zeros_like(frame)
    sfx = 0
    sfy = 0
    subFrame = frame[sfx*sNx:(1+sfx)*(sNx), sfy*sNy:(sfy+1)*sNy]
#features = np.zeros((int(Nx*0.5),int(Ny*0.5),nFeats), dtype=float)

    x = int(0.15*Nx)
    y = int(0.12*Ny)
    positions = np.array([(int(0.15*Nx)-1, int(0.12*Ny)-1)])
    aaa = np.nonzero(a)
    a=np.transpose(np.vstack((aaa[0],aaa[1])))

    N = np.size(positions,0)
    
    print x,y
    print 'extracting features from sector ' + str(sfx) + ','+ str(sfy) + ' ...'
    ch.prepareExtract(frame)    
    features = ch.denseExtract(frame, positions, N)    
    
    tmpImg = frame[int(x)-box_dim/2:int(x)+box_dim/2, int(y)-box_dim/2:int(y)+box_dim/2]
    res = ch.extract(tmpImg)
    for i in range(nFeats): print features[0,i], res[i]


    break




