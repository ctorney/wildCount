import sys, os
import numpy as np
import cv2
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

sys.path.append('../extractor/')

from circularHOGExtractor import circularHOGExtractor

ch = circularHOGExtractor(8,2,4) 

cls0 = './no3000/'
cls1 = './yes3000/'
#lst0 = [name for name in os.listdir(cls0)] 
#lst0c = [name for name in os.listdir(cls0c)]
#lst1 = [name for name in os.listdir(cls1)]# if os.path.isfile(name)]
lst0all = [name for name in os.listdir(cls0) if not name.startswith('.')] 
lst1all = [name for name in os.listdir(cls1) if not name.startswith('.')]


sampleSizes = np.array([10, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000])
for ss in sampleSizes:

    lst0 = np.random.choice(lst0all,ss,replace=False)
    lst1 = np.random.choice(lst1all,ss,replace=False)

    nFeats = ch.getNumFields() 
    trainData = np.zeros((len(lst0)+len(lst1),nFeats))
    targetData = np.hstack((np.zeros(len(lst0)),np.ones(len(lst1))))

    i = 0
    for imName in lst0:
        sample = cv2.imread(cls0 + imName)
        thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        trainData[i,:] = ch.extract(thisIm)
        i = i + 1

    for imName in lst1:
        sample = cv2.imread(cls1 + imName)
        thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        trainData[i,:] = ch.extract(thisIm)
        i = i + 1

    gnb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),algorithm="SAMME",n_estimators=250)


    y_pred = gnb.fit(trainData,targetData)
    pickle.dump(gnb, open( "../boosters/fhgBoosterS" + str(ss)+ ".p","wb"))
    y_pred = gnb.predict(trainData)
    print(str(ss)  + " - number of mislabeled points out of a total %d points : %d" % (trainData.shape[0],(targetData != y_pred).sum()))
