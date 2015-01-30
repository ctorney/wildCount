import sys, os
import numpy as np
import cv2
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

cls0 = './no/'
cls1 = './yes/'
lst0 = [name for name in os.listdir(cls0)]# if os.path.isfile(name)]
lst1 = [name for name in os.listdir(cls1)]# if os.path.isfile(name)]

nFeats = 1
trainData = np.zeros((len(lst0)+len(lst1),nFeats))
targetData = np.hstack((np.zeros(len(lst0)),np.ones(len(lst1))))


i = 0
for imName in lst0:
    sample = cv2.imread(cls0 + imName)
    thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    trainData[i,:] = (np.max(thisIm)-np.min(thisIm))/ (np.max(thisIm)+np.min(thisIm))
    i = i + 1

for imName in lst1:
    sample = cv2.imread(cls1 + imName)#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    trainData[i,:] = (np.max(thisIm)-np.min(thisIm))/ (np.max(thisIm)+np.min(thisIm))
    i = i + 1

gnb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=10)
y_pred = gnb.fit(trainData,targetData)
pickle.dump(gnb, open( "../boosters/contrastBooster.p","wb"))
y_pred = gnb.predict(trainData)
print("Number of mislabeled points out of a total %d points : %d" % (trainData.shape[0],(targetData != y_pred).sum()))
