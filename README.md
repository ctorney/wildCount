# wildCount
Automated counting of animals using rotation invariant features and machine learning classifier

Project is structured as follows

## boosters 
Empty directory for storing the adaboost classifiers created by sklearn.

## classify
This code finds wildebeest in images. Loops through a directory of aerial images, first identifiers pixels that occupy regions above a threshold contrast level, tests whether each pixel is identified as the centre of a wildebeest, then groups contiguous blocks of positive identifications into a single wildebeest

## countedImages
Empty directory for storing images that have been counted and have identified wildebeest marked

## extractor
This directory contains the classes to extract rotation invariant features. It is an implementation of the algorithm proposed in 
*Liu, Kun, et al. "Rotation-invariant HOG descriptors using fourier analysis in polar and spherical coordinates." International Journal of Computer Vision 106.3 (2014): 342-364.*
Two versions of the algorithm exist, **circularHOGExtractor** calculates features assuming it is a centred image of a single object whereas **cudaHOGExtractor** produces an array of features with each pixel of the image considered as a potential centre for the object

## training
Contains various utilities for creating a training data set (clicking on images and manually correcting a first-pass trained ML algorithm) and code for creating an adaboost classifier using the sklearn package
                                                                                                                                                                         
                                                                                                                                                                         





