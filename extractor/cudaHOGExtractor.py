
import pycuda.elementwise as elementwise
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
culinalg.init()
import cmath as cm
import numpy as np
import cv2

from pycuda.compiler import SourceModule

class cudaHOGExtractor():
    """
    This method takes in an image and extracts rotation invariant HOG features
    following the approach in this paper: 
    Liu, Kun, et al. "Rotation-invariant HOG descriptors using fourier analysis in polar and spherical coordinates."
    International Journal of Computer Vision 106.3 (2014): 342-364.
    HOG features are extracted with every pixel considered as the potential centre of an object,
    and extraction performed in parallel on a GPU
    """
    def __init__(self, bins=4, size=6, max_freq=4):

        # number of bins in the radial direction for large scale features
        self.mNBins = bins
        # size of bin in pixels, this sets the required radius for the image = bins*size
        self.mNSize = size
        # number of fourier modes that will be used (0:modes-1)
        self.mNMaxFreq = max_freq 

        mf = self.mNMaxFreq+1
        self.mNCount = 2*(bins-1) * (mf + 2*(np.dot([mf - i for i in range(mf)] , range(mf))  ))
        self.mNBinCount = 2* (mf + 2*(np.dot([mf - i for i in range(mf)] , range(mf))  ))
        # create a list to store kernels for regional descriptors based on circular harmonics
        self.ciKernel = []
        self.gpu_histF=None

        # build the internal regions - (bins-1) concentric circles
        modes = range(0, self.mNMaxFreq+1)
        scale = range(2, self.mNBins+1)

        for s in scale:
            r = int(self.mNSize * s)
            ll = range(1-r,r)
            [x,y] = np.meshgrid(ll,ll)
            z = x + 1j*y
            phase_z = np.angle(z);
                            
            for k in modes:
                kernel = self.mNSize - np.abs(np.abs(z) - (r-self.mNSize)) 
                kernel[kernel < 0] = 0
                kernel = np.multiply(kernel,np.exp(1j*phase_z*k))
                sa = np.ravel(np.abs(kernel))
                kernel = kernel / np.sqrt(np.sum(np.multiply(sa,sa)))

                self.ciKernel.append(kernel)

        self.mod = SourceModule(open("../extractor/kernel.cu", "r").read())


    def extract(self, img):
        I = img.astype(float)/255.0
#      I = (I-I.mean())/I.std()

        # size and centre of image
        (nx, ny) = I.shape
        cx = int(round(0.5*nx))
        cy = int(round(0.5*ny))

        # compute gradient with a central difference method and store in complex form
        (dy, dx) = np.gradient(I)
        dz = dx + 1j*dy

        # compute magnitude/phase of complex numbers
        phi = np.angle(dz)
        r = np.abs(dz)
        r = r/(r.std()+0.001)
 #       r = r/(r.mean()+0.001)


        # create an empty array for storing the dfft of the orientation vector
        histF = np.zeros([nx, ny, self.mNMaxFreq+1])+0j

        # take the dfft of the orientation vector up to order MaxFreq
        # positive values of k only since negative values give conjugate
        for k in range(0,self.mNMaxFreq+1):
            histF[:,:,k] = np.multiply(np.exp( -1j * (k) * phi) , r+0j)
        

        # compute regional descriptors by convolutions (these descriptors are not rotation invariant)
        fHOG = np.zeros([self.mNCount])
        scale = range(0, self.mNBins-1)
        f_index = 0
        for s in scale:
            allVals = np.zeros((self.mNMaxFreq+1,self.mNMaxFreq+1),dtype=np.complex64)
            for freq in range(0,self.mNMaxFreq+1):
                template = self.ciKernel[s*(self.mNMaxFreq+1)+freq]
                (tnx, tny) = template.shape
                tnx2 = int(round(0.5*tnx))
                for k in range(0,self.mNMaxFreq+1):
                    allVals[freq,k] = np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template)))
            for (x,y), val in np.ndenumerate(allVals):
                if x==y:
                    fHOG[f_index]=val.real
                    f_index+=1
                    fHOG[f_index]=val.imag
                    f_index+=1
                else:
                    for (x1,y1), val1 in np.ndenumerate(allVals):
                        if x1<x: continue
                        if y1<y: continue
                        if (x-y)==(x1-y1):
                            fHOG[f_index]=(val*val1.conjugate()).real
                            f_index+=1
                            fHOG[f_index]=(val*val1.conjugate()).imag
                            f_index+=1

        return fHOG.tolist()



    def prepareExtract(self, img):

        I = img.astype(float)/255.0

        # size and centre of image
        (nx, ny) = I.shape        
        maxThreads = 16

        bx = nx//maxThreads
        by = ny//maxThreads

        
        cx = int(round(0.5*nx))
        cy = int(round(0.5*ny))

        # compute gradient with a central difference method and store in complex form
        (dy, dx) = np.gradient(I)

        dz = dx + 1j*dy
#       ddd = np.abs(dz[500-16:500+16,100-16:100+16])
        # compute magnitude/phase of complex numbers
        phi = np.angle(dz)
        r = np.abs(dz)
        box = 32
#       tmp1 = cv2.blur(r,(box,box))
#        tmp2 = cv2.blur(np.multiply(r,r),(box,box))
#        stddev = np.sqrt(tmp2 - np.multiply(tmp1,tmp1)) + 0.0001

#       dz = np.divide(dx,stddev) + np.divide(dy,stddev)*1j
        dz = dz.astype(np.complex64)
        dz_gpu = gpuarray.to_gpu(dz)
	

        histF =np.zeros((self.mNMaxFreq + 1, nx, ny),dtype=np.complex64)
        self.gpu_histF = gpuarray.to_gpu(histF)

        func = self.mod.get_function("d_fft")

        func(dz_gpu, self.gpu_histF, np.int32(self.mNMaxFreq+1), np.int32(nx), np.int32(ny), block = (maxThreads, maxThreads, 1), grid = (bx, by, 1))


    def denseExtract(self, img, positions, N):
        I = img.astype(float)/255.0

        # size and centre of image
        (nx, ny) = I.shape        
        maxThreads = 256
        bx = 32
        if N>(maxThreads*bx):
            print 'ERROR: exceeds maximum size of cuda array'
            return
        gpu_positions = gpuarray.to_gpu(positions.astype(np.int32))
        features = np.zeros((N,self.mNCount),dtype=np.float32)
        func2 = self.mod.get_function("d_convolve")
        func3 = self.mod.get_function("d_rotation_inv")
        # compute regional descriptors by convolutions (these descriptors are not rotation invariant)
        fHOG = np.zeros([self.mNCount])
        f_index = 0
        template = self.ciKernel[0].astype(np.complex64)
        gpu_template = gpuarray.to_gpu(template)
        feature_slice = np.zeros((N,self.mNCount),dtype=np.float32)
        gpu_features = gpuarray.to_gpu(feature_slice)

        scale = range(0, self.mNBins-1)
        for s in scale:
            allVals = np.zeros((N,self.mNMaxFreq+1,self.mNMaxFreq+1),dtype=np.complex64)
            gpu_vals = gpuarray.to_gpu(allVals)
            for freq in range(0,self.mNMaxFreq+1):
                template = self.ciKernel[s*(self.mNMaxFreq+1)+freq].astype(np.complex64)
                gpu_template = gpuarray.to_gpu(template)
                (tnx, tny) = template.shape
                tnx2 = int(round(0.5*tnx))
                func2(gpu_vals, self.gpu_histF, gpu_positions, gpu_template, np.int32(N), np.int32(self.mNMaxFreq+1), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(freq), block = (maxThreads, 1, 1), grid = (bx, 1, 1))
            allVals = gpu_vals.get()
            func3(gpu_vals, gpu_features, np.int32(N),np.int32(self.mNMaxFreq+1), np.int32(s),np.int32(self.mNCount), np.int32(self.mNBinCount),  block = (maxThreads, 1, 1), grid = (bx, 1, 1))
        features = gpu_features.get()
        
        
        return features




    
    def getFieldNames(self):
        """
        Return the names of all of the length and angle fields. 
        """
        retVal = []
        for i in range(0,self.mNCount):
            name = "Length"+str(i)
            retVal.append(name)
                        
        return retVal
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """

    def getNumFields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.mNCount


