from SimpleCV.base import *
from SimpleCV.ImageClass import Image
from SimpleCV.Features.FeatureExtractorBase import *
import pycuda.elementwise as elementwise
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
culinalg.init()
import cmath as cm

from pycuda.compiler import SourceModule

class cudaHOGExtractor(FeatureExtractorBase):
    """
    Create a 1D edge length histogram and 1D edge angle histogram.
    
    This method takes in an image, applies an edge detector, and calculates
    the length and direction of lines in the image.
    
    bins = the number of bins
    """
    def __init__(self, bins=4, size=6, max_freq=4):

        # number of bins in the radial direction for large scale features
        self.mNBins = bins
        # size of bin in pixels, this sets the required radius for the image = bins*size
        self.mNSize = size
        # number of fourier modes that will be used (0:modes-1)
        self.mNMaxFreq = max_freq 

        mf = self.mNMaxFreq
        self.mNCount = ((mf+1)*(mf+1) + mf+1 + mf*(mf+1))*(bins-1) + (mf+1)

        # create a list to store kernels for regional descriptors based on circular harmonics
        self.ciKernel = []
        self.gpu_histF=None

        # first create the central region 
        [x,y]=np.meshgrid(range(-self.mNSize+1,self.mNSize),range(-self.mNSize+1,self.mNSize))
        z = x + 1j*y
        kernel = self.mNSize - np.abs(z)
        kernel[kernel < 0] = 0
        kernel = kernel/sum(sum(kernel))

        self.ciKernel.append(kernel)

        # next build the internal regions - (bins-1) concentric circles
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
 #       r = r/(r.mean()+0.001)
	

        # create an empty array for storing the dfft of the orientation vector
        histF = np.zeros([nx, ny, self.mNMaxFreq+1])+0j

        # take the dfft of the orientation vector up to order MaxFreq
        # positive values of k only since negative values give conjugate
        for k in range(0,self.mNMaxFreq+1):
            histF[:,:,k] = np.multiply(np.exp( -1j * (k) * phi) , r+0j)
        
# this makes no sense to me        histF[:,:,0] = histF[:,:,0] * 0.5

        # compute regional descriptors by convolutions (these descriptors are not rotation invariant)
        fHOG = np.zeros([self.mNCount])
        f_index = 0
        template = self.ciKernel[0]
        (tnx, tny) = template.shape
        tnx2 = int(round(0.5*tnx))
        featureDetail = []
        for k in range(0,self.mNMaxFreq+1):
            fHOG[f_index] = abs(np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template))))
            f_index+=1
        #   featureDetail.append([0,-1,-k, 0, -k])


        scale = range(2, self.mNBins+1)
        #for s in scale:
         #   featureDim = 0;
          #  for freq in range(-self.mNMaxFreq,self.mNMaxFreq+1):
           #     for k in range(0,self.mNMaxFreq+1):
            #        ff = freq - k
             #       if(ff >= -self.mNMaxFreq and ff <= self.mNMaxFreq and not(k==0 and freq < 0)):
              #          featureDim = featureDim + 1;
               #         featureDetail.append([s,-1,-k, freq, ff])

        # no descriptors should be conjugate as this gives no extra information, 
        # i.e. we want to keep (UmFk OR U-mF-k) and (U-mFk OR UmF-k). If m=k then 
        # this is a rotation invariant feature, if not we only keep the magnitude
        for s in scale:
            for freq in range(0,self.mNMaxFreq+1):
                k_index = 1 + (s-2)*(self.mNMaxFreq+1)+abs(freq)
                template = self.ciKernel[k_index]
                (tnx, tny) = template.shape
                tnx2 = int(round(0.5*tnx))
                for k in range(0,self.mNMaxFreq+1):
                    if (k==freq):
                        fHOG[f_index] = np.real(np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template))))
                        f_index+=1
                        fHOG[f_index] = np.imag(np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template))))
                        f_index+=1
                    else:
                        fHOG[f_index] = abs(np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template))))
                        f_index+=1

                    if(k > 0):
                        template = np.conjugate(template)
                        fHOG[f_index] = abs(np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template))))
                        f_index+=1


        return fHOG.tolist()

    def prepareExtract(self, img):

        I = img.astype(float)/255.0
#      I = (I-I.mean())/I.std()

        # size and centre of image
        (nx, ny) = I.shape        
        maxThreads = 16

        bx = nx//maxThreads
        by = ny//maxThreads
        features = np.zeros((nx,ny,self.mNCount),dtype=np.float32)

        
        cx = int(round(0.5*nx))
        cy = int(round(0.5*ny))

        # compute gradient with a central difference method and store in complex form
        (dy, dx) = np.gradient(I)
        dz = dx + 1j*dy
        dz = dz.astype(np.complex64)
        dz_gpu = gpuarray.to_gpu(dz)
#       a = np.random.randn(4,4) + 1j*np.random.randn(4,4)

        # compute magnitude/phase of complex numbers
        phi = np.angle(dz)
        r = np.abs(dz)
 #       r = r/(r.mean()+0.001)
	

        histF =np.zeros((self.mNMaxFreq + 1, nx, ny),dtype=np.complex64)
        self.gpu_histF = gpuarray.to_gpu(histF)

        func = self.mod.get_function("d_fft")

        func(dz_gpu, self.gpu_histF, np.int32(self.mNMaxFreq+1), np.int32(nx), np.int32(ny), block = (maxThreads, maxThreads, 1), grid = (bx, by, 1))

#       hf_out = np.empty_like(histF)
#       hf_out = gpu_histF.get()#np.empty_like(a)


        # create an empty array for storing the dfft of the orientation vector

        # take the dfft of the orientation vector up to order MaxFreq
        # positive values of k only since negative values give conjugate
#       for k in range(0,self.mNMaxFreq+1):
#            histF[k,:,:] = np.multiply(np.exp( -1j * (k) * phi) , r+0j)

#       return features
        
    def denseExtract(self, img, positions, N):
        I = img.astype(float)/255.0
#      I = (I-I.mean())/I.std()

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

        # compute regional descriptors by convolutions (these descriptors are not rotation invariant)
        fHOG = np.zeros([self.mNCount])
        f_index = 0
        template = self.ciKernel[0].astype(np.complex64)
        gpu_template = gpuarray.to_gpu(template)
        feature_slice = np.zeros((N),dtype=np.float32)
        gpu_features = gpuarray.to_gpu(feature_slice)
        (tnx, tny) = template.shape
        tnx2 = int(round(0.5*tnx))
        for k in range(0,self.mNMaxFreq+1):
#           fHOG[f_index] = abs(np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template))))
            func2(gpu_features, self.gpu_histF, gpu_positions, gpu_template, np.int32(N), np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(0), block = (maxThreads, 1, 1), grid = (bx, 1, 1))
            feature_slice = gpu_features.get()#np.empty_like(a)
            features[:,f_index]=feature_slice
            f_index+=1
#print 'done ' + str(f_index)  + ' of ' + str(self.mNCount)
        #   featureDetail.append([0,-1,-k, 0, -k])



        scale = range(2, self.mNBins+1)
        #for s in scale:
         #   featureDim = 0;
          #  for freq in range(-self.mNMaxFreq,self.mNMaxFreq+1):
           #     for k in range(0,self.mNMaxFreq+1):
            #        ff = freq - k
             #       if(ff >= -self.mNMaxFreq and ff <= self.mNMaxFreq and not(k==0 and freq < 0)):
              #          featureDim = featureDim + 1;
               #         featureDetail.append([s,-1,-k, freq, ff])

        # no descriptors should be conjugate as this gives no extra information, 
        # i.e. we want to keep (UmFk OR U-mF-k) and (U-mFk OR UmF-k). If m=k then 
        # this is a rotation invariant feature, if not we only keep the magnitude
        for s in scale:
            for freq in range(0,self.mNMaxFreq+1):
                k_index = 1 + (s-2)*(self.mNMaxFreq+1)+abs(freq)
                template = self.ciKernel[k_index].astype(np.complex64)
                gpu_template = gpuarray.to_gpu(template)
                (tnx, tny) = template.shape
                tnx2 = int(round(0.5*tnx))
                for k in range(0,self.mNMaxFreq+1):
                    if (k==freq):
#           func2(gpu_features, self.gpu_histF, gpu_template, np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(1), block = (maxThreads, maxThreads, 1), grid = (bx, by, 1))
                        func2(gpu_features, self.gpu_histF, gpu_positions, gpu_template, np.int32(N), np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(1), block = (maxThreads, 1, 1), grid = (bx, 1, 1))
                        feature_slice = gpu_features.get()#np.empty_like(a)
                        features[:,f_index]=feature_slice
                        f_index+=1
#    func2(gpu_features, self.gpu_histF, gpu_template, np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(2), block = (maxThreads, maxThreads, 1), grid = (bx, by, 1))
                        func2(gpu_features, self.gpu_histF, gpu_positions, gpu_template, np.int32(N), np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(2), block = (maxThreads, 1, 1), grid = (bx, 1, 1))
                        feature_slice = gpu_features.get()#np.empty_like(a)
                        features[:,f_index]=feature_slice
                        f_index+=1
                    else:
#                       func2(gpu_features, self.gpu_histF, gpu_template, np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(0), block = (maxThreads, maxThreads, 1), grid = (bx, by, 1))
                        func2(gpu_features, self.gpu_histF, gpu_positions, gpu_template, np.int32(N), np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(0), block = (maxThreads, 1, 1), grid = (bx, 1, 1))
                        feature_slice = gpu_features.get()#np.empty_like(a)
                        features[:,f_index]=feature_slice
                        f_index+=1

                    if(k > 0):
                        template = np.conjugate(template).astype(np.complex64)
                        gpu_template = gpuarray.to_gpu(template)
#func2(gpu_features, self.gpu_histF, gpu_template, np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(0), block = (maxThreads, maxThreads, 1), grid = (bx, by, 1))
                        func2(gpu_features, self.gpu_histF, gpu_positions, gpu_template, np.int32(N), np.int32(k), np.int32(tnx), np.int32(nx), np.int32(ny), np.int32(f_index), np.int32(0), block = (maxThreads, 1, 1), grid = (bx, 1, 1))
                        feature_slice = gpu_features.get()#np.empty_like(a)
                        features[:,f_index]=feature_slice
                        f_index+=1
#      print 'done ' + str(f_index)  + ' of ' + str(self.mNCount)
        return features

#        print "diff to original array:"
#        print features[0], fHOG[0]
#        print np.max(np.abs(features-fHOG))

        return fHOG.tolist()




    
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


