

#include <pycuda-complex.hpp>
#include <stdio.h>
__global__ void d_fft(pycuda::complex<float> *grad, pycuda::complex<float> *f_grad, int nm, int nx, int ny)
{
    
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i>=nx)
        return;
    if (j>=ny)
        return;
    int idx = i + j * nx;
 
    float phi = pycuda::arg(grad[idx]);
    float r = pycuda::abs(grad[idx]);
    pycuda::complex<float> x(0.0,1.0);
    for (int k=0;k<nm;k++)
    {
        f_grad[idx + (nx*ny*k)] =  r*pycuda::exp( -x * (float)k * phi);
    }
}

            

__global__ void d_convolve(pycuda::complex<float> *features, pycuda::complex<float> *gpu_hist, int *positions, pycuda::complex<float> *gpu_template, int maxsize, int maxK, int tn, int nx, int ny, int freq)
{
    
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx>=maxsize)
        return;

    int i = positions[2*idx];
    int j = positions[2*idx+1];
    int hnt = 0.5*tn;
    if ((i>=(nx-hnt))||((i<=hnt)))
        return;
    if ((j>=(ny-hnt))||((j<=hnt)))
        return;



    for (int k=0;k<maxK;k++)
    {

        pycuda::complex<float> retVal(0.0,0.0);

        for (int t_i=0;t_i<tn;t_i++)
            for (int t_j=0;t_j<tn;t_j++)
            {
                int i2 = i + (t_i - hnt );
                int j2 = j + (t_j - hnt );
                int idx2 = j2 + i2 * ny;
                retVal += gpu_hist[idx2+(nx*ny*k)]*gpu_template[t_j + (t_i*tn)];
            }
        features[idx*maxK*maxK + freq*maxK + k] = retVal;
    }


    

}


__global__ void d_rotation_inv(pycuda::complex<float> *vals, float *features, int maxsize, int maxK, int bin, int fullCount, int binCount)
{
    
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx>=maxsize)
        return;

    int f_index=0;
    int start_index = idx*fullCount + bin*binCount;

    for (int i=0;i<maxK;i++)
        for (int j=0;j<maxK;j++)
        {
            pycuda::complex<float> thisVal = vals[idx*maxK*maxK + i*maxK + j];
            if (i==j)
            {
                features[start_index+f_index++] = pycuda::real(thisVal);
                features[start_index+f_index++] = pycuda::imag(thisVal);
            }
            else
            {
                for (int i2=0;i2<maxK;i2++)
                    for (int j2=0;j2<maxK;j2++)
                    {
                        if (i2<i) continue;
                        if (j2<j) continue;
                        pycuda::complex<float> thisVal2 = vals[idx*maxK*maxK + i2*maxK + j2];
                        if ((i-j)==(i2-j2))
                        {
                            features[start_index+f_index++] = pycuda::real(thisVal*pycuda::conj(thisVal2));
                            features[start_index+f_index++] = pycuda::imag(thisVal*pycuda::conj(thisVal2));
                        }

                    }
            }
        }
    


    

}
