

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

            

__global__ void d_convolve(float *features, pycuda::complex<float> *gpu_hist, int *positions, pycuda::complex<float> *gpu_template, int maxsize, int k, int tn, int nx, int ny, int counter, int ABS_REAL_IMAG)
{
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx>=maxsize)
        return;

    int i = positions[2*idx];
    int j = positions[2*idx+1];
  
 
     if ((i>=(nx-tn))||((i<=tn)))
         return;
     if ((j>=(ny-tn))||((j<=tn)))
         return;

    pycuda::complex<float> retVal(0.0,0.0);
    int hnt = 0.5*tn;
    for (int t_i=0;t_i<tn;t_i++)
        for (int t_j=0;t_j<tn;t_j++)
        {
            int i2 = i + (t_i - hnt );
            int j2 = j + (t_j - hnt );
            int idx2 = j2 + i2 * ny;
            retVal += gpu_hist[idx2+(nx*ny*k)]*gpu_template[t_j + (t_i*tn)];
        }



    
    features[idx]=0.0;//idx + (nx*ny*counter)] = 0.0;

    if (ABS_REAL_IMAG==0)
        features[idx ] = pycuda::abs(retVal);
    if (ABS_REAL_IMAG==1)
        features[idx ] = pycuda::real(retVal);
    if (ABS_REAL_IMAG==2)
        features[idx] = pycuda::imag(retVal);

}
