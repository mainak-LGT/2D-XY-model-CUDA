#ifndef CUDARANDKERNELS_H
#define CUDARANDKERNELS_H

__device__ int cudaRanluxDec24(int x);
__device__ float cudaRanluxGetOneNumber(sRandState& s);
__global__ void ranluxKernel(float* pCudaResult, sRandState* pCudaRandState,
                             size_t randPitch, int nIterations);

#endif /* CUDARANDKERNELS_H */

