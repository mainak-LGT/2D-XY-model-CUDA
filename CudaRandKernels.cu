#include "CudaRand.h"
#include "ranluxCuda.constants"

__device__ int cudaRanluxDec24(int x)
{
  if(x == 0)
    return 23;
  else
    return x - 1;
}

__device__ float cudaRanluxGetOneNumber(float* seeds, float& carry)
{
  #include "ranluxCuda.constants"

  float uni = seeds[9] - seeds[23] - carry;
  if(uni < 0)
  {
    uni += 1.f;
    carry = CRAND.twom24;
  }
  else
    carry = 0.f;
  seeds[23] = uni;
  // rotate seeds
  float old23 = seeds[23];
  for(int i=23; i>0; i--)
    seeds[i] = seeds[i-1];
  seeds[0] = old23;

  // small numbers (with less than 12 "significant" bits) are "padded".
  if(uni < CRAND.twom12)
  {
    uni += CRAND.twom24 * seeds[9];
    // and zero is forbidden in case someone takes a logarithm
    if(uni == 0.f)
      uni = CRAND.twom24 * CRAND.twom24;
  }
  return uni;
}

__global__ void ranluxKernel(float* pCudaResult, sRandState* pCudaRandState,
                             size_t randPitch, int nIterations)
{
  #include "ranluxCuda.constants"

  float seeds[24];
  float carry;

  // Calculate basepointer of the randState-row corresponding to this
  // thread.
  char* pTemp = ((char*) pCudaRandState) + blockIdx.x * randPitch;
  sRandState* pCudaRS = (sRandState*) pTemp + threadIdx.x;
  // Load the randstates
  for(int i=0; i<24; i++)
    seeds[i] = pCudaRS->seeds[i];
  carry = pCudaRS->carry;

  // Calculate the basepointer of the cudaResult corresponding to this block.
  float* pResult = pCudaResult + blockIdx.x * CRAND.blockDim.x * nIterations * 24;

  for(int iRun=0; iRun < nIterations; iRun++)
  {
    // Produce 24 random numbers.
    for(int i=0; i<24; i++)
    {
      // Write number directly into device memory.
      pResult[(24*iRun + i) * CRAND.blockDim.x + threadIdx.x] =
        cudaRanluxGetOneNumber(seeds, carry);
    }
    // Throw away some numbers.
    for(int i=0; i<CRAND.p-24; i++)
        cudaRanluxGetOneNumber(seeds, carry);
  }

  // Saving randState back to (persistent) device memory.
  for(int i=0; i<24; i++)
    pCudaRS->seeds[i] = seeds[i];
  pCudaRS->carry = carry;

  // We're done. There are now
  // CRAND.gridDim.x * CRAND.blockDim.x * nIterations * 24
  // fresh random numbers at pCudaResult
}

