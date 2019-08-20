//#include "log4cxx/logger.h"
#include "CudaRand.h"
#include "CudaRandKernels.h"
#include "ranluxCuda.constants"

CudaRand* CudaRand::getInstance(const int deviceBufferSizeByte,
                                const int seed)
{
  static CudaRand instance(deviceBufferSizeByte, seed);
  return &instance;
}

CudaRand::CudaRand(const int deviceBufferSizeByte_, const int seed) :
  nNumbersInDeviceMemory(0),
  nUsedNumbersInDeviceMemory(0)
{
  //logger = log4cxx::Logger::getLogger("CudaRand");

  // Every Thread is an independent PRNG and needs it's own randState.
  // The randStates are organized in a pitched (1) 2-dimensional array:
  // one row per block.
  cudaMallocPitch((void**) &pCudaRandStates, &randPitch,
                  CRAND.blockDim.x * sizeof(sRandState),
                  CRAND.gridDim.x);
  checkCudaError("cudaMallocPitch(&pCudaRandStates...");
  // Create the same array on the host
  pHostRandStates = (sRandState*) sMalloc(CRAND.gridDim.x * randPitch);
  //LOG4CXX_INFO(logger, "Allocated host- and device-memory for randStates");

  // Seed all those randStates and copy them to device memory.
  seedRandStates(seed);
  cudaMemcpy((void*) pCudaRandStates,
             (void*) pHostRandStates,
             CRAND.gridDim.x * randPitch,
             cudaMemcpyHostToDevice);
  checkCudaError("cudaMemcpy(pCudaRandStates, ...");
  //LOG4CXX_INFO(logger, "Copied to device.");

  // Get memory on device for results.
  nNumbersInOneRun = CRAND.gridDim.x * CRAND.blockDim.x * 24;
  int nBytesInOneRun = nNumbersInOneRun * sizeof(float);
  int effectiveSizeByte = (deviceBufferSizeByte_ / nBytesInOneRun) *
                           nBytesInOneRun;
  deviceBufferSizeByte = effectiveSizeByte;
  //LOG4CXX_INFO(logger, "Allocating device memory for random numbers. "
    //<< "Maximum buffer size was " << deviceBufferSizeByte_
    //<< " bytes, i'm going to use "<< effectiveSizeByte << " bytes");

  cudaMalloc((void**) &pCudaRandResult, deviceBufferSizeByte);
  checkCudaError("cudaMalloc(&pCudaRandResult...");
}

void CudaRand::seedRandStates(const int seed)
{
  for(int iBlock=0; iBlock<CRAND.gridDim.x; iBlock++)
  {
    sRandState* pRow =
      (sRandState*) (((char*) pHostRandStates) + iBlock * randPitch);
    for(int iThread=0; iThread<CRAND.blockDim.x; iThread++)
    {
      ranluxSeed(&pRow[iThread], seed + iBlock * CRAND.blockDim.x + iThread);
    }
  }
  //LOG4CXX_INFO(logger, "Seeded randStates. Copying to device.");
}


void CudaRand::ranluxSeed(sRandState* s, const int seed)
{
  //static log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("ranlux.Seed"));

  int jseed = seed;
  int k;
  //LOG4CXX_DEBUG(logger, "Seeding " << s << " with seed " << seed);
  for(int i=0; i<24; i++)
  {
    k = jseed / 53668;
    jseed = 40014 * (jseed - k*53668) - k*12211;
    if(jseed < 0)
      jseed += 2147483563;
    s->seeds[i] = (jseed % 16777216) * CRAND.twom24;
  }
  s->i24 = 23;
  s->j24 = 9;
  s->carry = 0.f;
  if(s->seeds[23] == 0.)
    s->carry = CRAND.twom24;
}

// Generate new numbers. Those numbers will always start at the base-
// address of pCudaRandResult, probably overwriting a few unused numbers
// at the end.
void CudaRand::generateRandomNumbers(const int nIterations)
{
  // Run Kernel
  //LOG4CXX_INFO(logger, "Launching Kernel with " << CRAND.gridDim.x
  //  << " blocks and " << CRAND.blockDim.x << " threads per block. "
  //  << nIterations << " iterations yielding 24 numbers each.");
  ranluxKernel<<< CRAND.gridDim, CRAND.blockDim >>>(pCudaRandResult, pCudaRandStates, randPitch, nIterations);
  checkCudaError("Kernel launch ranluxKernel");
}


///I added  - start


  void CudaRand::test2()
{



  // Check prerequisites
  if(nNumbersInDeviceMemory == 0)
  {
    
    int nNumbers = deviceBufferSizeByte / sizeof(float);
    float* pLocal = (float*) sMalloc(deviceBufferSizeByte);
    getArray(pLocal, nNumbers);
    
    int thread = 0; 
    int block = 0;
    
    int nRuns = nNumbersInDeviceMemory / nNumbersInOneRun;
    int offset = block * CRAND.blockDim.x * nRuns * 24 + thread;
    int runOffset = 24 * CRAND.blockDim.x;
    int numberOffset = CRAND.blockDim.x;
    
    float test[nRuns];
    for(int i=0; i<nRuns; i++){
      test[i] = pLocal[offset + 0*runOffset + i*numberOffset];
      printf("%f\n",test[i]);
    }
    
   }
  
}

///I added - end


// This test will fail (likely with a segfault) if deviceBufferSize
// is too small to generate all numbers needed at once.
void CudaRand::test()
{
  // Reference numbers 0-4 and 100-104 published by F. James for seed
  // 314159265 and p=223
  const float r223[] = {0.53981817, 0.76155043, 0.06029940, 0.79600263,
                        0.30631220, 0.43156743, 0.03774416, 0.24897110,
                        0.00147784, 0.90274453};
  // Reference numbers 0-4 and 100-104 published by F. James for seed 1
  // and p=389
  const float r389[] = {0.94589490, 0.47347850, 0.95152789, 0.42971975,
                        0.09127384, 0.02618265, 0.03775346, 0.97274780,
                        0.13302165, 0.43126065};

  // You may change these numbers: 0 <= thread < blockDim.x
  //                               0 <= block < gridDim.x
  // but you should really know what you're doing since you have to
  // adjust the seed accordingly. You should also 
  int block = 0;
  int thread = 0;

  // Check prerequisites
  if(nNumbersInDeviceMemory == 0)
  {
    // Ok, this instance is new.
    // Get as many numbers as the device buffer is able to deliver.
    int nNumbers = deviceBufferSizeByte / sizeof(float);
    float* pLocal = (float*) sMalloc(deviceBufferSizeByte);
    getArray(pLocal, nNumbers);
    // Now that we have those numbers, we have to calculate at which position
    // our references should be. This depends on the number of runs
    // were requested from the kernel.
    int nRuns = nNumbersInDeviceMemory / nNumbersInOneRun;
    int offset = block * CRAND.blockDim.x * nRuns * 24 + thread;
    int runOffset = 24 * CRAND.blockDim.x;
    int numberOffset = CRAND.blockDim.x;
    // Put the numbers into another array
    float test[10];
    for(int i=0; i<5; i++)
      test[i] = pLocal[offset + 0*runOffset + i*numberOffset];
    for(int i=0; i<5; i++)
      test[i+5] = pLocal[offset + 4*runOffset + (i+4)*numberOffset];
    const float* r;
    if(CRAND.p == 223)
      r = r223;
    else if(CRAND.p == 389)
      r = r389;
    else
    {
    //  LOG4CXX_ERROR(logger, "p should be 223 or 389, not " << CRAND.p
      //  << ". You will get weird results now, since i'm assuming 223!");
      r = r223;
    }
    // Write differences to logger.
    //LOG4CXX_ERROR(logger, "Following are the differences between the "
      //<< "reference values and the calculated ones. If they differ only "
      //<< "about 1e-8, everything's ok. If not, please check if you "
      //<< "instantiated either with p=223 and seed=314159265 or with "
      //<< "p=389 and seed=1");
    for(int i=0; i<10; i++)
      //LOG4CXX_ERROR(logger, test[i] - r[i]);
      printf("test[%d]-r[%d] = %f - %f \n",i,i,test[i],r[i]);
    //free(pLocal);
  }
  else
  {
    //LOG4CXX_ERROR(logger, "Can't perform test. This instance has been used "
      //<< "before. Don't fetch any numbers before calling test()");
  }
}

void CudaRand::getArray(float* a, const int n)
{
  float* pDevice = getDeviceNumbers(n);
  //LOG4CXX_INFO(logger, "Copying " << n << " numbers from device to array.");
  cudaMemcpy((void*) a,
             (void*) pDevice,
             n * sizeof(float),
             cudaMemcpyDeviceToHost);
  checkCudaError("cudaMemcpy(device->a...");
}

// Reserves n numbers on the card for consumption on the card. Returns
// a pointer to the first usable number. If there are not enough numbers
// available, it generates those numbers. The returned pointer is always
// aligned at a multiple of 512bytes/128floats.
float* CudaRand::getDeviceNumbers(const int n)
{
//  LOG4CXX_DEBUG(logger, "Trying to reserve " << n << " numbers on device.");
  //printf("n=%d,deviceBufferSizeByte=%d\n",n,deviceBufferSizeByte);
  int nextUsableIdx = ((nUsedNumbersInDeviceMemory + 127) / 128) * 128;
  if(nNumbersInDeviceMemory - nextUsableIdx < n)
  {
    // We don't have enough numbers. Check if it's possible to create
    // enough.
    if(n > deviceBufferSizeByte / sizeof(float))
      //TODO variablen mit raus.
      throw RuntimeException("Device-buffer too small.");
    // Yes. Fill up device buffer.
    int nRuns;
    if(nUsedNumbersInDeviceMemory == 0)
    {
      // Since there are not enough numbers available, but it was calculated
      // that i may generate that many numbers but nUsedNumbers is zero, i
      // conclude that this is the first run. I'm going to fill the whole
      // buffer.
      nRuns = (deviceBufferSizeByte / sizeof(float)) / nNumbersInOneRun;
    }
    else
    {
      // We already generated numbers some time ago. Therfore the numbers
      // after nUsedNumbers are unused. We leave them (except a possible
      // overlap), fill up from the beginning up to nUsedNumbers, and set
      // back nUsedNumbers to zero.
      nRuns = (nUsedNumbersInDeviceMemory + nNumbersInOneRun - 1) /
              nNumbersInOneRun;
    }
    generateRandomNumbers(nRuns);
    // This will happen only at the first run, when nNumbersInDevice is 0.
    if(nRuns * nNumbersInOneRun > nNumbersInDeviceMemory)
      nNumbersInDeviceMemory = nRuns * nNumbersInOneRun;
    nUsedNumbersInDeviceMemory = 0;
  }
  else
  {
    // We do have enough numbers. Round up.
    nUsedNumbersInDeviceMemory = nextUsableIdx;
  }
  // Return the pointer to the first usable number, and mark those numbers
  // as used.
  int startIdx = nUsedNumbersInDeviceMemory;
  nUsedNumbersInDeviceMemory += n;
  return &pCudaRandResult[startIdx];
}

CudaRand::~CudaRand()
{
  // Release memory.
  //LOG4CXX_INFO(logger, "Freeing memory on host and device");
  cudaFree(pCudaRandStates);
  cudaFree(pCudaRandResult);
  free(pHostRandResult);
  free(pHostRandStates);
}

void CudaRand::checkCudaError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) 
  {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }                         
}

void* CudaRand::sMalloc(size_t size)
{
  void* p = malloc(size);
  if(!p)
    printf("malloc returned Null-Pointer!\n");
  return p;
}

extern "C"
{

 CudaRand* CudaRand__setRandom(const int buff, const int seed){
 
  CudaRand* pCR = CudaRand::getInstance(buff,seed);
  return pCR;
 
 
 }

 float* CudaRand__getRandom(CudaRand* pCR, const int n){
 
  float* x = pCR->getDeviceNumbers(n);
  return x;
 
 }


}

/* ==== REMARKS ====
(1) The pitching won't be used here: Since blockDim.x is 128 and the size
    of an integer is 4 bytes, the rows would be aligned at 512-byte-offsets
    anyway.

*/

