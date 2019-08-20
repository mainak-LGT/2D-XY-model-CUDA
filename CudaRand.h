#ifndef CUDARAND_H
#define CUDARAND_H

#include <stdexcept>
//#include "log4cxx/logger.h"

// sizeof(sRandState)/sizeof(int_or_float) is odd. This prevents
// bank-conflicts in shared memory.
struct sRandState
{
  int i24, j24;
  float seeds[24];
  float carry;
};

class CudaRand
{
  private:
    // This class is a singleton.
    CudaRand(const int deviceBufferSizeByte, const int seed);
    CudaRand(const CudaRand& cc);

    void seedRandStates(const int seed);
    void ranluxSeed(sRandState* s, const int seed);
    void generateRandomNumbers(const int nIterations);   
    void checkCudaError(const char *msg);

  //  log4cxx::LoggerPtr logger;
    void* sMalloc(size_t size);

    // Host- and device-memory for randStates and results.
    sRandState* pHostRandStates;
    sRandState* pCudaRandStates;
    size_t randPitch;
    float* pHostRandResult;
    float* pCudaRandResult;

    int deviceBufferSizeByte;
    int nNumbersInOneRun;
    int nNumbersInDeviceMemory;
    int nUsedNumbersInDeviceMemory;

  public:
    ~CudaRand();
    static CudaRand* getInstance(const int deviceBufferSizeByte = 256*1024*1024,
                                 const int seed = 0);

    void test();
    void test2();
    void getArray(float* a, const int n);
    float* getDeviceNumbers(const int n);

    class RuntimeException : public std::runtime_error
    {
      public:
        RuntimeException(const char* errorString) :
          runtime_error(errorString) {}
    };
    class CudaException : public std::runtime_error
    {
      public:
        CudaException(const char* errorString) :
          runtime_error(errorString) {}
    };

};

#endif /* CUDARAND_H */

