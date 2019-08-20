
echo "Linking CUDA libraries"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH 

echo "Compiling CUDA C/C++"
#nvcc -c CudaRand.cu CudaRandKernels.cu
echo "Compiling CUDA FORTRAN"
#pgfortran -c CudaRandF.f90 -Mcuda
pgfortran -c main.f90 -Mcuda
echo "Combining C and FORTRAN"
pgfortran -o a.out CudaRand.o CudaRandKernels.o CudaRandF.o main.o -lstdc++ -Mcuda
echo "Compilation complete!"
