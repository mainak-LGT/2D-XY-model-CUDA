module CudaRandF
  use, intrinsic :: iso_c_binding
  use cudafor
  implicit none
  private
  
  type CudaRand_generator
    private
    type(C_ptr) :: object = c_null_ptr
  end type CudaRand_generator
  
  interface 
    function C_CudaRand__setRandom(buff,seed) result(this) bind(C,name="CudaRand__setRandom")
      import 
      type(C_ptr),value :: this
      integer(C_int),value :: buff,seed
    end function C_CudaRand__setRandom
    function C_CudaRand__getRandom(pCR,n) result(this) bind(C,name="CudaRand__getRandom")
      import
      type(C_ptr),value :: pCR
      type(C_devptr) :: this
      integer, value :: n
    end function C_CudaRand__getRandom    
  end interface
  
  interface setRandom
    module procedure CudaRand__setRandom
  end interface setRandom
  
  interface getRandom
    module procedure CudaRand__getRandom
  end interface getRandom
  
  
  public :: CudaRand_generator, setRandom, getRandom
  
  contains
  
   subroutine CudaRand__setRandom(this,buff,seed) 
     type(CudaRand_generator) :: this
     integer :: buff,seed
     this%object = C_CudaRand__setRandom(buff,seed)
   end subroutine CudaRand__setRandom
   
   subroutine CudaRand__getRandom(x,this,n)
     type(CudaRand_generator) :: this
     type(C_devptr) :: x
     integer :: n
     x = C_CudaRand__getRandom(this%object,n)
   end subroutine CudaRand__getRandom
  
end module CudaRandF



