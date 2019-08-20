
module XY_lib

  implicit none
  real, parameter :: pi = acos(-1.)
  integer, parameter :: l = 32, nSites = l*l
  integer, parameter :: buff = 512*1024*1024, seed = 314159265
  integer, parameter :: nThreads = 128 ,nBlocks = nSites/nThreads
  integer, parameter :: nTherm = 5000, nMeas =2000, nSkip = 0
  contains
    
    attributes(global) subroutine cudaMagnetization(s_d,COS_s_d,SIN_s_d)
      implicit none
      real, dimension(:), intent(in) :: s_d
      real, dimension(:), intent(inout) :: COS_s_d,SIN_s_d
      integer :: a,b,i
      
      a = threadIdx%x ; b = blockIdx%x
      
      i = a + (b-1)*blockDim%x
      
      COS_s_d(i) = cos(s_d(i))
      SIN_s_d(i) = sin(s_d(i))
      
      
    end subroutine cudaMagnetization
    
!subroutine computes energy of the spin configuration
    attributes(global) subroutine cudaEnergy(s_d,E_d)
      implicit none
      real, dimension(:), intent(in) :: s_d
      real, dimension(:), intent(out) :: E_d
      integer ::a,b,i,x,y,xr,xl,yr,yl,ixr,ixl,iyr,iyl
      
      a = threadIdx%x ; b = blockIdx%x
      i = a + (b-1)*blockDim%x
      
      !determine x and y position from i
      x = mod(i,l)
      if (x==0) x = l
      y = (i-x)/l + 1
      
      if (mod(x+y,2)==0) then
       !find nearest neighbours according to periodic boundary conditions      
        xr = mod(x,l) + 1
        xl = mod(x-2+l,l) + 1
        yr = mod(y,l) + 1
        yl = mod(y-2+l,l) + 1
       
       !find linear index of the neighbour sites
        ixr = xr + (y-1)*l
        ixl = xl + (y-1)*l
        iyr = x + (yr-1)*l
        iyl = x + (yl-1)*l   
        
       !find energy of the site
        E_d(i) = -cos(s_d(i)-s_d(ixr))-cos(s_d(i)-s_d(ixl))-cos(s_d(i)-s_d(iyr))-cos(s_d(i)-s_d(iyl))
        
      else
      
        E_d(i) = 0.
            
      endif
      
      
    end subroutine cudaEnergy
    
    !subroutine to perform monte carlo sweep on a bipartite sublattice once
    attributes(global) subroutine cudaSweep(s_d,beta,xx_d,opt)
      implicit none
      integer, value, intent(in) :: opt
      real, value, intent(in) :: beta
      real, dimension(:), intent(in) :: xx_d
      real, dimension(:), intent(inout) :: s_d
      integer ::a,b,i,x,y,xr,xl,yr,yl,ixr,ixl,iyr,iyl
      real :: delE,oldE,newE,delta
      
      a = threadIdx%x ; b = blockIdx%x
      
      i = a + (b-1)*blockDim%x
      
      x = mod(i,l)
      if (x==0) x = l
      y = (i-x)/l + 1
      
      if (mod(x+y,2)==opt) then
       
       !find nearest neighbours according to periodic boundary conditions      
        xr = mod(x,l) + 1
        xl = mod(x-2+l,l) + 1
        yr = mod(y,l) + 1
        yl = mod(y-2+l,l) + 1
       
       !find linear index of the neighbour sites
        ixr = xr + (y-1)*l
        ixl = xl + (y-1)*l
        iyr = x + (yr-1)*l
        iyl = x + (yl-1)*l

       !compute energy of current configuration
        oldE = -cos(s_d(i)-s_d(ixr))-cos(s_d(i)-s_d(ixl))-cos(s_d(i)-s_d(iyr))-cos(s_d(i)-s_d(iyl))
               
        !propose a random change to the configuration      
        delta = xx_d(2*i)*2.*pi
        
        !compute energy of proposed configuration
        newE = -cos(s_d(i)+delta-s_d(ixr))-cos(s_d(i)+delta-s_d(ixl))-cos(s_d(i)+delta-s_d(iyr))-cos(s_d(i)+delta-s_d(iyl))
        
        !calculate change in energy
        delE = newE - oldE
        
        !accept or reject the change with Metropolis algorithm
        if (xx_d(i).le.(exp(-beta*delE))) then
           s_d(i) = s_d(i) + delta
        endif
        
        !keep the spins in the range [0,2pi)
        s_d(i) = mod(s_d(i),2.*pi)
               
      endif
      
    end subroutine cudaSweep


end module XY_lib

program  main
  use cudafor
  use CudaRandF
  use XY_lib
  use, intrinsic :: iso_c_binding
  implicit none
  real, dimension(:), allocatable :: s,values
  real, dimension(:), allocatable, device :: s_d,E_d,COS_s_d,SIN_s_d
  type(CudaRand_generator) :: pCR
  type(C_devptr) :: x_d,y_d
  real(C_float), dimension(:), pointer, device :: xx_d,yy_d
  real :: temp,beta,mx,my,res
  integer :: opt,istat,iupdate,i
  
  !allocate space on device to store spin configurations
  allocate(s(nSites),s_d(nSites),E_d(nSites), values(nMeas) &
  ,COS_s_d(nSites),SIN_s_d(nSites),stat = istat)
  if (istat /= 0) then
    print*,'** Memory Allocation Error, terminating application **'
    stop
  endif
  
  !set up GPU PRNG
  call setRandom(pCR,buff,seed)
  
  open(unit=11,file='gpu_data')
  
  temp = 0.0
!do 
  !set system's inverse temperature
  temp = temp + 2.025	
  !if (temp.gt.2.5) exit
  beta = 1./temp
  
  !start the system with coldstart and copy 
  s_d = 0.
  
   !thermalize 
  do iupdate=1,nTherm
    call getRandom(x_d,pCR,2*nSites)
    call c_f_pointer(x_d,xx_d,(/2*nSites/))
    do opt=0,1
      call cudaSweep<<<nThreads,nBlocks>>>(s_d,beta,xx_d,opt)
    enddo!opt    
  enddo!iupdate
  
  !measure energy of the system  
  do iupdate=1,nMeas
    call getRandom(x_d,pCR,2*nSites)
    call c_f_pointer(x_d,xx_d,(/2*nSites/))
    do opt=0,1
      call cudaSweep<<<nThreads,nBlocks>>>(s_d,beta,xx_d,opt)
    enddo!opt
    
    call cudaEnergy<<<nThreads,nBlocks>>>(s_d,E_d)
    values(iupdate) = sum(E_d)/float(nSites)
    !print*,values(iupdate)
    print*,values(iupdate)
    
  enddo!iupdate
  
  !res = sum(values)/float(nMeas)
  !print*,temp,res
  !write(11,*)temp,res
  
 !enddo !temp
 !close(11)

  !code for plotting XY spin configurations
  !s = s_d  
  !open(unit=11,file='data')
  !do i = 1,nSites
  !    x = mod(i,l)
  !    if (x==0) x = l
  !    y = (i-x)/l + 1
  !    u = 0.4*cos(s(i))
  !    v = 0.4*sin(s(i))
  !    write(11,*),x-u,y-v,2*u,2*v    
 ! enddo
 ! close(11)
 

  
end program main



























