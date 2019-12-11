! -*-F90-*-
!-- Make dz.F90.
program main
  use libmxe_display, only : libmxe_display__nml_block, &
                           & libmxe_display__var,       &
                           & libmxe_display__output,    &
                           & libmxe_display__file
  implicit none


  integer, parameter :: kmmax = 200, lun = 21

  !-- input variables --
  !- depth(km) :: vertical levels of box bottom [cm]
  integer :: km
  real(8) :: depth_cm(kmmax)

  integer :: k
  real(8),allocatable,dimension(:) :: dep,dzz,dz,dp

  namelist /grid_level/ km, depth_cm


  read( 5, nml=grid_level )

  call libmxe_display__nml_block( 'grid_level' )
  call libmxe_display__var( km, 0, 'km' )
  call libmxe_display__var( depth_cm, 0.d0, 'depth_cm' )

  allocate(dep(km+1))
  dep(1)=0.d0
  dep(2:km+1)=depth_cm(1:km)


  !---- create ----
  allocate(dz(km),dp(km),dzz(km+1))
  do k = 1, km
    dz(k) = dep(k+1) - dep(k)  !- height of box
  enddo

  dp=0.d0
  do k = 1, km
    dp(k) = dep(k) + 0.5d0 * dz(k) !- depth of box-center (T-point)
  enddo

  dzz(1) = dp(1)
  do k = 2, km
    dzz(k) = dp(k) - dp(k-1)  !- interval of T-point
  end do
  dzz(km+1) = 0.5d0 * dz(km)


  !---- check ----
  !-- dz must be positive --
  if (minval(dz) < 0.d0) then
    write(6,*) 'k, dp, dz'
    do k = 1, km
      write(6,'(I4,F12.4,F12.4)') k, dp(k), dz(k)
    enddo
    write(6,*) 'dz must be positive'
    stop
  endif


  !-- monotone increases --
  if ( minval( dz(2:km) - dz(1:km-1) ) < 0.d0 ) then
    write(6,*) 'k, dp, dz'
    do k = 1, km
      write(6,'(I4,F12.4,F12.4)') k, dp(k), dz(k)
    enddo
    write(6,*) 'caution: dz does not increase monotonically'
  endif


  call libmxe_display__output
  call libmxe_display__file( 'MRI.COM input file', 'dz_cm.d', 'sequential' )

  open( lun, file='dz_cm.d', action='write', form='unformatted' )
  write(lun) km
  write(lun) dz(1:km)
  close(lun)

  call libmxe_display__file( 'layer info', 'depth.txt', 'text' )
  open(lun,file='depth.txt',action='write')
  write(lun,'(A8,A12,A12,A12,A12)') 'level','dz','dep','dp','dzz'
  do k = 1, km
    write(lun,'(I8,12X,F12.4,12X,F12.4)') k,dep(k)/100.0,dzz(k)/100.0
    write(lun,'(8X,F12.4,12X,F12.4)') dz(k)/100.0,dp(k)/100.0
  enddo
  k = km + 1
  write(lun,'(I8,12X,F12.4,12X,F12.4)') k,dep(k)/100.0,dzz(k)/100.
  close(lun)

end program main
