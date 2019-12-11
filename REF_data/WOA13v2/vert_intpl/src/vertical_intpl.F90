! -*-F90-*-
!-  Check remap file between parent and sub models.
!-   Usage: See sample/check_tables.sh.
module vertical_interpolation

  use libmxe_para, only: type_libmxe_para, clen 
  use libmxe_grid, only: type_libmxe_grid
  use libmxe_io,   only: type_libmxe_io
  use io_interface,only: type_io_var
#ifdef MXE_NETCDF
  use netcdf
#endif /* MXE_NETCDF */

  implicit none

  real(4),allocatable :: r3d_in (:,:,:)
  real(4),allocatable :: r3d_out(:,:,:)
  real(8),allocatable :: d3d_out(:,:,:)
  real(8),allocatable :: datm(:), datn(:)

  type(type_libmxe_para),save :: para_src, para_dst
  type(type_libmxe_grid),save :: grid_src, grid_dst
  type(type_libmxe_io),  save :: io_src, io_dst

  real(8),allocatable,save :: dep_src(:)
  real(8),allocatable,save :: dep_dst(:)
  real(8),allocatable,save :: mask_in (:,:,:)
  real(8),allocatable,save :: mask_out(:,:,:)

  type(type_io_var),save :: var_in, var_out
  real(4),save :: undef_in, undef_out  
  real(8) :: rmiss8
  integer(4),save :: nrec_in, nrec_out
  integer(4),save :: nrec_in_last, nrec_out_last
  
  !-- netCDF --

  integer(4),parameter :: nvar_max = 30
  integer(4),save :: nvar
  character(clen),save :: name_in(nvar_max)  = 'd'
  character(clen),save :: name_out(nvar_max) = 'd'
  character(clen),save :: unit(nvar_max) = ''
  character(clen),save :: standard_name(nvar_max) = ''

  integer(4),save :: src_im, src_jm, src_km
  integer(4),save :: dst_im, dst_jm, dst_km

  public :: ini
  public :: read_and_write
  public :: next
  public :: has_next

contains

subroutine ini

  use libmxe_para, only: libmxe_para__register
  use libmxe_grid, only: libmxe_grid__register
  use libmxe_io, only: libmxe_io__register
  use io_interface,only: io_interface__register
  use netcdf_io,only: netcdf_io__read_attribute

  implicit none

  character(clen) :: src_namelist !- NAMELIST.MXE for source
  character(clen) :: dst_namelist !- NAMELIST.MXE for destination
  character(len=1) :: cgrid  !- T- or U-grid ("T" or "U", "X" and "Y" are not supported)
  character(len=3) :: shape = 'xyz'  !- xyz,xy,yz,xz,x,y,z,0
  character(clen) :: file_in, file_out
  integer(4) :: deflate_level_in  = 0 !- 0-9 (0: w/o compression)
  integer(4) :: deflate_level_out = 0 !- 0-9 (0: w/o compression)

  logical,save :: l_netcdf_in = .false.
  logical,save :: l_netcdf_out = .false.

  integer(4) :: nv
  character(clen) :: iomode_in, iomode_out
  integer(4) :: iostat
  integer(4) :: i,j,k,n

  !---------------------------

  namelist /nml_vert_intp/ &
       & src_namelist, dst_namelist, &
       & shape, cgrid, &
       & file_in, file_out, &
       & nvar, &
       & name_in, &
       & l_netcdf_in, l_netcdf_out, &
       & deflate_level_in, deflate_level_out, &
       & undef_in, undef_out

  !------ preprocessing ------

  !-- arguments --

  rewind(5)
  file_in='no_input'
  undef_in = 0.0
  undef_out = 0.0
  read(5,nml=nml_vert_intp,iostat=i)
  if ( i /= 0 ) then
    write(*,*) ' Read error : namelist nml_vert_intp'
    stop
  endif

  write(6,*) ' General checking '
  
  if ((trim(cgrid)=='X') .or. (trim(cgrid)=='Y')) then
    write(6,*) 'For cgrid = X or Y, program is not supported '
    stop
  end if

  if (file_in == 'no_input') then
    stop
  end if

  rmiss8 = real(undef_in,8)

  !------------------------------------------------------

  !-- EXP settings --

  write(6,*) 'Registering grid '
  
  call libmxe_para__register(para_src &
             & ,file_namelist=src_namelist)
  call libmxe_grid__register(grid_src, para_src)
  call libmxe_io__register(io_src, para_src)

  src_im = para_src%imut
  src_jm = para_src%jmut
  src_km = para_src%km + 1

  nrec_in_last = io_src%nm

  call libmxe_para__register(para_dst &
             & ,file_namelist=dst_namelist)
  call libmxe_grid__register(grid_dst, para_dst)
  call libmxe_io__register(io_dst, para_dst)
  dst_im = para_dst%imut
  dst_jm = para_dst%jmut
  dst_km = para_dst%km + 1

  nrec_out_last = io_src%nm

  allocate(r3d_in(1:src_im,1:src_jm,1:src_km))
  allocate(datm(1:src_km))
  allocate(dep_src(1:src_km))

  allocate(r3d_out(1:dst_im,1:dst_jm,1:dst_km))
  allocate(d3d_out(1:dst_im,1:dst_jm,1:dst_km))
  allocate(datn(1:dst_km))
  allocate(dep_dst(1:dst_km))

  allocate(mask_in(1:src_im,1:src_jm,1:src_km))
  mask_in(:,:,:) = 0.0d0
  allocate(mask_out(1:dst_im,1:dst_jm,1:src_km))
  mask_out(:,:,:) = 0.0d0

  !-----------------------------------
  ! File for Input data

  write(6,*) 'Input = ', trim(file_in)

  if ( l_netcdf_in ) then
    iomode_in = 'netcdf'
  else
    iomode_in = 'grads'
  end if

  dep_src(1:src_km) = grid_src%dep(1:src_km) * 1.d-2
  call io_interface__register( var_in, para_src, grid_src, trim(iomode_in), &
       & trim(file_in), nvar, name_in, shape, cgrid, src_km, dep=dep_src, &
       & deflate_level=deflate_level_in )

  ! File for Output data

  do nv = 1, nvar
    name_out(nv) = trim(name_in(nv))
    write(6,*) 'Variable ',nv
    write(6,*) '  name = ', trim(name_out(nv))
    write(6,*) '  standard_name = ', trim(standard_name(nv))
    write(6,*) '  unit = ', trim(unit(nv))
  end do

  write(6,*) 'Output (on parent) = ', trim(file_out)

  if ( l_netcdf_out ) then
    iomode_out = 'netcdf'
  else
    iomode_out = 'grads'
  end if

  dep_dst(1:dst_km) = grid_dst%dep(1:dst_km) * 1.d-2
  call io_interface__register( var_out, para_dst, grid_dst, trim(iomode_out), &
       & trim(file_out), nvar, name_out, shape, cgrid, dst_km, dep=dep_dst, &
       & unit=unit(:nvar), standard_name=standard_name(:nvar), &
       & deflate_level=deflate_level_out )

  write(6,*) 'size (org) ',src_im, src_jm, src_km
  write(6,*) 'size (new) ',dst_im, dst_jm, dst_km

  nrec_in  = 1
  nrec_out = 1
  write(6,*) nrec_in, nrec_in_last

end subroutine ini
!============================================================
!============================================================
subroutine read_and_write

  use io_interface,only:    &
       & io_interface__read, &
       & io_interface__write

  implicit none

  integer(4) :: nv
  integer(4) :: i,j,k,n
  integer(4) :: dst_i, dst_j, dst_k
  integer(4) :: src_i, src_j, src_k
  integer(4) :: mav, nav

  !----------------------------------------------------------

  write(6,*) 'read_and_write'

  do nv = 1, nvar

    d3d_out(:,:,:) = 0.0d0
    r3d_out(:,:,:) = 0.0e0

    !-----

    write(6,*) 'reading original data', nrec_in, nv
    call io_interface__read( para_src, io_src, var_in, nrec_in, r3d_in )

    do j = 1, src_jm
      do i = 1, src_im

        mav = 0
        do k = 1, src_km
          if (r3d_in(i,j,k) /= undef_in) then
            datm(k) = r3d_in(i,j,k)
            mav = k
          else
            exit
          end if
        end do
        if (mav > 0) then
          call vintpl(grid_src%dep,datm,src_km,mav,grid_dst%dep,datn,dst_km,nav,rmiss8)
        else
          datn(:) = rmiss8
        end if
        d3d_out(i,j,:) = datn(:)

      end do
    end do

    write(6,*) 'Transformation O.K.'

    do k = 1, dst_km
      do j = 1, dst_jm
        do i = 1, dst_im
          if (d3d_out(i,j,k) /= rmiss8) then
            r3d_out(i,j,k) = real(d3d_out(i,j,k),4)
          else
            r3d_out(i,j,k) = undef_out
          end if
        end do
      end do
    end do

    call io_interface__write( para_dst, io_dst, var_out, nrec_out, r3d_out(:,:,:), nv )

  end do

end subroutine read_and_write
!------------------------------------------------------------
function has_next( )
  logical :: has_next
  if ( nrec_in > nrec_in_last .and. nrec_out > nrec_out_last ) then
    has_next = .false.
  else
    has_next = .true.
  endif

end function has_next
!------------------------------------------------------------
!------------------------------------------------------------
subroutine next
  implicit none

  nrec_in  = nrec_in + 1
  nrec_out = nrec_out + 1
 
end subroutine next
!------------------------------------------------------------

end module vertical_interpolation
