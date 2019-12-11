!-*-F90-*-
program main

  use libmxe_para, only: libmxe_para__register, clen &
                     & , pi, radius, radian, radian_r, type_libmxe_para
  use libmxe_grid, only: libmxe_grid__register  &
                     & , type_libmxe_grid
  use libmxe_io,   only: type_libmxe_io,   libmxe_io__register

  use io_interface, only: type_io_var, &
       &                  io_interface__register, &
       &                  io_interface__read, &
       &                  io_interface__write

  implicit none

  integer(4) :: nx, ny

  real(4),allocatable :: work4 (:,:)

  integer(4) :: imut, jmut
  real(8),allocatable :: adtnew(:,:)
  real(8),allocatable :: arenew(:,:)

  character(256) :: file_adt_in
  character(256) :: file_adt_out
  character(256) :: adt_in_dir
  character(256) :: adt_in_base
  character(256) :: adt_out_base

  integer(4) :: ibyr, ieyr

  integer(4) :: nt, nyr, nm, nd, irec, i, j, ii, jj, iii, mi, mj
  integer(4) :: itmp, jtmp, ni, nj
  
  integer(4) :: irec1, irec2, irec3

  type(type_libmxe_para) :: adtp, newp
  type(type_libmxe_grid) :: adtg, newg
  type(type_libmxe_io),save   :: io_out
  type(type_io_var),save :: io_adtnew

  integer(4),parameter :: nmonyr = 12
  integer(4) :: ndmon(nmonyr) = (/ 31,28,31,30,31,30,31,31,30,31,30,31 /)

  integer(4) :: nrec
  real(4) :: undef_in, undef_out
  real(8) :: ratio
  
  logical :: l_apply_gauss_filter
  real(8) :: g_half_width_degree
  real(8),allocatable :: adt_ref(:,:), lon_ref(:)
  integer(4) :: i_range
  real(8) :: wgt_g, adt_g, wgt_tmp, dist
  
  !--------------------------------------------------------------------------

  namelist /nml_adt2onedeg/ ibyr, ieyr, &
       & adt_in_dir, adt_in_base, undef_in, &
       & adt_out_base, undef_out, &
       & l_apply_gauss_filter, g_half_width_degree, i_range

  !--------------------------------------------------------------------------

  l_apply_gauss_filter = .false.
  g_half_width_degree = 2.0d0
  i_range = 0
  
  open (10, file='namelist.adt2onedeg')
  read (10, nml=nml_adt2onedeg)
  close(10)

  !--------------------------------------------------------------------------
  ! set grid points

  !-- model settings --

  call libmxe_para__register(newp,file_namelist='NAMELIST.MXE.1x1')
  call libmxe_grid__register(newg,newp)
  call libmxe_io__register( io_out, newp )

  imut = newp%imut
  jmut = newp%jmut

  allocate(adtnew(1:imut,1:jmut))
  allocate(arenew(1:imut,1:jmut))

  write(6,*) ' New grid ', imut, jmut

  write(file_adt_out,'(1a,1a,1a)') 'CMEMS_new/',trim(adt_out_base),'_CMEMS_1x1_monthly_199301-201812.nc'
  call io_interface__register( io_adtnew, newp, newg, 'netcdf', &
       & file_adt_out, 1, 'zos', 'xy', 'U', 1, &
       & unit='m', standard_name='sea_surface_height', &
       & deflate_level=1 &
       & )

  if (l_apply_gauss_filter) then
    allocate (adt_ref(-i_range+1:imut+i_range,1:jmut))
    allocate (lon_ref(-i_range+1:imut+i_range))
    lon_ref(1:imut) = newg%lonu(1:imut)
    lon_ref(-i_range+1:0) = lon_ref(imut-i_range+1:imut) - 360.d0
    lon_ref(imut+1:imut+i_range) = lon_ref(1:i_range) + 360.d0
    write(6,*) lon_ref(-i_range+1:imut+i_range)
  end if
  
  !--------------------------------------------------------------------

  call libmxe_para__register(adtp,file_namelist='NAMELIST.MXE.CMEMS')
  call libmxe_grid__register(adtg,adtp)

  nx = adtp%imut
  ny = adtp%jmut

  allocate(work4(1:nx,1:ny))

  write(6,*) ' original grid ', nx, ny

  !--------------------------------------------------------------------

  nrec = 0
  
  do nyr = ibyr, ieyr

    !write(6,*) nyr
    !write(file_adt_out,'(1a,i4.4)') trim(adt_out_base),nyr
    !write(6,*) ' data written to ....', trim(file_adt_out)
    !open (30,file=file_adt_out,form='unformatted',access='direct',recl=4*imut*jmut)

    do nm = 1, nmonyr

      write(file_adt_in,'(1a,1a,i4.4,1a,1a,1a,i4.4,1a,i2.2)') &
           & trim(adt_in_dir),'/',nyr,'/',trim(adt_in_base),'.y',nyr,'m',nm
      write(6,*) trim(file_adt_in)
      open(10,file=file_adt_in,form='unformatted',status='old',access='direct',action='read',recl=4*nx*ny)

      irec = 0

      read(10,rec=1) work4
      close(10)

      adtnew(:,:) = 0.0d0
      arenew(:,:) = 0.0d0

      do j = 1, jmut
        do i = 1, imut
          do nj = 1, 4
            do ni = 1, 4
              itmp = 4*(i-1) + ni
              jtmp = 4*(j-1) + nj
              if ((itmp > nx) .or. (jtmp > ny)) then
                write(6,'(4i8)') ' Index Error ', itmp, jtmp, i, j
              endif
              if (work4(itmp,jtmp) /= undef_in) then
                adtnew(i,j) = adtnew(i,j) + real(work4(itmp,jtmp),8) * adtg%areau(itmp,jtmp)
                arenew(i,j) = arenew(i,j) + adtg%areau(itmp,jtmp)
              end if
            end do
          end do
        end do
      end do

      !-----------------------------------------------

      do j = 1, jmut
        do i = 1, imut
          ratio = arenew(i,j) / newg%areau(i,j)
          if (ratio >= 0.5d0) then
            adtnew(i,j) = adtnew(i,j) / arenew(i,j)
          else
            adtnew(i,j) = real(undef_out,8)
          end if
        end do
      end do

      if (l_apply_gauss_filter) then
        adt_ref(1:imut,:) = adtnew(1:imut,:)
        adt_ref(-3:0,:) = adtnew(imut-3:imut,:)
        adt_ref(imut+1:imut+4,:) = adtnew(1:4,:)
        do j = 1, jmut
          do i = 1, imut
            if (adtnew(i,j) == real(undef_out,8)) cycle
            wgt_g = 0.0d0
            adt_g = 0.0d0
            loop_nj: do nj = -i_range, i_range
              if ((j+nj < 1) .or. (j+nj > jmut)) cycle
              loop_ni: do ni = -i_range, i_range
                if (adt_ref(i+ni,j+nj) /= real(undef_out,8)) then
                  dist = (lon_ref(i) - lon_ref(i + ni)) ** 2 + (newg%latu(j) - newg%latu(j+nj)) ** 2
                  wgt_tmp = exp(- dist / (2.0d0 * g_half_width_degree**2))
                  wgt_g = wgt_g + wgt_tmp
                  adt_g = adt_g + adt_ref(i+ni,j+nj) * wgt_tmp
                end if
              end do loop_ni
            end do loop_nj
            if (wgt_g > 0.d0) then
              adtnew(i,j) = adt_g / wgt_g
            else
              adtnew(i,j) = real(undef_out,8)
            end if
          end do
        end do
      end if

      nrec = nrec + 1
      call io_interface__write( newp, io_out, io_adtnew, nrec, &
           & real(adtnew(1:imut,1:jmut), 4), 1 )
      !write(30,rec=nm) real(adtnew,4)

    end do

    !close(30)

  end do

end program main
