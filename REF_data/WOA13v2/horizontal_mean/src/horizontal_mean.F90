! -*-F90-*-
!horizontal_mean.F90
!====================================================
!
! Make Horizontal Mean
!
!====================================================
program horizontal_mean_woa

  use libmxe_para, only : &
  &   pi, radian_r, clen

  use oc_mod_param, only : &
  &   imut, jmut, km,      &
  &   ksgm, dz,            &
  &   ro,    cp,           &
  &   para,                &
  &   param_mxe__ini

  use libmxe_para, only: type_libmxe_para, libmxe_para__register
  use libmxe_grid, only: type_libmxe_grid, libmxe_grid__register
  use libmxe_io,   only: type_libmxe_io,   libmxe_io__register
  use libmxe_grads,only: libmxe_grads__make
  
  use io_interface, only: type_io_var, &
       &                  io_interface__register, &
       &                  io_interface__read, &
       &                  io_interface__write

  implicit none

  !----------------------------------------------

  integer(4) :: km_woa
  real(8),allocatable :: dep_woa(:)

  ! 海洋モデル地形

  character(len=clen) :: flin_basin ! basinインデックスファイル
  character(len=clen) :: flout_basin ! basinインデックスファイル
  integer(4), allocatable :: ibas(:,:)      ! basin区分, UV-Grid
  !             0:LAND, 1:ATL, 2:PAC, 3:IND, 4:MED, 9:SO

  real(8), allocatable :: mask(:,:)
  real(8), allocatable :: var(:,:,:)
  real(8), allocatable :: area(:,:)

  real(8), allocatable :: vol_total(:)
  real(8), allocatable :: mask_k(:)
  real(8), allocatable :: var_v(:)
  real(8), allocatable :: varhm(:)

  real(8) :: vol_0_700, vol_0_2000, vol_2000_bottom

  real(4), parameter :: undefgd = -9.99e33
  real(4), allocatable :: d3_r4(:,:,:)
  real(4), allocatable :: d2_r4(:,:)

  ! 入出力ファイル

  integer(4)      :: nstep     ! 読み出しセット数
  character(clen) :: flin_trc  ! 入力ファイル
  character(clen) :: flin_area ! 入力ファイル
  character(clen) :: flout     ! 出力ファイル
  character(clen) :: flout_vol ! ocean volume
  character(clen) :: name_trc = 'thetao'
  character(clen) :: name_area = 'areacello'
  character(clen) :: name_vol = 'volo'

  integer(4) :: ios          !  入出力エラーチェック用
  integer(4), parameter :: mtbas     = 81
  integer(4), parameter :: mtin      = 82
  integer(4), parameter :: mtin_area = 85
  integer(4), parameter :: mtout     = 86

  integer(4) :: irecw
  integer(4) :: i, j, k, m, jj, nb
  real(8)    :: wrk1, wrk2

  type(type_libmxe_io),save :: io_in
  type(type_libmxe_grid),save :: grid

  type(type_libmxe_para),save :: para_cnst
  type(type_libmxe_grid),save :: grid_cnst
  type(type_libmxe_io),save :: io_cnst
  type(type_libmxe_para),save :: para_out
  type(type_libmxe_grid),save :: grid_out
  type(type_libmxe_io),save   :: io_out

  type(type_io_var),save :: io_trc
  type(type_io_var),save :: io_area
  type(type_io_var),save :: io_vol
  type(type_io_var),save :: io_hm

  integer(4),save :: nrec_first !- first record
  integer(4),save :: nrec_last  !-  last record [default: nm]
  logical,save :: l_netcdf_in
  logical,save :: l_netcdf_out
  character(8),save :: iomode_in
  character(8),save :: iomode_out

  logical,save :: l_mks_in   = .false.
  logical,save :: l_mass_in  = .false.
  real(8),save :: factor_dzu = 1.d0

  integer(4) :: nrec

  character(clen) :: unit = ''
  character(clen) :: standard_name = ''
  character(clen) :: unit_arg(1), standard_name_arg(1), name_arg(1)
  integer(4) :: deflate_level = -1

  logical :: l_include_med_in_atl = .false.

  namelist /nml_horz_mean/  flin_trc, flin_area, flin_basin, &
       & flout, flout_vol, flout_basin, &
       & nrec_first, nrec_last, l_netcdf_in, l_netcdf_out, &
       & name_trc, name_area, name_vol, &
       & unit, standard_name, deflate_level

  !==============================================
  ! 海洋モデル格子情報等の準備

  call param_mxe__ini
  call libmxe_grid__register( grid, para )
!  call structure_mxe__ini
  km_woa = km + 1

  allocate(dep_woa(km_woa))

  call libmxe_io__register( io_in,   para )

  call libmxe_para__register( para_cnst, file_namelist='NAMELIST.MXE.const' )
  call libmxe_grid__register( grid_cnst, para_cnst )
  call libmxe_io__register( io_cnst, para_cnst )

  call libmxe_para__register( para_out, file_namelist='NAMELIST.MXE.OUT' )
  call libmxe_grid__register( grid_out, para_out )
  call libmxe_io__register( io_out, para_out )
  
  !----------------------------------------------
  ! 入力パラメタ規定値

  flin_trc = 'dummy'
  flin_area = 'dummy'
  flin_basin = 'basin_map.gd'
  flout     = 'horizontal_mean.gd'
  flout_vol = 'horizontal_vol.gd'
  nrec_first = 1
  nrec_last  = 0
  l_netcdf_in  = .false.
  l_netcdf_out = .false.
  iomode_in  = 'grads'
  iomode_out = 'grads'

  ! 標準入力から読み込み

  read(unit=5, nml_horz_mean)
  write(6,*) 'flin_trc :', trim(flin_trc)
  write(6,*) 'flin_area:', trim(flin_area)
  write(6,*) 'flin_basin:', trim(flin_basin)
  write(6,*) 'flout    :', trim(flout)
  write(6,*) 'flout_basin:', trim(flout_basin)
  write(6,*) 'flout_vol:', trim(flout_vol)
  write(6,*) 'nrec_first:', nrec_first
  write(6,*) 'nrec_last :', nrec_last
  write(6,*) 'l_netcdf_in :', l_netcdf_in
  write(6,*) 'l_netcdf_out:', l_netcdf_out
  write(6,*) 'name_trc :', trim(name_trc)
  write(6,*) 'name_area:', trim(name_area)
  write(6,*) 'name_vol:',  trim(name_vol)
  write(6,*) 'unit     :', trim(unit)
  write(6,*) 'standard_name:', trim(standard_name)
  write(6,*) 'deflate_level:', deflate_level

  !----------------------------------------------

  if ( standard_name == '' ) then
    standard_name_arg(:) = 'horizontal_mean_tracer'
  else
    standard_name_arg(:) = trim(standard_name)
  end if
  unit_arg(:) = trim(unit)

  if ( nrec_last == 0 ) nrec_last = io_in%nm

  if ( ( nrec_first < 1 ) .or. ( nrec_first > io_in%nm ) .or. &
       ( nrec_last  < 1 ) .or. ( nrec_last  > io_in%nm ) ) then
    write(6,*) 'Error at ini'
    write(6,*) '  nrec_first = ', nrec_first
    write(6,*) '  nrec_last  = ', nrec_last
    stop
  end if

  !----------------------------------------------
  ! io_interface__register
  
  if ( l_netcdf_in  ) iomode_in  = 'netcdf'
  if ( l_netcdf_out ) iomode_out = 'netcdf'

  write(6,*) ' km_woa =', km_woa
  dep_woa(1:km_woa) = grid%dep(1:km_woa) * 1.d-2

  ! ad hoc modification for the bottom layer 
  grid%dzm(km_woa) = 2.d0 * grid%dzm(km_woa)

  !----------------------------------------------
  
  name_arg(1) = trim(name_trc)
  call io_interface__register( io_trc, para, grid, trim(iomode_in), &
       & flin_trc, 1, name_arg, 'xyz', 'U', km_woa, dep=dep_woa )


  name_arg(1) = trim(name_area)
  call io_interface__register( io_area, para_cnst, grid_cnst, trim(iomode_in), &
       & flin_area, 1, name_arg, 'xy', 'U', 1 )

  call io_interface__register( io_hm, para_out, grid_out, trim(iomode_out), &
       & flout, 1, (/name_trc/), 'z', 'U', km_woa, &
       & dep=dep_woa, &
       & unit=unit_arg, standard_name=standard_name_arg, &
       & deflate_level=deflate_level )

  call io_interface__register( io_vol, para_cnst, grid_cnst, trim(iomode_out), &
       & flout_vol, 1, (/name_vol/), 'z', 'U', km_woa, &
       & dep=dep_woa, &
       & unit='m3', standard_name='ocean volume', &
       & deflate_level=deflate_level )

  !-- grads control file

  if ( .not. l_netcdf_out ) call libmxe_grads__make(io_hm%grads,para_out,grid_out,io_out)

  !----------------------------------------------

  allocate(ibas(imut,jmut))
  allocate(mask(imut,jmut))
  allocate(area(imut,jmut))

  allocate(var(imut,jmut,km))

  allocate(vol_total(km_woa))
  allocate(mask_k(km_woa))
  allocate(var_v(km_woa))
  allocate(varhm(km_woa))

  allocate(d3_r4(imut,jmut,km))
  allocate(d2_r4(imut,jmut))

  !----------------------------------------------

  call io_interface__read( para_cnst, io_cnst, io_area, 1, d2_r4 )
  area(:,:) = real(d2_r4(:,:),8)

  !----------------------------------------------
  !  basinインデックス読み込み

  open (mtbas, file=trim(flin_basin), access='direct', &
       & convert='big_endian', form='unformatted', recl=4*imut*jmut)
  write(6,*) ' Basin index is read from ', trim(flin_basin)
  read (mtbas, rec=1, iostat=ios) ibas
  close(mtbas)

  if(ios /= 0) write(6,*) 'reading error in file:', trim(flin_basin)

  where(ibas(1:imut, 1:jmut) == 0) ibas(1:imut, 1:jmut) = 3 ! West of Darwin

  mask(1:imut, 1:jmut) = 0.d0

  do j = 1, jmut
    do i = 1, imut
      if (ibas(i,j) > 0 .and. ibas(i,j) /= 53) then
        mask(i,j) = 1.d0
      end if
    end do
  end do

  open (mtbas, file=trim(flout_basin), access='direct', &
       & convert='big_endian', form='unformatted', recl=4*imut*jmut)
  write(6,*) ' Basin index is written to', trim(flout_basin)
  write(mtbas, rec=1, iostat=ios) real(mask(:,:),4)
  write(6,*) ' ..... done '
  close(mtbas)

  !==============================================

  do nrec = nrec_first, nrec_last

    !- read data

    write(6,*) ' READING Data '

    call io_interface__read( para, io_in, io_trc, nrec, d3_r4 )

    write(6,*) ' ....... done '

    vol_total(1:km_woa) = 0.d0
    var_v(1:km_woa) = 0.d0

    vol_0_700 = 0.0d0
    vol_0_2000 = 0.0d0
    vol_2000_bottom = 0.0d0

    do k = 1, km_woa
      do j = 1, jmut
        do i = 1, imut
          if (d3_r4(i,j,k) /= undefgd) then
            vol_total(k) = vol_total(k) + area(i,j) * mask(i,j) * (grid%dzm(k) * 1.d-2)
            var_v(k) = var_v(k) + real(d3_r4(i,j,k),8) * area(i,j) * mask(i,j) * (grid%dzm(k) * 1.d-2)
          end if
        end do
      end do

      if (grid%depm(k) + 1.d-6 < 700.d2) then
        vol_0_700 = vol_0_700 + vol_total(k)
      else if (grid%depm(k) == 700.d2) then
        vol_0_700 = vol_0_700 + vol_total(k) * 0.5d0 * grid%dz_cm(k) / grid%dzm(k)
      end if

      if (grid%depm(k) + 1.d-6 < 2000.d2) then
        vol_0_2000 = vol_0_2000 + vol_total(k)
      else if (grid%depm(k) == 2000.d2) then
        vol_0_2000 = vol_0_2000 + vol_total(k) * 0.5d0 * grid%dz_cm(k) / grid%dzm(k)
      end if
      
      if (grid%depm(k) - 1.d-6 > 2000.d2) then
        vol_2000_bottom = vol_2000_bottom + vol_total(k)
      else if (grid%depm(k) == 2000.d2) then
        vol_2000_bottom = vol_2000_bottom + vol_total(k) * 0.5d0 * grid%dz_cm(k+1) / grid%dzm(k)
      end if
      
    end do

    do k = 1, km_woa
      mask_k(k) = 0.5d0 + sign(0.5d0, vol_total(k) - 1.d-30)
    end do

    do k = 1, km_woa
      varhm(k) = var_v(k) * mask_k(k) / (vol_total(k) + 1.d0 - mask_k(k))
      write(6,*) varhm(k), vol_total(k), grid%dzm(k) * 1.d-2
    end do

    where(mask_k(1:km_woa) < 0.5d0)
      varhm(1:km_woa) = undefgd
    end where

    call io_interface__write( para_out, io_out, io_hm, nrec, real(varhm(1:km_woa),4), 1 )

  end do

  call io_interface__write( para_cnst, io_cnst, io_vol, 1, real(vol_total(1:km_woa),4), 1 )

  open(10,file='woa13v2_volume.txt',form='formatted')
  write(10,*) '0_700m ', vol_0_700
  write(10,*) '0_2000m ', vol_0_2000
  write(10,*) '2000m_bottom ', vol_2000_bottom
  write(10,*) '0_2000m + 2000m_bottom', vol_0_2000 + vol_2000_bottom
  write(10,*) 'Sum of vol ', sum(vol_total(1:km_woa))
  close(10)
  !====================================================
end program horizontal_mean_woa
