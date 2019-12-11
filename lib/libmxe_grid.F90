! -*-F90-*-
!- grid information
module libmxe_grid
  use libmxe_para, only: clen, type_libmxe_para
  implicit none
  private


  !-- structure --
  type,public :: type_libmxe_grid

    character(clen) :: namelist
    logical         :: ldef=.false.

    real(8)         :: lon_west_end_of_core
    real(8)         :: lat_south_end_of_core
    real(8)         :: north_pole_lon
    real(8)         :: north_pole_lat
    real(8)         :: south_pole_lon
    real(8)         :: south_pole_lat

    character(clen) :: file_scale

    real(8),pointer :: lont(:), latt(:)
                      !- longitude and latitude at T-grid
    real(8),pointer :: lonu(:), latu(:)   !- at U-grid
    real(8),pointer :: glont(:,:), glatt(:,:) 
               !- geographical longitude and latitude at T-grid
    real(8),pointer :: glonu(:,:), glatu(:,:)  !- at U-grid
    real(8),pointer :: cor(:,:)       !- Coriolis parameter
    real(8),pointer :: dxtdeg(:), dytdeg(:)
               !- longitude and latitude grid spacing

    real(8),pointer :: dz_cm(:)
    real(8),pointer :: dep(:)  !- box-top depth (1:km+1)
    real(8),pointer :: depm(:) !- box-center depth (1:km)
    real(8),pointer :: dzm(:)  !- box-center interval (1:km+1)

    real(8),pointer :: a_bl(:,:), a_br(:,:), a_tl(:,:), a_tr(:,:)
                               !- area of quarter 
                               !                  T----X----T
                               !                  | tl | tr |
                               !             U----Y----U----Y
                               !             |    | bl | br |
                               !             X----T----X----T
                               !             |    |    |
                               !             U----Y----U
                               !
    real(8),pointer :: areau(:,:) !- area of U-box
    real(8),pointer :: dx_bl(:,:), dx_br(:,:), dx_tl(:,:), dx_tr(:,:)
    real(8),pointer :: dy_bl(:,:), dy_br(:,:), dy_tl(:,:), dy_tr(:,:)
                               !- length
    real(8),pointer :: nendidx(:) !- north end index
               !- nendidx(para%jet)=0.5 if para%lfoldnp=.true.

  end type type_libmxe_grid


  !-- subroutine --
  public :: libmxe_grid__register
  public :: libmxe_grid__monotonic_glon
  public :: libmxe_grid__is_horz_grids_same


  private :: libmxe_grid__clear
  private :: libmxe_grid__metric_spherical


  integer,parameter,private :: lun = 88


contains
!-----------------------------------------------------------------


subroutine libmxe_grid__register(grid,para)

  use libmxe_para,    only : pi, radius, radian, radian_r, omega, &
                           & file_default, l_mxe_verbose
  use libmxe_display, only : libmxe_display__nml_block, &
                           & libmxe_display__var
  use libmxe_trnsfrm
  implicit none


  type(type_libmxe_grid),intent(inout) :: grid
  type(type_libmxe_para),intent(inout) :: para

  integer :: i, j, k, ios, im, jm, km, ii, jj
  real(8) :: lat, lon, rlat, rlon !, rot_cos, rot_sin

  character(clen) :: file_scale
  real(8)         :: lon_west_end_of_core
  real(8)         :: lat_south_end_of_core
  real(8)         :: dx_const_deg
  real(8)         :: dy_const_deg
  character(256)  :: file_dxdy_tbox_deg
  real(8)         :: north_pole_lon, north_pole_lat
  real(8)         :: south_pole_lon, south_pole_lat

  integer         :: margin_width_igrid
  integer         :: margin_width_jgrid
  real(8)         :: margin_deg

  real(8),allocatable :: a_bl(:,:), a_br(:,:), a_tl(:,:), a_tr(:,:)
  real(8),allocatable :: dx_bl(:,:), dx_br(:,:), dx_tl(:,:), dx_tr(:,:)
  real(8),allocatable :: dy_bl(:,:), dy_br(:,:), dy_tl(:,:), dy_tr(:,:)
  integer             :: i0, i1, j0, j1

  namelist /nml_horz_grid/ lon_west_end_of_core, lat_south_end_of_core,&
                         & dx_const_deg, dy_const_deg, &
                         & file_dxdy_tbox_deg,         &
                         & margin_width_igrid, margin_width_jgrid
  namelist /nml_poles/     north_pole_lon, north_pole_lat, &
                         & south_pole_lon, south_pole_lat
  namelist /nml_grid_scale/ file_scale


  !-- check --
  if ( .not. para%ldef )  then
    write(*,*) 'Error at libmxe_grid__register'
    write(*,*) '  para is not registered.'
    stop
  endif
  im = para%imut
  jm = para%jmut
  km = para%km

  call libmxe_grid__clear(grid)
  grid%namelist = para%namelist

  allocate( grid%dxtdeg(1:im) )
  allocate( grid%dytdeg(1:jm) )


  !-- vertical grid --
  allocate( grid%dz_cm(1:km) )
  grid%dz_cm(:) = para%dz(:)

  allocate( grid%dzm(km+1), grid%dep(km+1), grid%depm(km) )

  grid%dep(1) = 0.D0
  do k = 2, km+1
    grid%dep(k) = grid%dep(k-1) + grid%dz_cm(k-1)   !- box-top depth
  enddo
   
  grid%dzm(1) = 0.5D0 * grid%dz_cm(1)
  do k = 2, km
    grid%dzm(k) = 0.5D0 * ( grid%dz_cm(k) + grid%dz_cm(k-1) )
  end do
  grid%dzm(km+1) = 0.5D0 * grid%dz_cm(km)

  grid%depm(1) = grid%dzm(1)
  do k = 2, km
    grid%depm(k) = grid%depm(k-1) + grid%dzm(k)  !- box-center depth
  enddo


  !----  longitude/latitude ----

  open( lun, file=trim(para%namelist), status='old' )

  lon_west_end_of_core  = -9.99d33
  lat_south_end_of_core = -9.99d33
  dx_const_deg          = 0.d0
  dy_const_deg          = 0.d0
  file_dxdy_tbox_deg    = file_default
  margin_width_igrid    = 1
  margin_width_jgrid    = 1
  read( lun, nml=nml_horz_grid )
  if ( l_mxe_verbose ) then
    call libmxe_display__nml_block( 'grid blocks', trim(para%namelist) )
    call libmxe_display__var( lon_west_end_of_core, -9.99d33, 'lon_west_end_of_core' )
    call libmxe_display__var( lat_south_end_of_core, -9.99d33, 'lat_south_end_of_core' )
    call libmxe_display__var( dx_const_deg, 0.d0, 'dx_const_deg' )
    call libmxe_display__var( dy_const_deg, 0.d0, 'dy_const_deg' )
    call libmxe_display__var( trim(file_dxdy_tbox_deg), trim(file_default), 'file_dxdy_tbox_deg' )
  else
    write(6,'(a)') '  * '//trim(para%namelist)//' - horz_grid, poles, grid_scale'
  endif

  if ( para%lsub ) then
    margin_width_igrid = 2
    margin_width_jgrid = 2
  else if ( para%lcyclic ) then
    margin_width_igrid = 2
    margin_width_jgrid = 1
  else
    if ( l_mxe_verbose ) then
      call libmxe_display__var( margin_width_igrid, 1, 'margin_width_igrid' )
      call libmxe_display__var( margin_width_jgrid, 1, 'margin_width_jgrid' )
    endif
  endif

  rewind( lun )

  north_pole_lon = 0.d0
  north_pole_lat = 90.d0
  south_pole_lon = 0.d0
  south_pole_lat = -90.d0
  read( lun, nml=nml_poles, iostat=ios )  !- optional
  if ( ios > 0 ) then
    write(*,*) 'ERROR: Reading nml_poles fail.'
    stop
  endif
  if ( ( ios == 0 ).and. l_mxe_verbose ) then
    call libmxe_display__var( north_pole_lon, 0.d0, 'north_pole_lon' )
    call libmxe_display__var( north_pole_lat, 90.d0, 'north_pole_lat' )
    call libmxe_display__var( south_pole_lon, 0.d0, 'south_pole_lon' )
    call libmxe_display__var( south_pole_lat, -90.d0, 'south_pole_lat' )
  endif
  close( lun )

  grid%lon_west_end_of_core = lon_west_end_of_core
  grid%lat_south_end_of_core = lat_south_end_of_core

  grid%north_pole_lon = north_pole_lon
  grid%north_pole_lat = north_pole_lat
  if ( para%lspherical ) then
    grid%south_pole_lon = north_pole_lon + 180.d0
    grid%south_pole_lat = - north_pole_lat
  else
    grid%south_pole_lon = south_pole_lon
    grid%south_pole_lat = south_pole_lat
  endif

  !-- Set dxtdeg and dytdeg, and calc lon/lat. See MRI.COM gridm.F90

  allocate( grid%dxtdeg(1:im), grid%dytdeg(1:jm) )
  allocate( grid%lont(1:im), grid%lonu(1:im) )
  allocate( grid%latt(1:jm), grid%latu(1:jm) )

  if ( trim(file_dxdy_tbox_deg) /= trim(file_default) ) then

    open( lun, file=trim(file_dxdy_tbox_deg), form='unformatted', &
         & status='old' )
    read(lun) i, j
    if ( ( i == im ).and.( j == jm ) ) then
      read(lun) grid%dxtdeg
      read(lun) grid%dytdeg
    else
      write(*,*) 'ERROR: inconsistent size of ',trim(file_dxdy_tbox_deg)
      write(*,*) '  array size: ',i,j
      stop
    endif
    close( lun )

    select case ( margin_width_igrid )
    case( 0 )
      margin_deg = 0.d0
    case( 1 )
      margin_deg = 0.5d0*grid%dxtdeg(1) + 0.5d0*grid%dxtdeg(2)
    case( 2: )
      margin_deg = 0.5d0*grid%dxtdeg(1) &
                 & + sum(grid%dxtdeg(2:margin_width_igrid)) &
                 & + 0.5d0*grid%dxtdeg(margin_width_igrid+1)
    case default
      write(6,*) 'Error: margin_width_igrid = ', margin_width_igrid
      write(6,*) 'Terminating...'
      stop
    end select
    grid%lont(1) = grid%lon_west_end_of_core - margin_deg

    select case ( margin_width_jgrid )
    case( 0 )
      margin_deg = 0.d0
    case( 1 )
      margin_deg = 0.5d0*grid%dytdeg(1) + 0.5d0*grid%dytdeg(2)
    case( 2: )
      margin_deg = 0.5d0*grid%dytdeg(1) &
                 & + sum(grid%dytdeg(2:margin_width_jgrid)) &
                 & + 0.5d0*grid%dytdeg(margin_width_jgrid+1)
    case default
      write(6,*) 'Error: margin_width_jgrid = ', margin_width_jgrid
      write(6,*) 'Terminating...'
      stop
    end select
    grid%latt(1) = grid%lat_south_end_of_core - margin_deg

    grid%latu(1) = grid%latt(1) + 0.5d0*grid%dytdeg(1)
    grid%lonu(1) = grid%lont(1) + 0.5d0*grid%dxtdeg(1)
    do i = 2, im
      grid%lont(i) = grid%lont(i-1) + 0.5d0*( grid%dxtdeg(i-1) + grid%dxtdeg(i) )
      grid%lonu(i) = grid%lonu(i-1) + grid%dxtdeg(i)
    end do
    do j = 2, jm
      grid%latt(j) = grid%latt(j-1) + 0.5d0*( grid%dytdeg(j-1) + grid%dytdeg(j) )
      grid%latu(j) = grid%latu(j-1) + grid%dytdeg(j)
    end do

  else
    grid%dxtdeg(1:im) = dx_const_deg
    grid%dytdeg(1:jm) = dy_const_deg
      do i = 1, im
        ii = i - 1 - margin_width_igrid
        grid%lont(i) = lon_west_end_of_core + dble(ii)*dx_const_deg
        grid%lonu(i) = lon_west_end_of_core + ( dble(ii) + 0.5d0 )*dx_const_deg
      enddo
      do j = 1, jm
        jj = j - 1 - margin_width_jgrid
        grid%latt(j) = lat_south_end_of_core + dble(jj)*dy_const_deg
        grid%latu(j) = lat_south_end_of_core + ( dble(jj) + 0.5d0 )*dy_const_deg
      enddo
  endif

  if (para%ltripolar) then  
    do j = 2, jm-1
      if ( ( grid%latu(j+1) > grid%north_pole_lat ) &
          & .and.( grid%latu(j) < grid%north_pole_lat ) ) then
        grid%latt(j+1)=grid%north_pole_lat
        exit
      endif
    enddo
  endif

  !---- geographical latitude/longitude ----
  !write(6,*) ' setting geographical longitude/latitude '
  allocate( grid%glatu(im,jm), grid%glatt(im,jm) )
  allocate( grid%glonu(im,jm), grid%glont(im,jm) )
  allocate( grid%cor(im,jm) )


  if ( para%lspherical ) then

    do j = 1, jm
      grid%glatu(:,j) = grid%latu(j)
      grid%glatt(:,j) = grid%latt(j)
    enddo
    do i = 1, im
      grid%glonu(i,:) = grid%lonu(i)
      grid%glont(i,:) = grid%lont(i)
    enddo
    do j = 1, jm
      do i = 1, im
        grid%cor(i,j) = 2.d0 * omega * sin(grid%glatu(i,j)/radian)
      enddo
    enddo

  else

    call set_abc( grid%north_pole_lat, grid%north_pole_lon, &
                & grid%south_pole_lat, grid%south_pole_lon )
    do j = 1, jm
      do i = 1, im

        !- U-points
        rlat = grid%latu(j) * radian_r
        rlon = grid%lonu(i) * radian_r
        call mp2lp(lon, lat, rlon, rlat)
!        call rot_mp2lp(rot_cos, rot_sin, lon, lat, rlon, rlat)
!       Skip this, since rot_cos and rot_sin are not used.
        grid%cor(i,j) = 2.d0 * omega * sin(lat)
        grid%glatu(i,j) = lat * radian
        lon = lon * radian
        if ( lon < 0.d0 ) then
          grid%glonu(i,j) = lon + 360.d0
        else if ( lon >= 360.d0) then
          grid%glonu(i,j) = lon - 360.d0
        else
          grid%glonu(i,j) = lon
        endif

        !- T-points
        rlat = grid%latt(j) * radian_r
        rlon = grid%lont(i) * radian_r
        call mp2lp(lon, lat, rlon, rlat)
!        call rot_mp2lp(rot_cos, rot_sin, lon, lat, rlon, rlat)
        grid%glatt(i,j) = lat * radian
        lon = lon * radian
        if ( lon < 0.d0 ) then
          grid%glont(i,j) = lon + 360.d0
        else if ( lon >= 360.d0) then
          grid%glont(i,j) = lon - 360.d0
        else
          grid%glont(i,j) = lon
        endif

      enddo
    enddo

    if ( para%lfoldnp ) then
      do i = 1, im
        grid%cor(i,jm-2) = grid%cor(im-i+1,jm-3)
        grid%cor(i,jm-1) = grid%cor(im-i+1,jm-4)
        grid%cor(i,jm  ) = grid%cor(im-i+1,jm-5)
      enddo
    endif

  endif


  !------ scale factors (unit area and length) ------
  !write(6,*) ' determine scale factors (unit area and length) '
  allocate( a_bl(im, jm) )
  allocate( a_br(im, jm) )
  allocate( a_tl(im, jm) )
  allocate( a_tr(im, jm) )
  allocate( dx_bl(im, jm) )
  allocate( dx_br(im, jm) )
  allocate( dx_tl(im, jm) )
  allocate( dx_tr(im, jm) )
  allocate( dy_bl(im, jm) )
  allocate( dy_br(im, jm) )
  allocate( dy_tl(im, jm) )
  allocate( dy_tr(im, jm) )


  if ( para%lspherical ) then
    call libmxe_grid__metric_spherical( para, grid, a_bl, a_br, a_tl, a_tr, &
                                      & dx_bl, dx_br, dx_tl, dx_tr,   &
                                      & dy_bl, dy_br, dy_tl, dy_tr )

  else

    open( lun, file=trim(para%namelist), status='old')
    read( lun, nml=nml_grid_scale )
    close( lun )
    grid%file_scale = file_scale

    open( lun, file=trim(grid%file_scale), form='unformatted', &
         & access='sequential' )
      !write(6,*) 'libmxe_grid__register:'
      !write(6,*) '  Reading from ....',trim(fscale)
      read(lun)   a_bl  ! area
      read(lun)   a_br
      read(lun)   a_tl
      read(lun)   a_tr
      read(lun)   dx_bl ! X-ward length
      read(lun)   dx_br
      read(lun)   dx_tl
      read(lun)   dx_tr
      read(lun)   dy_bl ! Y-ward length
      read(lun)   dy_br
      read(lun)   dy_tl
      read(lun)   dy_tr
    close(lun)

  endif


  !-- cyclic condition --
  if ( para%lcyclic ) then
    dx_br(im,1:jm) = dx_br(4,1:jm)
    dx_tr(im,1:jm) = dx_tr(4,1:jm)
  else
    dx_br(im,1:jm) = dx_br(im-1,1:jm)
    dx_tr(im,1:jm) = dx_tr(im-1,1:jm)
  endif

  if ( para%lcyclic ) then
    a_br( im, 1:jm   ) = a_br( 4 , 1:jm   )
    a_tr( im, 1:jm-1 ) = a_tr( 4 , 1:jm-1 )
  else
    a_br( im, 1:jm   ) = a_bl( im , 1:jm   )
    a_tr( im, 1:jm-1 ) = a_tl( im , 1:jm-1 )
  endif


  !-- store data --
  i0 = para%input_region%ifirst
  i1 = para%input_region%ilast
  j0 = para%input_region%jfirst
  j1 = para%input_region%jlast
  allocate( grid%a_bl(i0:i1,j0:j1) )
  allocate( grid%a_br(i0:i1,j0:j1) )
  allocate( grid%a_tl(i0:i1,j0:j1) )
  allocate( grid%a_tr(i0:i1,j0:j1) )
  allocate( grid%dx_bl(i0:i1,j0:j1) )
  allocate( grid%dx_br(i0:i1,j0:j1) )
  allocate( grid%dx_tl(i0:i1,j0:j1) )
  allocate( grid%dx_tr(i0:i1,j0:j1) )
  allocate( grid%dy_bl(i0:i1,j0:j1) )
  allocate( grid%dy_br(i0:i1,j0:j1) )
  allocate( grid%dy_tl(i0:i1,j0:j1) )
  allocate( grid%dy_tr(i0:i1,j0:j1) )

  grid%a_bl(i0:i1,j0:j1)  = a_bl(i0:i1,j0:j1)
  grid%a_br(i0:i1,j0:j1)  = a_br(i0:i1,j0:j1)
  grid%a_tl(i0:i1,j0:j1)  = a_tl(i0:i1,j0:j1)
  grid%a_tr(i0:i1,j0:j1)  = a_tr(i0:i1,j0:j1)
  grid%dx_bl(i0:i1,j0:j1) = dx_bl(i0:i1,j0:j1)
  grid%dx_br(i0:i1,j0:j1) = dx_br(i0:i1,j0:j1)
  grid%dx_tl(i0:i1,j0:j1) = dx_tl(i0:i1,j0:j1)
  grid%dx_tr(i0:i1,j0:j1) = dx_tr(i0:i1,j0:j1)
  grid%dy_bl(i0:i1,j0:j1) = dy_bl(i0:i1,j0:j1)
  grid%dy_br(i0:i1,j0:j1) = dy_br(i0:i1,j0:j1)
  grid%dy_tl(i0:i1,j0:j1) = dy_tl(i0:i1,j0:j1)
  grid%dy_tr(i0:i1,j0:j1) = dy_tr(i0:i1,j0:j1)

  deallocate( a_bl, a_br, a_tl, a_tr )
  deallocate( dx_bl, dx_br, dx_tl, dx_tr )
  deallocate( dy_bl, dy_br, dy_tl, dy_tr )

  !-- areau ( Eq.(3.21) in MRI.COM manual ) --
  allocate( grid%areau(i0:i1,j0:j1) )
  grid%areau = grid%a_bl + grid%a_br + grid%a_tl + grid%a_tr


  !-- nendidx

  allocate( grid%nendidx(1:jm) )
  grid%nendidx(1:jm) = 1.0d0
  if (para%lfoldnp) then
    grid%nendidx(para%jet) = 0.5d0
  endif

  grid%ldef = .true.


end subroutine libmxe_grid__register
!-----------------------------------------------------------------


subroutine libmxe_grid__metric_spherical( para, grid, a_bl, a_br, a_tl, a_tr, &
                                      & dx_bl, dx_br, dx_tl, dx_tr,   &
                                      & dy_bl, dy_br, dy_tl, dy_tr )
  use libmxe_para, only: pi, radius, radian_r
  implicit none

  
  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_grid),intent(in) :: grid
  real(8),intent(out) :: a_bl(para%imut,para%jmut)
  real(8),intent(out) :: a_br(para%imut,para%jmut)
  real(8),intent(out) :: a_tl(para%imut,para%jmut)
  real(8),intent(out) :: a_tr(para%imut,para%jmut)
  real(8),intent(out) :: dx_bl(para%imut,para%jmut)
  real(8),intent(out) :: dx_br(para%imut,para%jmut)
  real(8),intent(out) :: dx_tl(para%imut,para%jmut)
  real(8),intent(out) :: dx_tr(para%imut,para%jmut)
  real(8),intent(out) :: dy_bl(para%imut,para%jmut)
  real(8),intent(out) :: dy_br(para%imut,para%jmut)
  real(8),intent(out) :: dy_tl(para%imut,para%jmut)
  real(8),intent(out) :: dy_tr(para%imut,para%jmut)

  real(8),allocatable,dimension(:) :: lat, dlat, dlon
  real(8),allocatable,dimension(:,:) :: anhft, ashft
  real(8) :: temp1, temp2, deg1, ld
  integer :: i, j, im, jm


  im = para%imut
  jm = para%jmut


  !---- areas ----
  !
  !  T(i,j+1)                             T(i+1,j+1)
  !   +-------------------+-------------------+
  !   |                   |                   |
  !   |      tl(i,j)      |      tr(i,j)      |
  !   |                   |                   |
  !   +-----------------U(i,j)----------------+
  !   |                   |                   |
  !   |      bl(i,j)      |      br(i,j)      |
  !   |                   |                   |
  !   +-------------------+-------------------+
  ! T(i,j)                                 T(i+1,j)
  !  

  !-- anhft and ashft (see p.36 of  MRI.COM manual in Japanese)--
  !- anhft: north-eastern part of T-box
  !-  = (a^2/2) * dlon * cos(lat) * sin(dlat/2) * ( 1 - tan(lat) *tan(dlat/4) )
  !- ashft: south-eastern part
  !-  = (a^2/2) * dlon * cos(lat) * sin(dlat/2) * ( 1 + tan(lat) *tan(dlat/4) )
  allocate( lat(1:jm), dlon(1:im), dlat(1:jm) )  !- radian
  lat = grid%latt * radian_r
  dlon = 0.d0
  dlat = 0.d0
  do i = 2, im
    dlon(i) = ( grid%lonu(i) - grid%lonu(i-1) ) * radian_r
  enddo
  dlon(1) = dlon(2)
  do j = 2, jm
    dlat(j) = ( grid%latu(j) - grid%latu(j-1) ) * radian_r
  enddo
  dlat(1) = dlat(2)


  allocate( anhft(1:im,1:jm), ashft(1:im,1:jm) )
  do j = 1, jm
    do i = 1, im
      if (lat(j) <= -90.D0) then
        temp1 = 0.5d0 * radius**2 * dlon(i) * sin( 0.5d0*dlat(j) )
        temp2 = tan( 0.25d0 * dlat(j) )
        ashft(i,j) = 0.d0
        anhft(i,j) = temp1 * temp2
      else if (lat(j) >= 90.D0) then
        temp1 = 0.5d0 * radius**2 * dlon(i) * sin( 0.5d0*dlat(j) )
        temp2 = tan( 0.25d0 * dlat(j) )
        ashft(i,j) = temp1 * temp2
        anhft(i,j) = 0.d0
      else
        temp1 = 0.5d0 * radius**2 * dlon(i) * cos( lat(j) ) * sin( 0.5d0*dlat(j) )
        temp2 = tan( lat(j) ) * tan( 0.25d0 * dlat(j) )
        anhft(i,j) = temp1 * ( 1.d0 - temp2 )
        ashft(i,j) = temp1 * ( 1.d0 + temp2 )
      end if
    end do
  end do
  deallocate(lat,dlon,dlat)


  !-- a_bl, a_br, a_tl, a_tr --
  a_bl=anhft

  a_br=0.d0
  do j = 1, jm
    do i = 1, im -1
      a_br(i,j)=anhft(i+1,j)
    enddo
  enddo

  a_tl=0.d0
  do j = 1, jm -1
    do i = 1, im
      a_tl(i,j)=ashft(i,j+1)
    enddo
  enddo

  a_tr=0.d0
  do j = 1, jm -1
    do i = 1, im -1
      a_tr(i,j)=ashft(i+1,j+1)
    enddo
  enddo

  deallocate(anhft, ashft)


  !---- grid lengths ----
  !-
  !  T(i,j+1)                               T(i+1,j+1)
  !   +---------------------+---------------------+
  !   |                     |                     |
  !   |dy_tl(i,j)           |dy_tr(i,j)           |
  !   |                     |                     |
  !   |                     |                     |
  !   |      dx_tl(i,j)     |     dx_tr(i,j)      | 
  !   +-------------------U(i,j)------------------+
  !   |                     |                     |
  !   |dy_bl(i,j)           |dy_br(i,j)           |
  !   |                     |                     |
  !   |                     |                     |
  !   |      dx_bl(i,j)     |     dx_br(i,j)      | 
  !   +---------------------+---------------------+
  ! T(i,j)                                     T(i+1,j)

  deg1 = ( 2.d0 * pi * radius ) / 360.d0   !- length of 1 degree [cm]

  !-- dx --
  dx_bl = 0.d0
  dx_tl = 0.d0
  dx_br = 0.d0
  dx_tr = 0.d0

  do j = 1, jm
    do i = 1, im
!      ld = grid%lonu(i) - grid%lont(i)
      ld = 0.5d0 * grid%dxtdeg(i)
      dx_bl(i,j) = deg1 * ld * cos( grid%latt(j) * radian_r )
      dx_tl(i,j) = deg1 * ld * cos( grid%latu(j) * radian_r )
    enddo
  enddo
  do j = 1, jm
    do i = 1, im -1
!      ld = grid%lont(i+1) - grid%lonu(i)
      ld = 0.5d0 * grid%dxtdeg(i+1)
      dx_br(i,j) = deg1 * ld * cos( grid%latt(j) * radian_r )
      dx_tr(i,j) = deg1 * ld * cos( grid%latu(j) * radian_r )
    enddo
  enddo

  !-- dy --
  dy_bl = 0.d0
  dy_tl = 0.d0
  dy_br = 0.d0
  dy_tr = 0.d0

  do j = 1, jm
    do i = 1, im
      dy_bl(i,j) = deg1 * ( grid%latu(j) - grid%latt(j) )
    enddo
  enddo
  dy_br = dy_bl

  do j = 1, jm -1
    do i = 1, im
      dy_tl(i,j) = deg1 * ( grid%latt(j+1) - grid%latu(j) )
    enddo
  enddo
  dy_tl(1:im,jm) = dy_tl(1:im,jm-1)
  dy_tr = dy_tl


end subroutine libmxe_grid__metric_spherical
!-----------------------------------------------------------------


subroutine libmxe_grid__clear(grid)
  implicit none

  type(type_libmxe_grid),intent(out) :: grid

  if ( .not. grid%ldef ) return

  deallocate(grid%lont,grid%latt)
  deallocate(grid%lonu,grid%latu)
  deallocate(grid%glont,grid%glatt)
  deallocate(grid%glont,grid%glatt)
  deallocate(grid%cor)
  deallocate(grid%dxtdeg,grid%dytdeg)
  deallocate(grid%dep,grid%depm,grid%dzm)
  deallocate(grid%a_bl,grid%a_br,grid%a_tl,grid%a_tr)
  deallocate(grid%areau)
  deallocate(grid%dx_bl,grid%dx_br,grid%dx_tl,grid%dx_tr)
  deallocate(grid%dy_bl,grid%dy_br,grid%dy_tl,grid%dy_tr)
  deallocate(grid%nendidx)

  grid%ldef = .false.


end subroutine libmxe_grid__clear
!-----------------------------------------------------------------


subroutine libmxe_grid__monotonic_glon( im, jm, glon, glon_mono  )
  !- Remove a -360 degree jump from longitude array.
  implicit none

  integer,intent(in) :: im,jm
  real(8),intent(in) :: glon(im,jm)
  real(8),intent(out) :: glon_mono(im,jm)

  integer :: i, j, ijump

  glon_mono(:,:) = glon(:,:)

  do j = 1, jm

    ijump = 0
    do i = 2, im
      if ( glon(i,j) >= glon(i-1,j) ) cycle
      ijump = i
      exit
    enddo
    if ( ijump == 0 ) cycle

    glon_mono(ijump:im,j) = glon(ijump:im,j) + 360.d0

  enddo

end subroutine libmxe_grid__monotonic_glon
!-----------------------------------------------------------------


logical function libmxe_grid__is_horz_grids_same( grid1, grid2 )
  implicit none

  type(type_libmxe_grid),intent(in) :: grid1, grid2

  libmxe_grid__is_horz_grids_same = .false.

  if ( size(grid1%lonu) /= size(grid2%lonu) ) return
  if ( size(grid1%latu) /= size(grid2%latu) ) return
  if ( maxval(abs(grid1%lonu - grid2%lonu)) > 0.d0 ) return
  if ( maxval(abs(grid1%latu - grid2%latu)) > 0.d0 ) return

  libmxe_grid__is_horz_grids_same = .true.

end function libmxe_grid__is_horz_grids_same


end module libmxe_grid
