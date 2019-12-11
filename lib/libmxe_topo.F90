! -*-F90-*-
!- topography information
module libmxe_topo
  use libmxe_para, only: clen, type_libmxe_para
  implicit none
  private


  !-- structure --
  type,public :: type_libmxe_topo
    logical         :: ldef=.false.
    character(clen) :: file_topo  !- "all_sea" => skip read file

    integer,pointer :: exnn(:,:), texnn(:,:)
                   !- k of U, T-grid bottom ( 0 means land )
    integer,pointer :: exnnbbl(:,:), texnnbbl(:,:)
    integer,pointer :: ho4(:,:)  !- water depth at U-grid [cm]
    integer,pointer :: ho4bbl(:,:)  !- water depth at U-grid [cm]
    integer,pointer :: depth_t_cm(:,:)  !- at T-grid
    real(8),pointer :: dzu(:,:,:) !- box thickness at U grid
    real(8),pointer :: dzt(:,:,:) !- box thickness at T grid
    real(8),pointer :: dzu1c(:,:,:)
         !-   dzu in sigma layer when SSH=0 (only ksgm >= 1)
    real(8),pointer :: aexl(:,:,:), atexl(:,:,:)
         !- ocean/land index at U, T grids
    real(8),pointer :: mask_x(:,:,:), mask_y(:,:,:)
         !-                  at X, Y grids
    real(8),pointer :: dsgm(:,:,:)
         !- ratio of each sigma layer (1:ksgm)
    integer,pointer :: kbtm(:,:), ktbtm(:,:)
         !- k of U, T-grid bottom (k of interior BBL if BBL exists)
    real(8),pointer :: volt(:,:,:) !- cell volume at T grid
    character(len=clen) :: namelist !- namelist file
  end type type_libmxe_topo


  !-- subroutine --
  public :: libmxe_topo__register
    !- register an object

  private :: libmxe_topo__clear
    !- clear an objcet

  public :: libmxe_topo__aexl
    !- Set aexl and atexl.

  public :: libmxe_topo__make_mask_xy


  public :: libmxe_topo__cell_volume
    !- Set volt

  public :: libmxe_topo__areat
  public :: libmxe_topo__dz3d
    !- Set dzu, dzt and dzu1c.

  public :: libmxe_topo__updatedz
    !- Update dz considering SSH. (only in Sigma layer)

  integer,parameter,private :: lun = 87 


contains 
!-----------------------------------------------------------------


subroutine libmxe_topo__register(topo,para)

  use libmxe_para,    only : l_mxe_verbose
  use libmxe_display, only : libmxe_display__nml_block, &
                           & libmxe_display__var
  implicit none

  type(type_libmxe_topo),intent(inout) :: topo
  type(type_libmxe_para),intent(in) :: para

  character(clen)   :: file_topo
  integer           :: i, j, im, jm, km

  integer,allocatable :: exnn(:,:), texnn(:,:)
  integer,allocatable :: exnnbbl(:,:), texnnbbl(:,:)
  integer,allocatable :: kbtm(:,:), ktbtm(:,:)
  integer,allocatable :: ho4(:,:), ho4bbl(:,:)
  integer,allocatable :: depth_t_cm(:,:)
  integer             :: i0, i1, j0, j1

  namelist /nml_topo/ file_topo

  !-- check --
  if ( .not. para%ldef )  then
    write(*,*) 'Error at libmxe_topo__register'
    write(*,*) '  para is not registered.'
    stop
  endif
  im = para%imut
  jm = para%jmut
  km = para%km

  call libmxe_topo__clear( topo )

  open( lun, file=para%namelist, status='old' )
  read( lun, nml=nml_topo )
  close(lun)

  if ( l_mxe_verbose ) then
    call libmxe_display__nml_block( 'nml_topo', trim(para%namelist) )
    call libmxe_display__var( file_topo, '', 'file_topo' )
  else
    write(6,'(a)') '  * '//trim(para%namelist)//' - topo'
  endif

  topo%file_topo = file_topo

  !---- read topography file ----
  allocate( ho4(im,jm), exnn(im,jm) )

  if ( trim(file_topo) == 'all_sea' ) then
    exnn(:,:) = para%km
    ho4(:,:)  = int( sum( para%dz(1:para%km) ) )
  else
    open( lun, file=trim(topo%file_topo), form='unformatted', status='old' )
    read(lun) ho4, exnn
    close(lun)
  endif

  if (para%lbbl) then
    allocate( ho4bbl(im,jm), exnnbbl(im,jm) )
    if ( trim(file_topo) == 'all_sea' ) then
      exnnbbl(:,:) = 1
      ho4bbl(:,:)  = int( para%dz(para%km) )
    else
      open( lun, file=trim(topo%file_topo), form='unformatted', status='old' )
      read(lun)
      read(lun) ho4bbl, exnnbbl
      close(lun)
    endif
  endif

  topo%namelist = para%namelist

  
  !---- texnn (equivalent to atexl) ----
  allocate( texnn(im,jm) )
  texnn(:,:) = 0
  texnn(1,1) = exnn(1,1)
  do i = 2, im
    texnn(i,1) = max( exnn(i,1), exnn(i-1,1) )
  enddo
  do j = 2, jm
    texnn(1,j) = max( exnn(1,j), exnn(1,j-1) )
  enddo
  do j = 2, jm
    do i = 2, im
      texnn(i,j) = maxval( exnn(i-1:i,j-1:j) )
    enddo
  enddo
  if ( para%lcyclic ) then
    texnn(1,2:jm)    = texnn(im-3,2:jm)
    texnn(2,2:jm)    = texnn(im-2,2:jm)
    texnn(im-1,2:jm) = texnn(3,2:jm)
    texnn(im,2:jm)   = texnn(4,2:jm)
  endif

  if (para%lbbl) then
    allocate( texnnbbl(im,jm) )
    texnnbbl(:,:) = 0
    texnnbbl(1,1) = exnn(1,1)
    do i = 2, im
      texnnbbl(i,1) = max( exnnbbl(i,1), exnnbbl(i-1,1) )
    end do
    do j = 2, jm
      texnnbbl(1,j) = max( exnnbbl(1,j), exnnbbl(1,j-1) )
    end do
    do j = 2, jm
      do i = 2, im
        texnnbbl(i,j) = maxval( exnnbbl(i-1:i,j-1:j) )
      end do
    end do
    if ( para%lcyclic ) then
      texnnbbl(1,2:jm)    = texnnbbl(im-3,2:jm)
      texnnbbl(2,2:jm)    = texnnbbl(im-2,2:jm)
      texnnbbl(im-1,2:jm) = texnnbbl(3,2:jm)
      texnnbbl(im,2:jm)   = texnnbbl(4,2:jm)
    end if
  end if

  !---- kbtm, ktbtm ----
  allocate( kbtm(im,jm), ktbtm(im,jm) )
  kbtm(:,:)  = 1
  ktbtm(:,:) = 1
  if (para%lbbl) then
    where(  exnn > 0 ) kbtm  =  exnn +  exnnbbl
    where( texnn > 0 ) ktbtm = texnn + texnnbbl
  else
    where(  exnn > 0 ) kbtm  =  exnn
    where( texnn > 0 ) ktbtm = texnn
  end if

  !---- depth at T-point ----
  allocate( depth_t_cm(im,jm) )
  depth_t_cm(:,:) = 0
  depth_t_cm(1,1) = ho4(1,1)
  do i = 2, im
    depth_t_cm(i,1) = max( ho4(i,1), ho4(i-1,1) )
  enddo
  do j = 2, jm
    depth_t_cm(1,j) = max( ho4(1,j), ho4(1,j-1) )
  enddo
  do j = 2, jm
    do i = 2, im
      depth_t_cm(i,j) = maxval( ho4(i-1:i,j-1:j) )
    enddo
  enddo
  if ( para%lcyclic ) then
    depth_t_cm(1,2:jm)    = depth_t_cm(im-3,2:jm)
    depth_t_cm(2,2:jm)    = depth_t_cm(im-2,2:jm)
    depth_t_cm(im-1,2:jm) = depth_t_cm(3,2:jm)
    depth_t_cm(im,2:jm)   = depth_t_cm(4,2:jm)
  endif

  !---- store data ----
  i0 = para%input_region%ifirst
  i1 = para%input_region%ilast
  j0 = para%input_region%jfirst
  j1 = para%input_region%jlast

  allocate( topo%ho4(i0:i1,j0:j1), topo%exnn(i0:i1,j0:j1) )
  allocate( topo%texnn(i0:i1,j0:j1) )
  allocate( topo%depth_t_cm(i0:i1,j0:j1) )
  allocate( topo%kbtm(i0:i1,j0:j1), topo%ktbtm(i0:i1,j0:j1) )

  topo%ho4(i0:i1,j0:j1) = ho4(i0:i1,j0:j1)
  topo%texnn(i0:i1,j0:j1) = texnn(i0:i1,j0:j1)
  topo%exnn(i0:i1,j0:j1) = exnn(i0:i1,j0:j1)
  topo%depth_t_cm(i0:i1,j0:j1) = depth_t_cm(i0:i1,j0:j1)
  topo%kbtm(i0:i1,j0:j1) = kbtm(i0:i1,j0:j1)
  topo%ktbtm(i0:i1,j0:j1) = ktbtm(i0:i1,j0:j1)

  deallocate( ho4, exnn, texnn, depth_t_cm, kbtm, ktbtm )

  if (para%lbbl) then
    allocate( topo%ho4bbl(i0:i1,j0:j1), topo%exnnbbl(i0:i1,j0:j1) )
    allocate( topo%texnnbbl(i0:i1,j0:j1) )
    topo%ho4bbl(i0:i1,j0:j1) = ho4bbl(i0:i1,j0:j1)
    topo%exnnbbl(i0:i1,j0:j1) = exnnbbl(i0:i1,j0:j1)
    topo%texnnbbl(i0:i1,j0:j1) = texnnbbl(i0:i1,j0:j1)
    deallocate( ho4bbl, exnnbbl, texnnbbl )
  end if

  topo%ldef = .true.

end subroutine libmxe_topo__register
!-----------------------------------------------------------------


subroutine libmxe_topo__dz3d(topo,para)
  implicit none


  type(type_libmxe_topo),intent(inout) :: topo
  type(type_libmxe_para),intent(in) :: para

  integer :: i, j, k, im, jm, km, ksgm
  real(8) :: depth_sigma_cm
  integer :: i0, i1, j0, j1

  !-- check --
  if ( .not. topo%ldef )  then
    write(*,*) 'Error at libmxe_topo__dz3d'
    write(*,*) '  topo is not registered.'
    stop
  endif

  im = para%imut
  jm = para%jmut
  km = para%km
  ksgm = para%ksgm
  i0 = para%input_region%ifirst
  i1 = para%input_region%ilast
  j0 = para%input_region%jfirst
  j1 = para%input_region%jlast

  !---- dzu ----
  if ( associated(topo%dzu) ) deallocate( topo%dzu )
  allocate( topo%dzu(i0:i1,j0:j1,km) )
  topo%dzu(:,:,:) = 0.d0  ! set 0.d0 at land grids.
  do j = j0, j1
    do i = i0, i1

      do k = 1, topo%exnn(i,j) - 1
        topo%dzu(i, j, k) = para%dz(k)
      enddo

      !- partial cell (bottom)
      k = topo%exnn(i,j)
      if ( k == 1 ) then
        topo%dzu(i, j, k) = dble(topo%ho4(i,j))
      endif
      if ( k >= 2 ) then
        topo%dzu(i, j, k) = dble(topo%ho4(i,j)) &
                        &  - sum( para%dz(1:k-1) )
      endif

    enddo
  enddo

  if (para%lbbl) then
    topo%dzu(:,:,km) = 0.d0
    do j = j0, j1
      do i = i0, i1
        if ( topo%exnn(i,j) > 0 .and. topo%exnnbbl(i,j) > 0) then
          topo%dzu(i,j,km) = dble( topo%ho4bbl(i,j) )
        end if
      end do
    end do
  end if

  !---- dzt ----
  ! NOTE: dzt for interior BBL is zero (this is actually not zero in MRI.COM)
  if ( associated(topo%dzt) ) deallocate( topo%dzt )
  allocate( topo%dzt(i0:i1,j0:j1,km) )
  topo%dzt(:,:,:) = 0.d0
  do k = 1, km
    do j = j0 + 1, j1
      do i = i0 + 1, i1
        topo%dzt(i, j, k) = max ( &
            &   topo%dzu(i-1, j-1, k), topo%dzu(i-1, j, k)   &
            & , topo%dzu(i  , j-1, k), topo%dzu(i  , j, k)  )
      enddo
    enddo
  enddo
  !- TODO: dzt(i0,:) and dzt(:,j0) are all zero. OK?


  !---- dzu1c, dsgm (used in libmxe_topo_updatedz) ----
  if ( ksgm < 1 ) return

  if ( associated(topo%dzu1c) ) deallocate( topo%dzu1c )
  if ( associated(topo%dsgm) ) deallocate( topo%dsgm )
  allocate( topo%dzu1c(i0:i1,j0:j1,ksgm) , topo%dsgm(i0:i1,j0:j1,1:ksgm) )

  topo%dzu1c(:,:,:) = 0.d0
  topo%dsgm(:,:,:) = 0.d0

  do j = j0, j1
    do i = i0, i1

      if ( topo%exnn(i,j) == 0 ) cycle

      depth_sigma_cm = sum(topo%dzu(i,j,1:ksgm))
      do k = 1, ksgm
        topo%dzu1c(i,j,k) = topo%dzu(i,j,k)
        topo%dsgm(i,j,k)  = topo%dzu(i,j,k) / depth_sigma_cm
      enddo

    enddo
  enddo

end subroutine libmxe_topo__dz3d
!-----------------------------------------------------------------


subroutine libmxe_topo__updatedz(ht,topo,para,grid)
  use libmxe_grid, only: type_libmxe_grid
  use libmxe_para,   only: itspnt
  use libmxe_stmrgn, only: libmxe_stmrgn__var3_x, &
       &                   libmxe_stmrgn__var3_n
  implicit none


  type(type_libmxe_topo),intent(inout) :: topo
  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_grid),intent(in) :: grid
  real(8),intent(in) :: ht( para%input_region%ifirst:para%input_region%ilast, &
                          & para%input_region%jfirst:para%input_region%jlast )

  real(8) :: htu
  integer :: i, j, k, im, jm, km, ksgm
  integer :: i0, i1, j0, j1

  !-- check --
  if ( para%ksgm == 0 ) return

  if ( .not. topo%ldef )  then
    write(*,*) 'Error at libmxe_topo__updatedz'
    write(*,*) '  topo is not registered.'
    stop
  endif

  if ( .not. grid%ldef )  then
    write(*,*) 'Error at libmxe_topo__updatedz'
    write(*,*) '  grid is not registered.'
    stop
  endif

  im = para%imut
  jm = para%jmut
  km = para%km
  ksgm = para%ksgm
  i0 = para%input_region%ifirst
  i1 = para%input_region%ilast
  j0 = para%input_region%jfirst
  j1 = para%input_region%jlast

  !-- dzu --
  do j = j0, j1 - 1
    do i = i0, i1 - 1
      if ( topo%exnn(i,j) >= 1 ) then
        htu = ( grid%a_bl(i,j) * ht(i,j) &
             & + grid%a_br(i,j) * ht(i+1,j) &
             & + grid%a_tl(i,j) * ht(i,j+1) &
             & + grid%a_tr(i,j) * ht(i+1,j+1) ) / grid%areau(i,j)
        do k = 1, ksgm
          topo%dzu(i,j,k) = topo%dzu1c(i,j,k) + htu * topo%dsgm(i,j,k)
        enddo
      endif
    enddo
  enddo
  !- dzu(:,j1) and dzu(i1,:) are not updated

  !-- dzt --
  do k = 1, ksgm
    do j = j0 + 1, j1
      do i = i0 + 1, i1
        topo%dzt(i, j, k) = max ( &
            &   topo%dzu(i-1, j-1, k), topo%dzu(i-1, j, k)   &
            & , topo%dzu(i  , j-1, k), topo%dzu(i  , j, k)  )
      enddo
    enddo
  enddo
  !- dzt(:,j0) and dzu(i0,:) are not updated


  !-- volt --
  if (.not. associated(topo%volt)) return

  do k = 1, ksgm
    do j = j0 + 1, j1
      do i = i0 + 1, i1
        topo%volt(i,j,k) = &
             &   grid%a_bl(i  ,j  ) * (topo%dzu1c(i  ,j  ,k) + ht(i,j)*topo%dsgm(i  ,j  ,k)) &
             & + grid%a_br(i-1,j  ) * (topo%dzu1c(i-1,j  ,k) + ht(i,j)*topo%dsgm(i-1,j  ,k)) &
             & + grid%a_tl(i  ,j-1) * (topo%dzu1c(i  ,j-1,k) + ht(i,j)*topo%dsgm(i  ,j-1,k)) &
             & + grid%a_tr(i-1,j-1) * (topo%dzu1c(i-1,j-1,k) + ht(i,j)*topo%dsgm(i-1,j-1,k))
      end do
    end do
  end do

  if ( i0 /= 1 .or. i1 /= im .or. j0 /=1 .or. j1 /= jm ) return

  call libmxe_stmrgn__var3_x( topo%volt, km, itspnt, para )
  call libmxe_stmrgn__var3_n( topo%volt, km, itspnt, 1, para )

end subroutine libmxe_topo__updatedz
!-----------------------------------------------------------------


subroutine libmxe_topo__aexl(topo,para)
  implicit none

  type(type_libmxe_topo),intent(inout) :: topo
  type(type_libmxe_para),intent(in) :: para

  integer :: i, j, k, im, jm, km
  integer :: ifirst, ilast, jfirst, jlast


  !-- check --
  if ( .not. topo%ldef )  then
    write(*,*) 'Error at libmxe_topo__aexl'
    write(*,*) '  topo is not registered.'
    stop
  endif
  im     = para%imut
  jm     = para%jmut
  ifirst = para%input_region%ifirst
  ilast  = para%input_region%ilast
  jfirst = para%input_region%jfirst
  jlast  = para%input_region%jlast
  km     = para%km

  !---- aexl ----
  if (associated(topo%aexl)) deallocate(topo%aexl)
  allocate( topo%aexl(ifirst:ilast,jfirst:jlast,km) )

  topo%aexl(:,:,:) = 1.d0
  do j = jfirst, jlast
    do i = ifirst, ilast
      do k = 1, km
        if ( k > topo%exnn(i,j) ) topo%aexl(i,j,k) = 0.d0
      enddo
    enddo
  enddo

  if (para%lbbl) then
    topo%aexl(:,:,km) = 0.d0
    do j = jfirst, jlast
      do i = ifirst, ilast
        if (topo%exnn(i,j) > 0 .and. topo%exnnbbl(i,j) > 0) then
          topo%aexl(i,j,km) = 1.d0
        end if
      end do
    end do
  end if

  !---- atexl ----
  if (associated(topo%atexl)) deallocate(topo%atexl)
  allocate( topo%atexl(ifirst:ilast,jfirst:jlast,km) )
  topo%atexl(:,:,:) = 0.d0

  ! NOTE: atexl for interior BBL is zero (this is actually unity in MRI.COM)

  do k = 1, km
    topo%atexl(ifirst,jfirst,k) = topo%aexl(ifirst,jfirst,k)

    j = jfirst
    do i = ifirst + 1, ilast
      topo%atexl(i,j,k) = max( topo%aexl(i,j,k) , topo%aexl(i-1,j,k) )
    enddo

    i = ifirst
    do j = jfirst + 1, jlast
      topo%atexl(i,j,k) = max( topo%aexl(i,j,k) , topo%aexl(i,j-1,k) )
    enddo

    do j = jfirst + 1, jlast
      do i = ifirst + 1, ilast
        topo%atexl(i,j,k) = max( topo%aexl(i-1, j-1, k) &
                           &   , topo%aexl(i-1, j, k) &
                           &   , topo%aexl(i  , j-1, k) &
                           &   , topo%aexl(i  , j, k) )
      enddo
    enddo
  enddo

  if ( .not. para%lcyclic ) return
  if ( ( ifirst <= 1 ) .and. ( ilast >= im-3 ) ) then
    topo%atexl(1,:,:) = topo%atexl(im-3,:,:)
  end if

end subroutine libmxe_topo__aexl
!-----------------------------------------------------------------


subroutine libmxe_topo__make_mask_xy( topo,para )
  implicit none


  type(type_libmxe_topo),intent(inout) :: topo
  type(type_libmxe_para),intent(in) :: para

  integer :: i, j, k, km
  integer :: ifirst, ilast, jfirst, jlast

  if ( .not. associated(topo%aexl) ) then
    call libmxe_topo__aexl( topo, para )
  endif

  ifirst = para%input_region%ifirst
  ilast  = para%input_region%ilast
  jfirst = para%input_region%jfirst
  jlast  = para%input_region%jlast
  km     = para%km

  if (associated(topo%mask_x)) deallocate(topo%mask_x)
  allocate( topo%mask_x(ifirst:ilast,jfirst:jlast,km) )

  topo%mask_x(:,jfirst,:) = 0.d0
  do k = 1, km
    do j = jfirst + 1, jlast
      do i = ifirst, ilast
        topo%mask_x(i,j,k) = max(topo%aexl(i,j,k),topo%aexl(i,j-1,k))
        !- same as coefx in MRI.COM
      enddo
    enddo
  enddo

  if (associated(topo%mask_y)) deallocate(topo%mask_y)
  allocate( topo%mask_y(ifirst:ilast,jfirst:jlast,km) )

  topo%mask_y(ifirst,:,:) = 0.d0
  do k = 1, km
    do j = jfirst, jlast
      do i = ifirst + 1, ilast
        topo%mask_y(i,j,k) = max(topo%aexl(i,j,k),topo%aexl(i-1,j,k))
        !- same as coefy in MRI.COM
      enddo
    enddo
  enddo

end subroutine libmxe_topo__make_mask_xy
!-----------------------------------------------------------------


subroutine libmxe_topo__cell_volume( topo, para, grid )
  use libmxe_grid,   only: type_libmxe_grid
  use libmxe_para,   only: itspnt
  use libmxe_stmrgn, only: libmxe_stmrgn__var3_x, &
       &                   libmxe_stmrgn__var3_n
  implicit none

  type(type_libmxe_topo),intent(inout) :: topo
  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_grid),intent(in) :: grid

  integer(4) :: i, j, k
  integer(4) :: ifirst, ilast, jfirst, jlast, km

  !-- check --
  if ( .not. topo%ldef ) then
    write(6,*) 'Error at libmxe_topo__cell_volume'
    write(6,*) '  topo is not registered.'
    stop
  end if
  if ( .not. grid%ldef )  then
    write(*,*) 'Error at libmxe_topo__cell_volume'
    write(*,*) '  grid is not registered.'
    stop
  endif
  if ( .not. associated(topo%dzu) ) then
    call libmxe_topo__dz3d( topo, para )
  end if

  ifirst = para%input_region%ifirst
  ilast  = para%input_region%ilast
  jfirst = para%input_region%jfirst
  jlast  = para%input_region%jlast
  km     = para%km

  if (associated(topo%volt)) deallocate(topo%volt)
  allocate( topo%volt(ifirst:ilast,jfirst:jlast,km) )
  topo%volt(:,:,:) = 0.d0

  do k = 1, para%ksgm
    do j = jfirst + 1, jlast
      do i = ifirst + 1, ilast
        topo%volt(i,j,k) = ( topo%dzu1c(i-1,j-1,k)*grid%a_tr(i-1,j-1) &
             &             + topo%dzu1c(i  ,j-1,k)*grid%a_tl(i  ,j-1) &
             &             + topo%dzu1c(i-1,j  ,k)*grid%a_br(i-1,j  ) &
             &             + topo%dzu1c(i  ,j  ,k)*grid%a_bl(i  ,j  ) &
             &             )
      end do
    end do
  end do
  do k = para%ksgm + 1, km
    do j = jfirst + 1, jlast
      do i = ifirst + 1, ilast
        topo%volt(i,j,k) = ( topo%dzu(i-1,j-1,k)*grid%a_tr(i-1,j-1) &
             &             + topo%dzu(i  ,j-1,k)*grid%a_tl(i  ,j-1) &
             &             + topo%dzu(i-1,j  ,k)*grid%a_br(i-1,j  ) &
             &             + topo%dzu(i  ,j  ,k)*grid%a_bl(i  ,j  ) &
             &             )
      end do
    end do
  end do

  if ( ifirst /= 1 .or. ilast /= para%imut .or. &
       jfirst /= 1 .or. jlast /= para%jmut ) return

  call libmxe_stmrgn__var3_x( topo%volt, km, itspnt, para )
  call libmxe_stmrgn__var3_n( topo%volt, km, itspnt, 1, para )

end subroutine libmxe_topo__cell_volume
!-----------------------------------------------------------------


subroutine libmxe_topo__areat( para, grid, topo, k, areat_cm2 )
  use libmxe_grid,   only: type_libmxe_grid
  use libmxe_para,   only: itspnt
  use libmxe_stmrgn, only: libmxe_stmrgn__var3_x, &
       &                   libmxe_stmrgn__var3_n
  implicit none

  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_grid),intent(in) :: grid
  type(type_libmxe_topo),intent(inout) :: topo
  integer,intent(in)                :: k
  real(8),intent(out)               :: areat_cm2(:,:)

  integer(4) :: i, j
  integer(4) :: ifirst, ilast, jfirst, jlast

  if ( .not. topo%ldef ) then
    write(6,*) 'Error at libmxe_topo__make_areat'
    write(6,*) '  topo is not registered.'
    stop
  end if
  if ( .not. grid%ldef )  then
    write(*,*) 'Error at libmxe_topo__make_areat'
    write(*,*) '  grid is not registered.'
    stop
  endif

  if ( .not. associated(topo%aexl) ) call libmxe_topo__aexl( topo, para )

  ifirst = para%input_region%ifirst
  ilast  = para%input_region%ilast
  jfirst = para%input_region%jfirst
  jlast  = para%input_region%jlast

  do j = jfirst, jlast
    do i = ifirst, ilast

      if ( j==1 ) then
        if ( i==1 ) then
          areat_cm2(i,j) = topo%aexl(i,j,k)*grid%a_bl(i,j)
        else
          areat_cm2(i,j) = topo%aexl(i,j,k)*grid%a_bl(i,j) &
               & + topo%aexl(i-1,j,k)*grid%a_br(i-1,j)
        endif
      else
        if ( i==1 ) then
          areat_cm2(i,j) = topo%aexl(i,j,k)*grid%a_bl(i,j) &
               & + topo%aexl(i,j-1,k)*grid%a_tl(i,j-1)
        else
          areat_cm2(i,j) = topo%aexl(i,j,k)*grid%a_bl(i,j) &
               & + topo%aexl(i-1,j,k)*grid%a_br(i-1,j) &
               & + topo%aexl(i,j-1,k)*grid%a_tl(i,j-1) &
               & + topo%aexl(i-1,j-1,k)*grid%a_tr(i-1,j-1)
        endif
      endif
      !- areat at western and southern ends has value.

    enddo
  enddo


!  if ( ifirst /= 1 .or. ilast /= para%imut .or. &
!       jfirst /= 1 .or. jlast /= para%jmut ) return
!
!  if ( para%lcyclic ) &
!   & call libmxe_stmrgn__var2_x( areat_cm2, km, itspnt, para )
!
!  if ( para%lfoldnp ) &
!   & call libmxe_stmrgn__var2_n( areat_cm2, km, itspnt, 1, para )

end subroutine libmxe_topo__areat
!-----------------------------------------------------------------


subroutine libmxe_topo__clear(topo)
  implicit none

  type(type_libmxe_topo),intent(inout) :: topo

  if ( .not. topo%ldef ) return

  deallocate(topo%exnn,topo%texnn)
  deallocate(topo%ho4)
  deallocate(topo%dzu,topo%dzt,topo%dzu1c)
  deallocate(topo%aexl,topo%atexl)
  deallocate(topo%dsgm)
  topo%ldef = .false.

end subroutine libmxe_topo__clear


end module libmxe_topo
