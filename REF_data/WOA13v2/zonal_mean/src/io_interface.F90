! -*-F90-*-
!- I/O interface
!
!- Usage:
!-  [read data]
!-    call io_interface__register( ... )
!-    call io_interface__read( ... )
!-  [write data]
!-    call io_interface__register( ... )
!-    call io_interface__write( ... )
!
module io_interface
  use libmxe_para,  only: clen, type_libmxe_para
  use libmxe_grid,  only: type_libmxe_grid
  use libmxe_io,    only: type_libmxe_io
  use libmxe_grads, only: type_grads
  use netcdf_io,    only: type_netcdf
  implicit none
  private

  type,public :: type_io_var
    character(clen) :: iomode
    character(clen) :: file_base
    character(3)    :: shape
    integer(4)      :: nvars
    integer(4),pointer :: ndim(:) => null()
    integer(4) :: lun
    type(type_grads) :: grads
    type(type_netcdf) :: netcdf
  end type type_io_var

  !-- subroutine --
  public :: io_interface__register
  public :: io_interface__read
  interface io_interface__read
    module procedure io_interface__read_3d
    module procedure io_interface__read_2d
    module procedure io_interface__read_1d
    module procedure io_interface__read_0d
  end interface io_interface__read
  public :: io_interface__write
  interface io_interface__write
    module procedure io_interface__write_3d
    module procedure io_interface__write_2d
    module procedure io_interface__write_1d
    module procedure io_interface__write_0d
  end interface io_interface__write


contains
!============================================================

  !- SUBROUTINE io_interface__register
  !-
  !- [required arguments]
  !-  * io_var: type_io_var型構造体
  !-  * para: type_libmxe_para型構造体
  !-  * grid: type_libmxe_grid型構造体
  !-  * iomode: 入出力形式 ('grads' or 'netcdf')
  !-  * file_base: ファイルベース名
  !-  * nvars: 格納変数の数
  !-  * name(nvars): 格納変数の short name
  !-    - iomode='netcdf' では変数読込に使用するため
  !-      入力ファイルと整合する値を指定すること
  !-  * shape: 変数の形
  !-    - 'xyz': 3次元変数 (海氷カテゴリー別変数等も含む)
  !-    -  'xy': 水平2次元変数
  !-    -  'yz': 子午面断面変数
  !-    -  'xz': 東西断面変数
  !-    -   'x': 東西1次元変数
  !-    -   'y': 南北1次元変数
  !-    -   'z': 鉛直1次元変数
  !-    -   '0': スカラー変数
  !-  * cgrid: 水平方向の変数定義位置
  !-    - 'T': T点 (MRICOM W点変数も'T'を指定)
  !-    - 'U': U点
  !-    - 'X': X点 (iomode='grads'の場合、ctlファイルはU点情報を出力)
  !-    - 'Y': Y点 (同上)
  !-  * klayer: 鉛直層数 (shape='xyz','yz','xz','z'のみ有効)
  !-
  !- [optional arguments]
  !-  * ztype: 鉛直方向の変数定義位置 (標準:'center')
  !-    - shape='xyz','yz','xz','z'のみ有効
  !-    - 'center': セル中央
  !-    - 'bottom': セル下端
  !-    -    'top': セル上端 (iomode='netcdf'のみ有効)
  !-  * dep(klayer): 定義点の深さ情報 (出力時使用)
  !-    - shape='xyz','yz','xz','z'のみ有効
  !-    - ztype で指定できない定義位置の場合に使用
  !-    - GrADS形式ctlファイルおよびNetCDF形式出力ファイルの
  !-      鉛直座標情報として使用
  !-  * standard_name(nvars): 格納変数の standard name
  !-  * unit(nvars): 格納変数の単位
  !-  * lun: ファイルオープン時の装置番号 (標準: 61)
  !-  * deflate_level: NetCDF出力時の圧縮レベル (標準:-1)
  !-    -  -1: 圧縮機能未使用
  !-    -   0: 無圧縮
  !-    - 1-9: 圧縮あり (高レベル程処理時間がかかる)
  !-    - -1 と 0 は厳密には異なる
  !-  * rundef: 出力変数の未定義値 (標準: para%rundefout)
  !-  * file_namelist_depth: 定義点深さ情報記載 namelist ファイル名
  !-    - dep(klayer) と使用目的は同じ (同時指定時は dep が優位)
  !-    - /nml_io_interface_depth/
  !-       kmax: 鉛直層数 (klayer と不一致の場合エラー)
  !-       depth(kmax): 深さ情報
  !-
  !- TODO: NetCDF形式入力ファイルから次元情報を取得可能にする
  !-
  subroutine io_interface__register( io_var, para, grid, &
       & iomode, file_base, nvars, name, shape, cgrid, klayer, &
       & ztype, dep, standard_name, unit, lun, deflate_level, rundef, &
       & file_namelist_depth, x_units, y_units, z_units, &
       & basin, basin_name )
    implicit none
    type(type_io_var),    intent(out) :: io_var
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_grid),intent(in) :: grid
    character(*),intent(in) :: file_base
    character(*),intent(in) :: iomode
    integer(4),intent(in)   :: nvars
    character(*),intent(in) :: name(nvars)
    character(*),intent(in) :: shape
    character(*),intent(in) :: cgrid
    integer(4),intent(in) :: klayer
    character(*),intent(in),optional :: ztype
    real(8),intent(in),optional      :: dep(klayer)
    character(*),intent(in),optional :: standard_name(nvars)
    character(*),intent(in),optional :: unit(nvars)
    integer(4),intent(in),optional   :: lun
    integer(4),intent(in),optional   :: deflate_level
    real(4),intent(in),optional :: rundef
    character(*),intent(in),optional :: file_namelist_depth
    character(*),intent(in),optional :: x_units
    character(*),intent(in),optional :: y_units
    character(*),intent(in),optional :: z_units
    integer(4),intent(in),optional :: basin
    character(*),intent(in),optional :: basin_name(basin)

    integer(4) :: ndims_world, ndims
    character(4) :: shape_world
    
    integer(4),parameter :: ndims_max = 5
    integer(4)   :: ndim_world(ndims_max)
    character(5) :: cdim(ndims_max)
    character(1) :: caxis(ndims_max)
    character(clen) :: axis_unit(ndims_max)
    logical      :: dummy_dim(ndims_max)
    integer(4)   :: start(ndims_max)
    integer(4)   :: ndim(ndims_max)

    integer(4) :: i0, i1, j0, j1
    integer(4) :: n, levs
    character(1) :: cgrid_up
    character(clen) :: ztype_low

    integer(4),parameter :: klayer_max = 1000
    integer(4) :: ios
    integer(4) :: kmax
    real(8) :: depth(klayer_max)
    logical :: l_set_depth = .false.

    integer(4),parameter :: lun_default = 61

    namelist /nml_io_interface_depth/ kmax, depth

    io_var%iomode    = to_lower(iomode)
    io_var%file_base = file_base
    io_var%shape     = to_lower(shape)
    io_var%nvars     = nvars
    io_var%lun       = lun_default
    if (present(lun)) io_var%lun = lun

    cgrid_up = to_upper(cgrid)
    if ( present(ztype) ) then
      ztype_low = to_lower(ztype)
    else
      ztype_low = 'center'
    end if

    if ( present(dep) ) then
      l_set_depth = .true.
      depth(1:klayer) = dep(1:klayer)
    end if
    if ( present(file_namelist_depth) ) then
      if ( file_namelist_depth /= '' ) then
        l_set_depth = .true.
        open(io_var%lun,file=file_namelist_depth,form='formatted',status='old',action='read')
        read(io_var%lun,nml=nml_io_interface_depth,iostat=ios)
        if ( ios /= 0 .or. kmax /= klayer ) then
          write(6,*) 'Error at io_interface_register'
          write(6,*) ' wrong nml_io_interface_depth'
          stop
        end if
        close(io_var%lun)
      end if
    end if

    ndims = 1 !- 'time'
    if ( io_var%shape /= '0' ) ndims = ndims + len_trim(io_var%shape)
    do n = 1, ndims-1
      select case (io_var%shape(n:n))
      case('x')
        ndim(n) = para%input_region%ilast - para%input_region%ifirst + 1
      case('y')
        ndim(n) = para%input_region%jlast - para%input_region%jfirst + 1
      case('z')
        ndim(n) = klayer
      case('b')
        if (.not. present(basin)) then
          write(6,*) ' Error! please define basin for ', trim(io_var%shape)
          stop
        end if
        ndim(n) = basin
      case default
        write(6,*) 'Error at io_interface__register'
        write(6,*) ' unknown shape : ', trim(io_var%shape)
        stop
      end select
    end do
    ndim(ndims) = 1
    allocate( io_var%ndim(ndims) )
    io_var%ndim = ndim(1:ndims)

    select case(trim(io_var%iomode))
    case('grads')

      io_var%grads%file_base = io_var%file_base
      io_var%grads%title     = io_var%file_base
      io_var%grads%istr      = para%input_region%ifirst
      io_var%grads%iend      = para%input_region%ilast
      select case (shape)
      case('yz','y','z','0')
        io_var%grads%iend = io_var%grads%istr
      end select
      io_var%grads%jstr      = para%input_region%jfirst
      io_var%grads%jend      = para%input_region%jlast
      select case (shape)
      case('xz','x','z','0')
        io_var%grads%jend = io_var%grads%jstr
      end select
      select case (cgrid)
      case('T','U')
        io_var%grads%cgrid   = cgrid_up
      case default
        write(6,*) 'Attention: libmxe_grads.F90 does not support X/Y-points'
        write(6,*) '             incorrect lat/lon info in a ctl file'
        io_var%grads%cgrid   = 'U'
      end select
      io_var%grads%km        = klayer
      select case (io_var%shape)
      case('xy','x','y','0')
        io_var%grads%ztype = 'surface'
      case default
        io_var%grads%ztype = ztype
        if ( l_set_depth ) then
          io_var%grads%ztype = 'specify'
          allocate( io_var%grads%z(1:klayer) )
          io_var%grads%z(:) = depth(:klayer)
        end if
      end select
      io_var%grads%nvar      = nvars
      levs = klayer
      if ( levs == 1 ) levs = 0
      if (present(standard_name)) then
        do n = 1, nvars
          if ( standard_name(n) /= '' ) then
            write(io_var%grads%var(n),'(A,X,I3,X,A)') trim(name(n)), levs, '99 '//trim(standard_name(n))
          else
            write(io_var%grads%var(n),'(A,X,I3,X,A)') trim(name(n)), levs, '99 '//trim(name(n))
          end if
        end do
      else
        do n = 1, nvars
          write(io_var%grads%var(n),'(A,X,I3,X,A)') trim(name(n)), levs, '99 '//trim(name(n))
        end do
      end if
      if (present(rundef)) io_var%grads%undef = rundef

    case('netcdf')

      do n = 1, ndims-1
        select case (io_var%shape(n:n))
        case('x')
          start(n) = para%input_region%ifirst
        case('y')
          start(n) = para%input_region%jfirst
        case('z')
          start(n) = 1
        case('b')
          start(n) = 1
        end select
      end do
      start(ndims) = 1

      select case (trim(io_var%shape))
      case('xyz','xy')
        shape_world = io_var%shape
      case('xz')
        shape_world = 'xYz'
      case('yz')
        shape_world = 'Xyz'
      case('yzb')
        shape_world = 'Xyzb'
      case('x')
        shape_world = 'xY'
      case('y')
        shape_world = 'Xy'
      case('z')
        shape_world = 'XYz'
      case('0')
        shape_world = 'XY'
      end select

      ndims_world = len_trim(shape_world) + 1
      do n = 1, ndims_world-1
        caxis(n) = to_upper(shape_world(n:n))
        select case(shape_world(n:n))
        case('x')
          ndim_world(n) = para%input_region%ilast - para%input_region%ifirst + 1
          cdim(n) = 'lon'
          dummy_dim(n) = .false.
          if ( present(x_units) ) then
            axis_unit(n) = trim(x_units)
          else
            axis_unit(n) = 'degrees_east'
          end if
        case('y')
          ndim_world(n) = para%input_region%jlast - para%input_region%jfirst + 1
          cdim(n) = 'lat'
          dummy_dim(n) = .false.
          if ( present(y_units) ) then
            axis_unit(n) = trim(y_units)
          else
            axis_unit(n) = 'degrees_north'
          end if
        case('z')
          ndim_world(n) = klayer
          cdim(n) = 'depth'
          dummy_dim(n) = .false.
          if ( present(z_units) ) then
            axis_unit(n) = trim(z_units)
          else
            axis_unit(n) = 'm'
          end if
        case('b')
          if (.not. present(basin)) then
            write(6,*) ' Error! please define basin for', trim(shape_world(n:n))
            stop
          end if
          ndim_world(n) = basin
          cdim(n) = 'basin'
          dummy_dim(n) = .false.
          axis_unit(n) = 'basin_name'
        case('X')
          ndim_world(n) = 1
          cdim(n) = 'lon'
          dummy_dim(n) = .true.
          if ( present(x_units) ) then
            axis_unit(n) = trim(x_units)
          else
            axis_unit(n) = 'degrees_east'
          end if
        case('Y')
          ndim_world(n) = 1
          cdim(n) = 'lat'
          dummy_dim(n) = .true.
          if ( present(y_units) ) then
            axis_unit(n) = trim(y_units)
          else
            axis_unit(n) = 'degrees_north'
          end if
        end select
      end do
      caxis(ndims_world) = 'T'
      ndim_world(ndims_world)  = 1
      cdim(ndims_world)  = 'time'
      dummy_dim(ndims_world) = .false.

      io_var%netcdf%file_base = io_var%file_base
      io_var%netcdf%nvars     = nvars
      allocate(io_var%netcdf%name(nvars))
      io_var%netcdf%name(:)   = name(:)
      if (present(standard_name)) then
        allocate(io_var%netcdf%standard_name(nvars))
        io_var%netcdf%standard_name(:) = standard_name(:)
      end if
      if (present(unit)) then
        allocate(io_var%netcdf%unit(nvars))
        io_var%netcdf%unit(:) = unit(:)
      end if
      !
      io_var%netcdf%ndims_world = ndims_world
      allocate( io_var%netcdf%ndim_world(ndims_world) )
      allocate( io_var%netcdf%cdim(ndims_world) )
      allocate( io_var%netcdf%caxis(ndims_world) )
      allocate( io_var%netcdf%axis_unit(ndims_world) )
      allocate( io_var%netcdf%dummy_dim(ndims_world) )
      io_var%netcdf%ndim_world  = ndim_world(1:ndims_world)
      io_var%netcdf%cdim        = cdim(1:ndims_world)
      io_var%netcdf%caxis       = caxis(1:ndims_world)
      io_var%netcdf%axis_unit(1:ndims_world-1) = axis_unit(1:ndims_world-1)
      io_var%netcdf%dummy_dim   = dummy_dim(1:ndims_world)
      !
      allocate(io_var%netcdf%varid(nvars))
      !
      io_var%netcdf%ndims    = ndims
      allocate( io_var%netcdf%start(ndims) )
      allocate( io_var%netcdf%ndim(ndims) )
      io_var%netcdf%start(:) = start(1:ndims)
      io_var%netcdf%ndim(:)  = ndim(1:ndims)
      !
      i0 = para%input_region%ifirst
      i1 = para%input_region%ilast
      j0 = para%input_region%jfirst
      j1 = para%input_region%jlast
      allocate( io_var%netcdf%lon(i1-i0+1) )
      allocate( io_var%netcdf%lat(j1-j0+1) )
      select case(cgrid_up)
      case('T','Y')
        io_var%netcdf%lon(:) = grid%lont(i0:i1)
      case('U','X')
        io_var%netcdf%lon(:) = grid%lonu(i0:i1)
      end select
      select case(cgrid_up)
      case('T','X')
        io_var%netcdf%lat(:) = grid%latt(j0:j1)
      case('U','Y')
        io_var%netcdf%lat(:) = grid%latu(j0:j1)
      end select
      !
      allocate( io_var%netcdf%dep(klayer) )
      if ( l_set_depth ) then
        io_var%netcdf%dep(:) = depth(:klayer)
      else
        select case(ztype_low)
        case('center')
          io_var%netcdf%dep(1:klayer) = grid%depm(1:klayer) * 1d-2 !- cm -> m
        case('bottom')
          io_var%netcdf%dep(1:klayer) = grid%dep(2:klayer+1)* 1d-2 !- cm -> m
        case('top')
          io_var%netcdf%dep(1:klayer) = grid%dep(1:klayer) * 1d-2 !- cm -> m
        case default
          write(6,*) 'Error at io_interface__register'
          write(6,*) ' unknown ztype: '//trim(ztype_low)
          stop
        end select
      end if
      !
      if (present(basin)) then
        allocate(io_var%netcdf%basin_name(1:basin))
        io_var%netcdf%basin_name(1:basin) = basin_name(1:basin)
      end if
      !
      if (present(deflate_level)) then
        io_var%netcdf%deflate_level = deflate_level
      else
        io_var%netcdf%deflate_level = -1
      end if
      allocate( io_var%netcdf%chunksize(1:ndims) )
      io_var%netcdf%chunksize(:) = io_var%netcdf%ndim(:)
      !-- For 3D variables, chunksize in z-dim is set to 1
      !--  to make chunksize smaller than GrADS default cache size.
      if ( ndims == 4 ) io_var%netcdf%chunksize(3) = 1
      if (present(rundef)) then
        io_var%netcdf%rundef = rundef
      else
        io_var%netcdf%rundef = para%rundefout
      end if

    case default

      write(6,*) 'Error at io_interface__register'
      write(6,*) ' unknown iomode: ', trim(io_var%iomode)
      stop

    end select
    
  end subroutine io_interface__register

!============================================================

  subroutine io_interface__read_3d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__read
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(io_var%ndim(1),io_var%ndim(2),io_var%ndim(3))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_read_3d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__read( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__read_3d

!------------------------------------------------------------

  subroutine io_interface__read_2d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__read
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(io_var%ndim(1),io_var%ndim(2))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_read_2d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__read( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__read_2d

!------------------------------------------------------------

  subroutine io_interface__read_1d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__read
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(io_var%ndim(1))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_read_1d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__read( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__read_1d

!------------------------------------------------------------

  subroutine io_interface__read_0d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__read
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_read_0d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__read( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__read_0d

!============================================================

  subroutine io_interface__write_3d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__write
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(io_var%ndim(1),io_var%ndim(2),io_var%ndim(3))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_write_3d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__write( io, io_var%netcdf, nrec, r, nvar=nv )
    end select

  end subroutine io_interface__write_3d

!------------------------------------------------------------

  subroutine io_interface__write_2d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__write
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(io_var%ndim(1),io_var%ndim(2))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_write_2d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__write( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__write_2d

!------------------------------------------------------------

  subroutine io_interface__write_1d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__write
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(io_var%ndim(1))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_write_1d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__write( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__write_1d

!------------------------------------------------------------

  subroutine io_interface__write_0d( para, io, io_var, nrec, r, nvar )
    use netcdf_io, only: netcdf_io__write
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),  intent(inout) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    select case(trim(io_var%iomode))
    case('grads')
      call io_interface__grads_write_0d( para, io, io_var, nrec, r, nvar=nv )
    case('netcdf')
      call netcdf_io__write( io, io_var%netcdf, nrec, r, nvar=nv )
    end select
      
  end subroutine io_interface__write_0d

!============================================================

  subroutine io_interface__grads_read_3d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(io_var%ndim(1),io_var%ndim(2),io_var%ndim(3))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_read_3d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_read_3d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_read_3d: empty file_base'
      stop
    endif

    reclen = io_var%ndim(1) * io_var%ndim(2) * io_var%ndim(3) * 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='read' )
    read(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_read_3d

!------------------------------------------------------------

  subroutine io_interface__grads_read_2d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(io_var%ndim(1),io_var%ndim(2))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_read_2d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_read_2d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_read_2d: empty file_base'
      stop
    endif

    reclen = io_var%ndim(1) * io_var%ndim(2) * 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='read' )
    read(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_read_2d

!------------------------------------------------------------

  subroutine io_interface__grads_read_1d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(io_var%ndim(1))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_read_1d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_read_1d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_read_1d: empty file_base'
      stop
    endif

    reclen = io_var%ndim(1) * 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='read' )
    read(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_read_1d

!------------------------------------------------------------

  subroutine io_interface__grads_read_0d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_read_0d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_read_0d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_read_0d: empty file_base'
      stop
    endif

    reclen = 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='read' )
    read(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_read_0d

!============================================================

  subroutine io_interface__grads_write_3d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(io_var%ndim(1),io_var%ndim(2),io_var%ndim(3))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_write_3d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_write_3d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_write_3d: empty file_base'
      stop
    endif

    reclen = io_var%ndim(1) * io_var%ndim(2) * io_var%ndim(3) * 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='write' )
    write(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_write_3d

!------------------------------------------------------------

  subroutine io_interface__grads_write_2d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(io_var%ndim(1),io_var%ndim(2))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_write_2d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_write_2d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_write_2d: empty file_base'
      stop
    endif

    reclen = io_var%ndim(1) * io_var%ndim(2) * 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='write' )
    write(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_write_2d

!------------------------------------------------------------

  subroutine io_interface__grads_write_1d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(io_var%ndim(1))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_write_1d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_write_1d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_write_1d: empty file_base'
      stop
    endif

    reclen = io_var%ndim(1) * 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='write' )
    write(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_write_1d

!------------------------------------------------------------

  subroutine io_interface__grads_write_0d( para, io, io_var, nrec, r, nvar )
    use libmxe_io, only: libmxe_io__open
    implicit none
    type(type_libmxe_para),intent(in) :: para
    type(type_libmxe_io),  intent(in) :: io
    type(type_io_var),     intent(in) :: io_var
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1
    integer(4) :: reclen, irec

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef )  then
      write(6,*) 'Error at io_interface__grads_write_0d'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at io_interface__grads_write_0d'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(io_var%file_base) == 0 )  then
      write(6,*) 'Error at io_interface__grads_write_0d: empty file_base'
      stop
    endif

    reclen = 4
    if ( io%l_1record_in_file ) then
      irec = nv
    else
      irec = (io%nrec_file(nrec) - 1) * io_var%nvars + nv
    end if

    call libmxe_io__open( io, io_var%file_base, nrec, reclen, io_var%lun, action='write' )
    write(io_var%lun,rec=irec) r
    close(io_var%lun)

  end subroutine io_interface__grads_write_0d

!============================================================

  function to_lower( string_in ) result( string_out )
    implicit none

    character(*),intent(in)   :: string_in
    character(len(string_in)) :: string_out

    integer(4) :: i, ic

    string_out = string_in
    do i = 1, len(string_in)
      ic = ichar(string_in(i:i))
      if (ic >= 65 .and. ic <= 90) string_out(i:i) = char(ic+32)
    enddo

  end function to_lower

!============================================================

  function to_upper( string_in ) result( string_out )
    implicit none

    character(*),intent(in)   :: string_in
    character(len(string_in)) :: string_out

    integer(4) :: i, ic

    string_out = string_in
    do i = 1, len(string_in)
      ic = ichar(string_in(i:i))
      if (ic >= 97 .and. ic <= 122) string_out(i:i) = char(ic-32)
    enddo

  end function to_upper

!============================================================
end module io_interface
