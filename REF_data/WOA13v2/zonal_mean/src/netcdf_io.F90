! -*-F90-*-
!- For NetCDF I/O
module netcdf_io
  use libmxe_para, only: clen, type_libmxe_para
  use libmxe_io,   only: type_libmxe_io
#ifdef MXE_NETCDF
  use netcdf
#endif /* MXE_NETCDF */
  implicit none
  private


  type,public :: type_netcdf
    character(clen)         :: file_base
    integer(4)              :: nvars
    character(clen),pointer :: name(:)          => null()
    character(clen),pointer :: standard_name(:) => null()
    character(clen),pointer :: unit(:)          => null()
    integer(4)           :: ncid
    integer(4)           :: ndims_world
    integer(4),pointer   :: ndim_world(:) => null()
    character(5),pointer :: cdim(:)       => null()
    character(1),pointer :: caxis(:)      => null()
    character(clen),pointer :: axis_unit(:)    => null()
    logical,pointer      :: dummy_dim(:)  => null()
    real(8),pointer      :: lon(:)        => null()
    real(8),pointer      :: lat(:)        => null()
    real(8),pointer      :: dep(:)        => null()
    character(clen),pointer :: basin_name(:)    => null()
    integer(4),pointer   :: varid(:)      => null()
    integer(4)           :: timevarid
    integer(4)           :: ndims
    integer(4),pointer   :: start(:)      => null()
    integer(4),pointer   :: ndim(:)       => null()
    character(clen)      :: time_unit
    integer(4)           :: deflate_level
    integer(4),pointer   :: chunksize(:)  => null()
    real(4)              :: rundef
  end type type_netcdf


  !-- subroutine --
  public :: netcdf_io__open
  public :: netcdf_io__close
  public :: netcdf_io__read_attribute

  public :: netcdf_io__read
  interface netcdf_io__read
    module procedure netcdf_io__read_3d
    module procedure netcdf_io__read_2d
    module procedure netcdf_io__read_1d
    module procedure netcdf_io__read_0d
  end interface netcdf_io__read

  public :: netcdf_io__write
  interface netcdf_io__write
    module procedure netcdf_io__write_3d
    module procedure netcdf_io__write_2d
    module procedure netcdf_io__write_1d
    module procedure netcdf_io__write_0d
  end interface netcdf_io__write

contains
!============================================================

  subroutine netcdf_io__open( io, netcdf, nrec, action, nvar )
    use libmxe_io, only: libmxe_io__suffix
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in)           :: nrec
    character(*),intent(in)         :: action
    integer(4),intent(in),optional  :: nvar

    character(clen) :: cfile
    integer(4) :: mode, dimid_tmp, dimid_tmp2
    integer(4),allocatable  :: dimid(:), dimvarid(:)

    character(clen) :: time_unit
    character(6)    :: time_intv
    integer(4) :: time_int_array(6)

    integer(4) :: n, nmax, nv = 1, nd
    integer(4) :: nrec_ini

#ifdef MXE_NETCDF

    if (present(nvar)) nv = nvar

    !-- check --
    if ( .not. io%ldef ) then
      write(6,*) 'Error at netcdf_io__open'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at netcdf_io__open'
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(netcdf%file_base) == 0 )  then
      write(6,*) 'Error at netcdf_io__open: empty file_base'
      stop
    endif

    !-- Add suffix --
    cfile = libmxe_io__suffix(io,netcdf%file_base,nrec)

    !-- set time_unit & time_int --
    time_int_array(1) = io%calint%year
    time_int_array(2) = io%calint%month
    time_int_array(3) = io%calint%day
    time_int_array(4) = io%calint%hour
    time_int_array(5) = io%calint%minute
    time_int_array(6) = io%calint%second
    do n = 6, 1, -1
      if (time_int_array(n) /= 0) then
        nmax = n
        exit
      end if
    end do
    select case ( nmax )
    case ( 1, 2 )
      time_intv = 'day'
    case default
      time_intv = io%timemode
    end select
    if ( io%l_1record_in_file ) then
      nrec_ini = nrec
    else
      do n = nrec, 1, -1
        nrec_ini = n
        if ( io%nrec_file(n) == 1 ) exit
      end do
    end if
    write(time_unit,'(a,i4.4,a,i2.2,a,i2.2,a,i2.2,a,i2.2,a,i2.2)') trim(time_intv)//'s since ', &
         & io%calrec(nrec_ini)%year, '-', io%calrec(nrec_ini)%month, '-', io%calrec(nrec_ini)%day, ' ', &
         & io%calrec(nrec_ini)%hour, ':', io%calrec(nrec_ini)%minute, ':', io%calrec(nrec_ini)%second
    netcdf%time_unit = time_unit

    !-- open --
    select case(trim(action))
    case('read')
      mode = nf90_nowrite
    case('write','readwrite')
      mode = nf90_write
    case default
      write(6,*) 'Error at netcdf_io__open'
      write(6,*) ' unknown action: ', trim(action)
      stop
    end select

    if ( action == 'write' .and. netcdf%start(netcdf%ndims) == 1 .and. nv == 1 ) then
      call check( nf90_create(trim(cfile),nf90_netcdf4,netcdf%ncid) )
      write(6,*) 'file open (write)', trim(cfile)
      allocate( dimid(1:netcdf%ndims), dimvarid(1:netcdf%ndims_world) )
      nd = 0 ! counts vaild dimention
      do n = 1, netcdf%ndims_world-1
        if (netcdf%dummy_dim(n) ) cycle
        if (trim(netcdf%cdim(n)) == 'basin') then
          call check( nf90_def_dim(netcdf%ncid,trim(netcdf%cdim(n)),netcdf%ndim_world(n),dimid_tmp) )
          call check( nf90_def_dim(netcdf%ncid,'str_len',clen,dimid_tmp2) )
          call check( nf90_def_var(netcdf%ncid,trim(netcdf%cdim(n)), &
               &                   nf90_char,(/dimid_tmp2,dimid_tmp/),dimvarid(n)) )
          call check( nf90_put_att(netcdf%ncid,dimvarid(n),'long_name','basin_name') )
        else
          call check( nf90_def_dim(netcdf%ncid,trim(netcdf%cdim(n)), &
               &                   netcdf%ndim_world(n),dimid_tmp) )
          call check( nf90_def_var(netcdf%ncid,trim(netcdf%cdim(n)), &
               &                   nf90_double,dimid_tmp,dimvarid(n)) )
          call check( nf90_put_att(netcdf%ncid,dimvarid(n),'axis',netcdf%caxis(n)) )
          call check( nf90_put_att(netcdf%ncid,dimvarid(n),'units',trim(netcdf%axis_unit(n))) )
        end if
        nd = nd + 1
        dimid(nd) = dimid_tmp
      end do
      call check( nf90_def_dim(netcdf%ncid,'time',nf90_unlimited,dimid(netcdf%ndims)) )
      call check( nf90_def_var(netcdf%ncid,'time',nf90_double, &
           &                   dimid(netcdf%ndims),netcdf%timevarid) )
      call check( nf90_put_att(netcdf%ncid,netcdf%timevarid,'units',trim(time_unit)) )
      call check( nf90_put_att(netcdf%ncid,netcdf%timevarid,'axis','T') )
      !
      if ( netcdf%deflate_level < 1 ) then
        do n = 1, netcdf%nvars
          call check( nf90_def_var(netcdf%ncid,trim(netcdf%name(n)),&
               &                   nf90_float,dimid,netcdf%varid(n), &
               &                   chunksizes=netcdf%chunksize) )
        end do
      else
        do n = 1, netcdf%nvars
          call check( nf90_def_var(netcdf%ncid,trim(netcdf%name(n)),&
               &                   nf90_float,dimid,netcdf%varid(n), &
               &                   deflate_level=netcdf%deflate_level, &
               &                   chunksizes=netcdf%chunksize) )
        end do
      end if
      do n = 1, netcdf%nvars
        call check( nf90_put_att(netcdf%ncid,netcdf%varid(n),'missing_value',netcdf%rundef) )
        call check( nf90_put_att(netcdf%ncid,netcdf%varid(n),'_FillValue',netcdf%rundef) )
      end do
      if (associated(netcdf%standard_name)) then
        do n = 1, netcdf%nvars
          if ( netcdf%standard_name(n) /= '' ) &
               & call check( nf90_put_att(netcdf%ncid,netcdf%varid(n),&
               &                         'standard_name',netcdf%standard_name(n)) )
        end do
      end if
      if (associated(netcdf%unit)) then
        do n = 1, netcdf%nvars
          if ( netcdf%unit(n) /= '' ) &
               & call check( nf90_put_att(netcdf%ncid,netcdf%varid(n),'units',netcdf%unit(n)) )
        end do
      end if
      call check( nf90_enddef(netcdf%ncid) )
      do n = 1, netcdf%ndims_world-1
        if (netcdf%dummy_dim(n) ) cycle
        select case(trim(netcdf%cdim(n)))
        case('lon')
          call check( nf90_put_var(netcdf%ncid,dimvarid(n),netcdf%lon(1:netcdf%ndim_world(n))) )
        case('lat')
          call check( nf90_put_var(netcdf%ncid,dimvarid(n),netcdf%lat(1:netcdf%ndim_world(n))) )
        case('depth')
          call check( nf90_put_var(netcdf%ncid,dimvarid(n),netcdf%dep(1:netcdf%ndim_world(n))) )
        case('basin')
          call check( nf90_put_var(netcdf%ncid,dimvarid(n),netcdf%basin_name(1:netcdf%ndim_world(n)), &
               & start=(/1,1/), count=(/clen,netcdf%ndim_world(n)/)) )
        end select
      end do
    else
      call check( nf90_open(trim(cfile),mode,netcdf%ncid) )
      !call check( nf90_inq_varid(netcdf%ncid,'time',netcdf%timevarid) )
      do n = 1, netcdf%nvars
        call check( nf90_inq_varid(netcdf%ncid,trim(netcdf%name(n)),netcdf%varid(n)) )
      end do
    end if

#else /* MXE_NETCDF */

    write(6,*) 'Error: NetCDF I/O is not supported'
    stop

#endif /* MXE_NETCDF */

  end subroutine netcdf_io__open

!============================================================

  subroutine netcdf_io__read_attribute( io, netcdf, unit, standard_name )
    use libmxe_io, only: libmxe_io__suffix
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout)    :: netcdf
    character(*),intent(out) :: unit(netcdf%nvars)
    character(*),intent(out) :: standard_name(netcdf%nvars)

    character(clen) :: cfile
    integer(4) :: n

#ifdef MXE_NETCDF

    !-- check --
    if ( .not. io%ldef ) then
      write(6,*) 'Error at netcdf_io__open'
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( len(netcdf%file_base) == 0 )  then
      write(6,*) 'Error at netcdf_io__open: empty file_base'
      stop
    endif

    !-- Add suffix --
    cfile = libmxe_io__suffix(io,netcdf%file_base,1)

    !-- open netcdf file --
    call check( nf90_open(trim(cfile),nf90_nowrite,netcdf%ncid) )
    do n = 1, netcdf%nvars
      call check( nf90_inq_varid(netcdf%ncid,trim(netcdf%name(n)),netcdf%varid(n)) )
    end do

    unit(:) = ''
    standard_name(:) = ''
    do n = 1, netcdf%nvars
      call write_error( nf90_get_att(netcdf%ncid,netcdf%varid(n),'units',unit(n)) )
      call write_error( nf90_get_att(netcdf%ncid,netcdf%varid(n),'standard_name',standard_name(n)) )
    end do

    call netcdf_io__close( netcdf )
    
#endif /* MXE_NETCDF */

  end subroutine netcdf_io__read_attribute

!============================================================

  subroutine netcdf_io__common( io, netcdf, nrec, action, nvar )
    use libmxe_calendar, only: libmxe_calendar__diffsec
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    character(*), intent(in) :: action
    integer(4),intent(in) :: nvar

    integer(4) :: idx, nrec_ini, n
    real(8) :: time

#ifdef MXE_NETCDF

    !-- check --
    if ( .not. io%ldef ) then
      write(6,*) 'Error at netcdf_io__'//action
      write(6,*) '  io is not registered.'
      stop
    endif
    if ( nrec > io%nm )  then
      write(6,*) 'Error at netcdf_io__'//action
      write(6,*) '  nrec=',nrec,' > io%nm=',io%nm
      stop
    endif
    if ( len(netcdf%file_base) == 0 )  then
      write(6,*) 'Error at netcdf_io__'//action//': empty file_base'
      stop
    endif

    if ( io%l_1record_in_file ) then
      netcdf%start(netcdf%ndims) = 1
    else
      netcdf%start(netcdf%ndims) = io%nrec_file(nrec)
    end if

    call netcdf_io__open( io, netcdf, nrec, action, nvar=nvar )

    if ( nvar == 1 ) then
      if ( action == 'write' ) then
        if ( io%l_1record_in_file ) then
          nrec_ini = nrec
        else
          do n = nrec, 1, -1
            nrec_ini = n
            if ( io%nrec_file(n) == 1 ) exit
          end do
        end if
        time = real(libmxe_calendar__diffsec(io%calrec(nrec_ini),io%calrec(nrec),io%l_leap_year),8)
        idx = index(netcdf%time_unit," ") - 1
        select case (netcdf%time_unit(1:idx))
        case( 'days' )
          time = time / 86400.d0
        case( 'hours' )
          time = time / 3600.d0
        case( 'minutes' )
          time = time / 60.d0
        end select
        call check( nf90_put_var( netcdf%ncid, netcdf%timevarid, time, &
             &                    start=(/netcdf%start(netcdf%ndims)/) ))
      end if
    end if

#else /* MXE_NETCDF */

    write(6,*) 'Error: NetCDF I/O is not supported'
    stop

#endif /* MXE_NETCDF */

  end subroutine netcdf_io__common

!============================================================

  subroutine netcdf_io__read_3d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(netcdf%ndim(1),netcdf%ndim(2),netcdf%ndim(3))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'read', nv )

#ifdef MXE_NETCDF
    call check( nf90_get_var( netcdf%ncid, netcdf%varid(nv), r, &
         &      start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__read_3d

!------------------------------------------------------------

  subroutine netcdf_io__read_2d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(netcdf%ndim(1),netcdf%ndim(2))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'read', nv )

#ifdef MXE_NETCDF
    call check( nf90_get_var( netcdf%ncid, netcdf%varid(nv), r, &
         &      start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__read_2d

!------------------------------------------------------------

  subroutine netcdf_io__read_1d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r(netcdf%ndim(1))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'read', nv )

#ifdef MXE_NETCDF
    call check( nf90_get_var( netcdf%ncid, netcdf%varid(nv), r, &
         &      start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__read_1d

!------------------------------------------------------------

  subroutine netcdf_io__read_0d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(out) :: r
    integer(4),intent(in),optional :: nvar

    real(4) :: tmp(1)

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'read', nv )

#ifdef MXE_NETCDF
    call check( nf90_get_var( netcdf%ncid, netcdf%varid(nv), tmp, &
         &      start=netcdf%start, count=netcdf%ndim ) )
    r = tmp(1)

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__read_0d


!============================================================

  subroutine netcdf_io__write_3d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(netcdf%ndim(1),netcdf%ndim(2),netcdf%ndim(3))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'write', nv )
#ifdef MXE_NETCDF
    call check( nf90_put_var( netcdf%ncid, netcdf%varid(nv), r, &
         &                    start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__write_3d

!------------------------------------------------------------

  subroutine netcdf_io__write_2d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(netcdf%ndim(1),netcdf%ndim(2))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'write', nv )
#ifdef MXE_NETCDF
    call check( nf90_put_var( netcdf%ncid, netcdf%varid(nv), r, &
         &                    start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__write_2d

!------------------------------------------------------------

  subroutine netcdf_io__write_1d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r(netcdf%ndim(1))
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'write', nv )
#ifdef MXE_NETCDF
    call check( nf90_put_var( netcdf%ncid, netcdf%varid(nv), r, &
         &                    start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__write_1d

!------------------------------------------------------------

  subroutine netcdf_io__write_0d( io, netcdf, nrec, r, nvar )
    implicit none
    type(type_libmxe_io),intent(in) :: io
    type(type_netcdf),intent(inout) :: netcdf
    integer(4),intent(in) :: nrec
    real(4),intent(in) :: r
    integer(4),intent(in),optional :: nvar

    integer(4) :: nv = 1

    if (present(nvar)) nv = nvar

    call netcdf_io__common( io, netcdf, nrec, 'write', nv )
#ifdef MXE_NETCDF
    call check( nf90_put_var( netcdf%ncid, netcdf%varid(nv), (/r/), &
         &                    start=netcdf%start, count=netcdf%ndim ) )

    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__write_0d

!============================================================

  subroutine netcdf_io__close( netcdf )
    implicit none
    type(type_netcdf),intent(in) :: netcdf
#ifdef MXE_NETCDF
    call check( nf90_close(netcdf%ncid) )
#endif /* MXE_NETCDF */
  end subroutine netcdf_io__close

!============================================================

  subroutine check ( status )
    implicit none
    integer(4), intent(in) :: status
#ifdef MXE_NETCDF
    if (status /= nf90_noerr) then
      write(6,*) trim(nf90_strerror(status))
      stop
    end if
#endif /* MXE_NETCDF */
  end subroutine check

!============================================================

  subroutine write_error ( status )
    implicit none
    integer(4), intent(in) :: status
#ifdef MXE_NETCDF
    if (status /= nf90_noerr) write(6,*) trim(nf90_strerror(status))
#endif /* MXE_NETCDF */
  end subroutine write_error

!============================================================
end module netcdf_io
