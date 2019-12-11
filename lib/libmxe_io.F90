! -*-F90-*-
!- Control file input/output.
module libmxe_io
  use libmxe_para, only: clen, type_libmxe_para
  use libmxe_calendar, only: type_calendar
  implicit none
  private


  type,public :: type_libmxe_io
    logical :: ldef=.false. !- .true. : this object is registered
    integer :: nm           !- number of record
    integer :: suffix_digit !- digit of suffix (like year:4)
    character(6) :: timemode  !- time unit of records:
                            !- year/month/day/hour/minute/second
    character(6)        :: suffix       !- time format of file suffix
                                        !-   same as timemode/auto/no
    type(type_calendar) :: calint !- time interval of record
    type(type_calendar),pointer :: calrec(:)
                            !- calrec(1:nm): calendar of record
    character(len=clen) :: namelist !- namelist file
    logical :: l_leap_year = .false.
               !- F: use calendar which ignores leap year
               !- T:   take leap year into account
    integer :: dt_sec
    logical             :: l_1record_in_file
    integer,pointer     :: nrec_file(:)
    integer             :: input_header_rec
                         !- input file header to be skipped [record]
  end type type_libmxe_io


  !-- subroutine --
  public :: libmxe_io__register
  public :: libmxe_io__read_hst

  public :: libmxe_io__read
  interface libmxe_io__read
    module procedure libmxe_io__read_hst
    module procedure libmxe_io__read_rst
  end interface libmxe_io__read

  public :: libmxe_io__read_hst_1grid  !- read only 1 grid point

  public :: libmxe_io__write_hst
  public :: libmxe_io__write
  interface libmxe_io__write
    module procedure libmxe_io__write_hst
    module procedure libmxe_io__write_rst
  end interface libmxe_io__write

  public :: libmxe_io__set_calendar
  public :: libmxe_io__make_dummy

  public :: libmxe_io__open
    !- Open a file to write or read data by direct access.


  !-- function --
  public :: libmxe_io__suffix
    !- Return a file name with a suffix indicating time.

  public :: libmxe_io__nrec_in_file

  integer,parameter,private :: lun=89


contains 
!-----------------------------------------------------------------


subroutine libmxe_io__register( io, para )
  use libmxe_calendar, only: libmxe_calendar__intarr2cal, &
                           & libmxe_calendar__set_config
  use libmxe_display,  only: libmxe_display__nml_block, &
                           & libmxe_display__var
  use libmxe_para,    only : l_mxe_verbose
  implicit none

  type(type_libmxe_io),intent(out)  :: io
  type(type_libmxe_para),intent(in) :: para

  integer             :: rec_first_date(6)   !- Y/M/D/H/M/S
  integer             :: rec_last_date(6)
  integer             :: rec_interval_date(6)
  logical             :: l_leap_year
  character(6)        :: suffix    !- auto/year/month/day/hour/minute/no
  integer             :: input_header_rec

  type(type_calendar) :: first_cal, last_cal, interval_cal

  namelist /nml_record_date/ rec_first_date, rec_last_date, &
                           & rec_interval_date, l_leap_year, suffix, &
                           & input_header_rec


  if ( .not. para%ldef )  then
    write(*,*) 'Error at libmxe_io__register'
    write(*,*) '  para is not registered.'
    stop
  endif

  !-- read namelist --
  rec_first_date(:)    = 0
  rec_last_date(:)     = 0
  rec_interval_date(:) = 0
  l_leap_year          = .false.
  suffix               = 'auto'
  input_header_rec     = 0

  open( lun, file=trim(para%namelist), status='old' )
  read( lun, nml=nml_record_date )
  close(lun)

  if ( l_mxe_verbose ) then
    call libmxe_display__nml_block( 'nml_record_date', trim(para%namelist) )
    call libmxe_display__var( rec_first_date, 0, 'rec_first_date' )
    call libmxe_display__var( rec_last_date, 0, 'rec_last_date' )
    call libmxe_display__var( rec_interval_date, 0, 'rec_interval_date' )
    call libmxe_display__var( l_leap_year, .false., 'l_leap_year' )
    call libmxe_display__var( suffix, 'auto', 'suffix' )
    call libmxe_display__var( input_header_rec, 0, 'input_header_rec' )
  else
    write(6,'(a)') '  * '//trim(para%namelist)//' - record_date'
  endif

  io%l_leap_year = l_leap_year
  io%suffix      = suffix
  io%namelist    = para%namelist
  io%ldef        = .true.
  io%input_header_rec = input_header_rec

  call libmxe_calendar__set_config( l_leap_year )

  first_cal    = libmxe_calendar__intarr2cal( rec_first_date    )
  last_cal     = libmxe_calendar__intarr2cal( rec_last_date     )
  interval_cal = libmxe_calendar__intarr2cal( rec_interval_date )
  call libmxe_io__set_calendar( io, first_cal, last_cal, &
                              & interval_cal )

end subroutine libmxe_io__register
!-----------------------------------------------------------------


!- TODO: merge this subroutine into __register?
subroutine libmxe_io__set_calendar( io, first_cal, last_cal, interval_cal )
  use libmxe_calendar, only: type_calendar, &
                           & libmxe_calendar__diffsec, &
                           & libmxe_calendar__addcal,  &
                           & libmxe_calendar__time_digit
  implicit none

  type(type_libmxe_io),intent(inout) :: io
  type(type_calendar),intent(in)   :: first_cal, last_cal, interval_cal

  integer                          :: n
  character(clen),allocatable      :: csuffix(:)

  io%calint = interval_cal

  !-- time mode (record interval) --
  if ( io%calint%second /= 0 ) then
    io%timemode = 'second'
  else if ( io%calint%minute /= 0 ) then
    io%timemode = 'minute'
  else if ( io%calint%hour /= 0 ) then
    io%timemode = 'hour'
  else if ( io%calint%day /= 0 ) then
    io%timemode = 'day'
  else if ( io%calint%month /= 0 ) then
    io%timemode = 'month'
  else if ( io%calint%year /= 0 ) then
    io%timemode = 'year'
  else
    write(*,*) 'Error at libmxe_io__set_calendar'
    write(*,*) ' Wrong interval_cal:',interval_cal
    stop
  endif

  !-- digit of records suffix --
  if ( io%suffix == 'auto' ) io%suffix = io%timemode
  if ( io%suffix == 'no' ) then
    io%suffix_digit = 0
  else
    io%suffix_digit = libmxe_calendar__time_digit( io%suffix )
  endif

  !-- nm (number of data records) --
  io%dt_sec = 0
  if ( ( io%timemode == 'day' ) &
        .or. ( io%timemode == 'hour' ) &
        .or. ( io%timemode == 'minute' ) &
        .or. ( io%timemode == 'second' ) ) then
    io%dt_sec = io%calint%day * 24 * 3600 + io%calint%hour * 3600 &
        & + io%calint%minute * 60 + io%calint%second 
    io%nm = libmxe_calendar__diffsec( first_cal, last_cal, &
          &  l_leap=io%l_leap_year ) / io%dt_sec + 1
  endif
  if ( io%timemode == 'month' ) then
    io%nm = ( ( last_cal%year - first_cal%year )*12 &
           & + last_cal%month - first_cal%month &
           &) / io%calint%month + 1
  endif
  if ( io%timemode =='year' ) then
    io%nm = ( last_cal%year - first_cal%year ) / io%calint%year + 1
  endif
  if ( io%nm <= 0 ) then
    write(*,*) 'Error at libmxe_io__set_calendar: nm=',io%nm
    stop
  endif

  !-- date of each record --
  allocate( io%calrec(io%nm) )
  io%calrec(1) = first_cal
  do n = 2, io%nm
    io%calrec(n) = libmxe_calendar__addcal( io%calrec(n-1), &
                 &   io%calint, l_leap=io%l_leap_year )
  enddo

  !-- number of records in 1 file --
  if ( io%suffix_digit >= libmxe_calendar__time_digit( io%timemode ) ) then
    io%l_1record_in_file = .true.

  else

    io%l_1record_in_file = .false.

    allocate( csuffix( io%nm ) )
    do n = 1, io%nm
      csuffix(n) = libmxe_io__suffix( io, 'f', n )
    enddo

    allocate( io%nrec_file( io%nm ) )
    io%nrec_file(1) = 1
    do n = 2, io%nm
      if ( csuffix(n) == csuffix(n-1) ) then
        io%nrec_file(n) = io%nrec_file(n-1) + 1
      else
        io%nrec_file(n) = 1
      endif
    enddo

    deallocate( csuffix )

  endif

end subroutine libmxe_io__set_calendar
!-----------------------------------------------------------------


subroutine libmxe_io__make_dummy( io )
  use libmxe_calendar, only : libmxe_calendar__intarr2cal
  implicit none

  type(type_libmxe_io),intent(out) :: io

  io%ldef         = .true.
  io%nm           = 1
  io%suffix_digit = 0
  io%timemode     = 'year'
  io%suffix       = 'no'
  io%calint       = libmxe_calendar__intarr2cal( (/1,0,0,0,0,0/))
  allocate( io%calrec(1) )
  io%calrec(1)    = libmxe_calendar__intarr2cal( (/1001,1,1,0,0,0/))
  io%namelist     = 'dummy'
  io%dt_sec       = 1
  io%l_1record_in_file = .true.
  allocate( io%nrec_file(1) )
  io%nrec_file(1) = 1

end subroutine libmxe_io__make_dummy
!-----------------------------------------------------------------


function libmxe_io__suffix( io, file_base, n )
  use libmxe_calendar, only: libmxe_calendar__cdate_simple
  implicit none

  character(clen) :: libmxe_io__suffix

  type(type_libmxe_io),intent(in) :: io
  character(*),intent(in) :: file_base
                 !- base name of input/output file
  integer,intent(in) :: n  !- record number

  character(15) :: csuffix


  if ( io%suffix=='no' ) then
    libmxe_io__suffix = trim(file_base)
    return
  endif

  !-- check --
  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__suffix'
    write(*,*) '  io is not registered.'
    stop
  endif
  if ( n > io%nm )  then
    write(*,*) 'Error at libmxe_io__suffix'
    write(*,*) '  n=',n,' > io%nm=',io%nm
    stop
  endif

  csuffix = libmxe_calendar__cdate_simple( io%calrec(n), io%suffix )

  libmxe_io__suffix = trim(file_base)//'.'//trim(csuffix)


end function libmxe_io__suffix
!-----------------------------------------------------------------


integer function libmxe_io__nrec_in_file( io, nrec )
  implicit none

  type(type_libmxe_io),intent(in) :: io
  integer,intent(in)              :: nrec

  if ( io%l_1record_in_file ) then
    libmxe_io__nrec_in_file = 1
  else
    libmxe_io__nrec_in_file = io%nrec_file(nrec)
  endif

end function libmxe_io__nrec_in_file
!-----------------------------------------------------------------


subroutine libmxe_io__open( io, file_base, n, reclen, lun, action )
  implicit none

  type(type_libmxe_io),intent(in) :: io
  character(*),intent(in) :: file_base  !- file name (base)
  integer,intent(in) :: lun    !- logical unit number to open
  integer,intent(in) :: n      !- record number
  integer,intent(in) :: reclen !- record length [byte]
  character(*),intent(in),optional :: action

  character(clen) :: cfile


  !-- check --
  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__open'
    write(*,*) '  io is not registered.'
    stop
  endif
  if ( n > io%nm )  then
    write(*,*) 'Error at libmxe_io__open'
    write(*,*) '  n=',n,' > io%nm=',io%nm
    stop
  endif
  if ( len(file_base) == 0 )  then
    write(*,*) 'Error at libmxe_io__open: empty file_base'
    stop
  endif


  !-- Add suffix --    
  cfile = libmxe_io__suffix(io,file_base,n)

  !-- open --
  if (present(action)) then
    open(lun, file=cfile, recl=reclen &
        & , access='direct', form='unformatted', action=action )
  else
    open(lun, file=cfile, recl=reclen &
        & , access='direct', form='unformatted' )
  endif


end subroutine libmxe_io__open
!-----------------------------------------------------------------


subroutine libmxe_io__read_hst( para, io, file_base, nrec, klayer, r )
  implicit none

  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_io),intent(in)   :: io
  character(*),intent(in)           :: file_base
  integer,intent(in) :: nrec   !- record number
  integer,intent(in) :: klayer !- number of record layer
  real(4),intent(out):: r( para%input_region%ifirst:para%input_region%ilast, &
                      &    para%input_region%jfirst:para%input_region%jlast, &
                      &    1:klayer )

  integer,parameter   :: lun = 61
  integer             :: k, reclen, nrec_file
  integer             :: ifirst, ilast, jfirst, jlast
  real(4),allocatable :: rin(:,:)

  !-- check --
  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__read_hst'
    write(*,*) '  io is not registered.'
    stop
  endif
  if ( nrec > io%nm )  then
    write(*,*) 'Error at libmxe_io__read_hst'
    write(*,*) '  nrec=',nrec,' > io%nm=',io%nm
    stop
  endif
  if ( len(file_base) == 0 )  then
    write(*,*) 'Error at libmxe_io__read_hst: empty file_base'
    stop
  endif

  reclen = para%imut * para%jmut * 4
  ifirst = para%input_region%ifirst
  ilast  = para%input_region%ilast
  jfirst = para%input_region%jfirst
  jlast  = para%input_region%jlast

  call libmxe_io__open( io, file_base, nrec, reclen, lun, action='read' )

  nrec_file = libmxe_io__nrec_in_file( io, nrec ) + io%input_header_rec
  allocate( rin(para%imut,para%jmut) )
  do k = 1, klayer
    read(lun,rec=k+(nrec_file-1)*klayer) rin(:,:)
    r(ifirst:ilast,jfirst:jlast,k) = rin(ifirst:ilast,jfirst:jlast)
  enddo
  deallocate( rin )
  close( lun )

end subroutine libmxe_io__read_hst
!-----------------------------------------------------------------


subroutine libmxe_io__read_rst( para, io, file_base, nrec, klayer, r )
  implicit none

  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_io),intent(in)   :: io
  character(*),intent(in)           :: file_base
  integer,intent(in) :: nrec   !- record number
  integer,intent(in) :: klayer !- number of record layer
  real(8),intent(out):: r( para%input_region%ifirst:para%input_region%ilast, &
                      &    para%input_region%jfirst:para%input_region%jlast, &
                      &    1:klayer )

  integer,parameter   :: lun = 61
  integer             :: k, reclen, nrec_file
  integer             :: ifirst, ilast, jfirst, jlast
  real(8),allocatable :: rin(:,:)

  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__read_hst: io is not registered.'
    stop
  endif
  if ( nrec > io%nm )  then
    write(*,*) 'Error at libmxe_io__read_hst: nrec=',nrec,' > io%nm=',io%nm
    stop
  endif
  if ( len(file_base) == 0 )  then
    write(*,*) 'Error at libmxe_io__read_hst: empty file_base'
    stop
  endif

  reclen = para%imut * para%jmut * 8
  ifirst = para%input_region%ifirst
  ilast  = para%input_region%ilast
  jfirst = para%input_region%jfirst
  jlast  = para%input_region%jlast

  call libmxe_io__open( io, file_base, nrec, reclen, lun, action='read' )

  nrec_file = libmxe_io__nrec_in_file( io, nrec ) + io%input_header_rec
  allocate( rin(para%imut,para%jmut) )
  do k = 1, klayer
    read(lun,rec=k+(nrec_file-1)*klayer) rin(:,:)
    r(ifirst:ilast,jfirst:jlast,k) = rin(ifirst:ilast,jfirst:jlast)
  enddo
  deallocate( rin )
  close( lun )

end subroutine libmxe_io__read_rst
!-----------------------------------------------------------------


subroutine libmxe_io__read_hst_1grid( para, io, file_base, km, i, j, k, nrec, r )
  implicit none

  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_io),intent(in)   :: io
  character(*),intent(in)           :: file_base
  integer,intent(in) :: km      !- layer number of history data
  integer,intent(in) :: i, j, k !- array index of target grid
  integer,intent(in) :: nrec    !- record number
  real(4),intent(out):: r

  integer,parameter   :: lun = 61
  integer             :: nrec_file, irec

  !-- check --
  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__read_hst'
    write(*,*) '  io is not registered.'
    stop
  endif
  if ( nrec > io%nm )  then
    write(*,*) 'Error at libmxe_io__read_hst'
    write(*,*) '  nrec=',nrec,' > io%nm=',io%nm
    stop
  endif
  if ( len(file_base) == 0 )  then
    write(*,*) 'Error at libmxe_io__read_hst: empty file_base'
    stop
  endif

  call libmxe_io__open( io, file_base, nrec, 4, lun, action='read' )

  nrec_file = libmxe_io__nrec_in_file( io, nrec ) + io%input_header_rec

  irec = i + (j-1)*para%imut + (k-1)*para%imut*para%jmut &
       & + (nrec_file-1)*para%imut*para%jmut*km
  read(lun,rec=irec ) r

  close( lun )

end subroutine libmxe_io__read_hst_1grid
!-----------------------------------------------------------------


subroutine libmxe_io__write_hst( para, io, file_base, nrec, klayer, r )
  implicit none

  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_io),intent(in)   :: io
  character(*),intent(in)           :: file_base
  integer,intent(in) :: nrec   !- record number
  integer,intent(in) :: klayer !- number of record layer
  real(4),intent(in) :: r( para%input_region%ifirst:para%input_region%ilast, &
                      &    para%input_region%jfirst:para%input_region%jlast, &
                      &    1:klayer )

  integer,parameter   :: lun = 61
  integer             :: k, reclen, rec0
  integer             :: ifirst, ilast, jfirst, jlast
  real(4),allocatable :: rout(:,:)

  character(clen)     :: file_out
  logical             :: lexist

  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__write_rst: io is not registered.'
    stop
  endif
  if ( nrec > io%nm )  then
    write(*,*) 'Error at libmxe_io__write_rst: nrec=',nrec,' > io%nm=',io%nm
    stop
  endif
  if ( len(file_base) == 0 )  then
    write(*,*) 'Error at libmxe_io__write_hst: empty file_base'
    stop
  endif

  reclen = para%imut * para%jmut * 4
  rec0 = ( libmxe_io__nrec_in_file(io,nrec) - 1 ) * klayer

  if ( para%l_anl_region ) then

    ifirst = para%anl_region%ifirst
    ilast  = para%anl_region%ilast
    jfirst = para%anl_region%jfirst
    jlast  = para%anl_region%jlast   !- anl_region not input_region

    file_out = libmxe_io__suffix( io, file_base, nrec )
    inquire( file=trim(file_out), exist=lexist )
    open( lun, file=trim(file_out), recl=reclen, &
         & access='direct', form='unformatted', action='readwrite' )

    allocate( rout(para%imut,para%jmut) )
    rout(:,:) = para%rundefout
    do k = 1, klayer
      if ( lexist ) read( lun, rec=k+rec0 ) rout
      rout(ifirst:ilast,jfirst:jlast) = r(ifirst:ilast,jfirst:jlast,k)
      write( lun, rec=k+rec0 ) rout(:,:)
    enddo
    deallocate( rout )

  else

    call libmxe_io__open( io, file_base, nrec, reclen, lun, action='write' )
    do k = 1, klayer
      write( lun, rec=k+rec0 ) r(:,:,k)
    enddo
    !- layered output to save momery
  endif

  close( lun )

end subroutine libmxe_io__write_hst
!-----------------------------------------------------------------


subroutine libmxe_io__write_rst( para, io, file_base, nrec, klayer, r )
  implicit none

  type(type_libmxe_para),intent(in) :: para
  type(type_libmxe_io),intent(in)   :: io
  character(*),intent(in)           :: file_base
  integer,intent(in) :: nrec   !- record number
  integer,intent(in) :: klayer !- number of record layer
  real(8),intent(in) :: r( para%input_region%ifirst:para%input_region%ilast, &
                      &    para%input_region%jfirst:para%input_region%jlast, &
                      &    1:klayer )

  integer,parameter   :: lun = 61
  integer             :: k, reclen, rec0
  integer             :: ifirst, ilast, jfirst, jlast
  real(8),allocatable :: rout(:,:)

  character(clen)     :: file_out
  logical             :: lexist

  if ( .not. io%ldef )  then
    write(*,*) 'Error at libmxe_io__write_rst: io is not registered.'
    stop
  endif
  if ( nrec > io%nm )  then
    write(*,*) 'Error at libmxe_io__write_rst: nrec=',nrec,' > io%nm=',io%nm
    stop
  endif
  if ( len(file_base) == 0 )  then
    write(*,*) 'Error at libmxe_io__write_hst: empty file_base'
    stop
  endif

  reclen = para%imut * para%jmut * 8
  rec0 = ( libmxe_io__nrec_in_file(io,nrec) - 1 ) * klayer

  if ( para%l_anl_region ) then

    ifirst = para%anl_region%ifirst
    ilast  = para%anl_region%ilast
    jfirst = para%anl_region%jfirst
    jlast  = para%anl_region%jlast   !- anl_region not input_region

    file_out = libmxe_io__suffix( io, file_base, nrec )
    inquire( file=trim(file_out), exist=lexist )
    open( lun, file=trim(file_out), recl=reclen, &
         & access='direct', form='unformatted', action='readwrite' )

    allocate( rout(para%imut,para%jmut) )
    rout(:,:) = para%rundefout
    do k = 1, klayer
      if ( lexist ) read( lun, rec=k+rec0 ) rout
      rout(ifirst:ilast,jfirst:jlast) = r(ifirst:ilast,jfirst:jlast,k)
      write( lun, rec=k+rec0 ) rout(:,:)
    enddo
    deallocate( rout )

  else

    call libmxe_io__open( io, file_base, nrec, reclen, lun, action='write' )
    do k = 1, klayer
      write( lun, rec=k+rec0 ) r(:,:,k)
    enddo
    !- layered output to save momery
  endif

  close( lun )

end subroutine libmxe_io__write_rst





end module libmxe_io
