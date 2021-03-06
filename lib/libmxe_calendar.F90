! -*-F90-*-
module libmxe_calendar
 implicit none
 private


  type,public :: type_calendar
    integer :: year, month, day, hour, minute, second 
      !- year >= 0
  end type type_calendar


  public :: libmxe_calendar__set_config

  public :: libmxe_calendar__addcal
    !- get a calendar date elapsed by CalAdd since Cal 
    !-  Usage: CalNew = libmxe_calendar__addcal(Cal,CalAdd) 

  public :: libmxe_calendar__addsec
    !- get a calendar date elapsed by sec [second] since Cal 
    !-  Usage: CalNew = libmxe_calendar__addsec(Cal,sec) 

  public :: libmxe_calendar__addmonth
    !- get a calendar date elapsed by m [month] since Cal 
    !-  Usage: CalNew = libmxe_calendar__addmonth(Cal,m) 

  public :: libmxe_calendar__diffsec
    !- get time interval [second]
    !-  Usage: second = libmxe_calendar__diffsec(cal1,cal2) 

  private :: libmxe_calendar__cal2sec
    !- get total second from 1jan0001 0:00
    !-  Usage: second = libmxe_calendar__cal2sec(cal) 
    !-  Notice!!: Function is integer(8).

  private :: libmxe_calendar__sec2cal
    !- convert second from 1jan0001 0:00 to calendar
    !-  Usage: cal = libmxe_calendar__sec2cal(sec) 
    !-  Notice!!: Argument SEC is integer(8).

  public :: libmxe_calendar__display_num
  public :: libmxe_calendar__cdate
    !- get character(clen=15) like 10:30Z12JAN2000
    !-  Usage: cdate = libmxe_calendar__cdate(cal) 

  public :: libmxe_calendar__cdate_simple  !- get string like YYYYMMDD

  public :: libmxe_calendar__intarr2cal
    !- integer :: intarr(6)=(/year, month, day, hour, minute, second/)
    !- cal = libmxe_calendar__intarr2cal( intarr(6) )

  public :: libmxe_calendar__l_leap_year
    !- Usage: l_leap_year = libmxe_calendar__l_leap_year( year )
    !-  .true. : leap year

  public :: libmxe_calendar__ileap_year
  public :: libmxe_calendar__day_of_year

  public :: libmxe_calendar__time_digit

  public :: libmxe_calendar__seek
  public :: libmxe_calendar__verify_date
  public :: libmxe_calendar__is_date_wrong


  integer :: month(12) =(/31,28,31,30,31,30,31,31,30,31,30,31/) 

  logical,save :: l_leap_default = .false.

contains 
!---------------------------------------------------------------------


subroutine libmxe_calendar__set_config( l_leap )
  implicit none

  logical,intent(in) :: l_leap

  l_leap_default = l_leap

end subroutine libmxe_calendar__set_config
!---------------------------------------------------------------------


function libmxe_calendar__addcal( cal, caladd, l_leap )
  implicit none

  type(type_calendar)            :: libmxe_calendar__addcal
  type(type_calendar),intent(in) :: cal, caladd
  logical,optional,intent(in)    :: l_leap

  type(type_calendar) :: caltemp
  integer :: i
  logical :: lleap

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap

  !-- + day, hour, minute, second
  i = caladd%day * 24 * 3600 + caladd%hour * 3600 &
              & + caladd%minute * 60  + caladd%second
  caltemp = libmxe_calendar__addsec( cal, i, l_leap=lleap )

  !-- + month
  caltemp = libmxe_calendar__addmonth( caltemp, caladd%month )

  !-- + year
  caltemp%year = caltemp%year + caladd%year

  libmxe_calendar__addcal = caltemp
    

end function libmxe_calendar__addcal
!---------------------------------------------------------------------


function libmxe_calendar__addsec( cal, sec, l_leap )
  implicit none

  type(type_calendar)            :: libmxe_calendar__addsec
  type(type_calendar),intent(in) :: cal
  integer,intent(in)             :: sec
  logical,optional,intent(in)    :: l_leap

  integer(8) :: i
  logical :: lleap

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap

  i = libmxe_calendar__cal2sec( cal, lleap )
  i = i + sec
  libmxe_calendar__addsec = libmxe_calendar__sec2cal( i, lleap )

end function libmxe_calendar__addsec
!---------------------------------------------------------------------


function libmxe_calendar__addmonth(cal, m)
  implicit none

  type(type_calendar) :: libmxe_calendar__addmonth

  type(type_calendar),intent(in)  :: cal
  integer,intent(in) :: m
  type(type_calendar) :: calnew
  integer :: year_change, new_month

  calnew = cal
  new_month = cal%month + m

  year_change = floor( ( dble( new_month - 1 ) + 1.d-10 ) / 12.d0 )
  calnew%year = cal%year + year_change
  calnew%month = new_month - year_change * 12

  libmxe_calendar__addmonth = calnew

!- This function fails when
!-   FFLAGS = -fastsse, and
!-     year_change = ( new_month - 1 ) / 12

end function libmxe_calendar__addmonth
!---------------------------------------------------------------------


function libmxe_calendar__cal2sec( cal, l_leap )
  implicit none

  integer(8)                     :: libmxe_calendar__cal2sec
  type(type_calendar),intent(in) :: cal
  logical,            intent(in) :: l_leap

  integer :: n, nsum(0:11)
  integer(8) :: nday

  nsum(0) = 0
  do n = 1, 11
    nsum(n) = nsum(n-1) + month(n)
  enddo

  call libmxe_calendar__verify_date( cal, l_leap )

  !-- day number (except for today) since 1JAN0001 --
  nday = ( cal%year -1 )* 365 + nsum(cal%month-1) + cal%day - 1


  !-- additional days of leap years --
  if (l_leap) then

    nday = nday + ( cal%year - 1 ) / 4  - ( cal%year - 1 ) / 100  &
          &  + ( cal%year - 1 ) / 400 
    !- Count leap year from 0001 to cal%year-1

    if ( ( mod( cal%year , 4 ) == 0 ) &
      & .and. ( ( mod( cal%year , 100 ) /= 0 ) &
      &           .or. ( mod( cal%year , 400 ) == 0 ) ) &
      & .and. ( cal%month >= 3 ) ) then
      nday = nday + 1
    endif
    !- Add 1 day in MAR - DEC of leap yaer

  endif


  libmxe_calendar__cal2sec = nday * 24 * 3600 + cal%hour * 3600 &
                           & + cal%minute * 60  + cal%second

end function libmxe_calendar__cal2sec
!---------------------------------------------------------------------


function libmxe_calendar__sec2cal( sec, l_leap )
  implicit none

  !- time[sec] start from 0:00 1JAN0001 to calendar[ymdhms]

  type(type_calendar)   :: libmxe_calendar__sec2cal
  integer(8),intent(in) :: sec
  logical,intent(in)    :: l_leap

  integer :: n, nsum(0:12), n400y, nday, nsec
  type(type_calendar) :: cal
  integer :: nsum_yr(0:400)

  nsum(0) = 0
  do n = 1, 12
    nsum(n) = nsum(n-1) + month(n)
  enddo

  !---- year ----

  nday = int( sec / ( 24 * 3600 ) )

  if (l_leap) then

    n400y = nday / ( 400 * 365 + 97)   !- cycle of leap year: 400yr
    nday = mod( nday, ( 400 * 365 + 97 ) ) + 1

    nsum_yr(0) = 0
    do n = 1, 400
      nsum_yr(n) = nsum_yr(n-1) + 365 + libmxe_calendar__ileap_year( n )
    enddo 

    do n = 1, 400
      if ( nday <= nsum_yr(n) ) then
        cal%year = n
        nday = nday - nsum_yr( n-1 )  !- ordinal day of year
        exit
      endif
    enddo
    cal%year = cal%year + n400y * 400

    if ( libmxe_calendar__l_leap_year( cal%year ) ) then
      nsum(2:12) = nsum(2:12) + 1
    endif

  else

    cal%year = nday / 365 + 1     !- start: B.C. 0001
    nday = mod( nday, 365 ) + 1   !- ordinal day of year
    
  endif


  !---- month, day ----
  do n = 1, 12
    if (  nday <= nsum(n) ) then
      cal%month = n
      exit
    endif
  enddo
  cal%day = nday - nsum(cal%month-1)


  !---- hour, minute, second ----
  nsec = int( mod( sec, int( 24*3600, 8 ) ) )
  cal%hour = nsec / 3600
  cal%minute = mod( nsec, 3600) / 60
  cal%second = mod( nsec, 60 )


  !---- check ----
  if ( ( cal%month < 1 ).or.(cal%month > 12) ) then
    write(*,*) 'Error at libmxe_calendar__sec2cal, cal%month:',cal%month
    stop
  endif

  libmxe_calendar__sec2cal = cal


end function libmxe_calendar__sec2cal
!---------------------------------------------------------------------


function libmxe_calendar__diffsec( cal1, cal2, l_leap )
  implicit none

  integer :: libmxe_calendar__diffsec
  type(type_calendar),intent(in)  :: cal1, cal2 
  logical,optional,intent(in)   :: l_leap

  integer(8) :: i1, i2
  logical :: lleap

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap

  i1 = libmxe_calendar__cal2sec( cal1, lleap )
  i2 = libmxe_calendar__cal2sec( cal2, lleap )
  libmxe_calendar__diffsec = int( i2 - i1 )

end function libmxe_calendar__diffsec
!---------------------------------------------------------------------


subroutine libmxe_calendar__display_num( cal )
  implicit none

  type(type_calendar),intent(in)  :: cal
  character(100) :: cout
  character(20)  :: ctemp

  write(ctemp,'(i10)') cal%year
  cout=trim(adjustl(ctemp))
  write(ctemp,'(i10)') cal%month
  cout=trim(cout)//'/'//trim(adjustl(ctemp))
  write(ctemp,'(i10)') cal%day
  cout=trim(cout)//'/'//trim(adjustl(ctemp))
  write(ctemp,'(i10)') cal%hour
  cout=trim(cout)//'/'//trim(adjustl(ctemp))
  write(ctemp,'(i10)') cal%minute
  cout=trim(cout)//'/'//trim(adjustl(ctemp))
  write(ctemp,'(i10)') cal%second
  cout=trim(cout)//'/'//trim(adjustl(ctemp))

  ctemp='date'
  write(6,'(2x,a2,a20,a2,1x,a)') '* ',ctemp,' =',trim(cout)

end subroutine libmxe_calendar__display_num
!---------------------------------------------------------------------


function libmxe_calendar__cdate(cal)
  implicit none

  character(15) :: libmxe_calendar__cdate
  type(type_calendar),intent(in)  :: cal

  integer,parameter :: clen=15
  character(3) :: cmon(12) =(/'JAN','FEB','MAR','APR','MAY' &
                    & ,'JUN','JUL','AUG','SEP','OCT','NOV','DEC'/) 
  character(15) :: ctemp


  write(ctemp,'(i2,a1,i2.2,a1,i2.2,a3,i4.4)') &
   &  cal%hour, ':' ,cal%minute, 'Z', cal%day &
   & ,cmon(cal%month),cal%year

  libmxe_calendar__cdate = ctemp
    
end function libmxe_calendar__cdate
!---------------------------------------------------------------------


function libmxe_calendar__cdate_simple( cal, timemode )
  implicit none

  character(14) :: libmxe_calendar__cdate_simple

  type(type_calendar),intent(in) :: cal
  character(*),intent(in)        :: timemode

  character(14) :: ctemp

  write(ctemp,'(i4.4,5(i2.2))') cal%year,cal%month,cal%day,cal%hour,cal%minute,cal%second

  libmxe_calendar__cdate_simple = ctemp(1:libmxe_calendar__time_digit(timemode))

end function libmxe_calendar__cdate_simple
!---------------------------------------------------------------------


function libmxe_calendar__intarr2cal( intarr )
  implicit none

  type(type_calendar) :: libmxe_calendar__intarr2cal
  integer,intent(in)  :: intarr(6)

  libmxe_calendar__intarr2cal%year   = intarr(1)
  libmxe_calendar__intarr2cal%month  = intarr(2)
  libmxe_calendar__intarr2cal%day    = intarr(3)
  libmxe_calendar__intarr2cal%hour   = intarr(4)
  libmxe_calendar__intarr2cal%minute = intarr(5)
  libmxe_calendar__intarr2cal%second = intarr(6)

end function libmxe_calendar__intarr2cal
!---------------------------------------------------------------------


function libmxe_calendar__l_leap_year( year )
  implicit none

  logical            :: libmxe_calendar__l_leap_year
  integer,intent(in) :: year

  logical :: l_leap

  l_leap = .false.
  if ( mod(year,4  )==0 ) l_leap = .true.
  if ( mod(year,100)==0 ) l_leap = .false.
  if ( mod(year,400)==0 ) l_leap = .true.

  libmxe_calendar__l_leap_year = l_leap

end function libmxe_calendar__l_leap_year


!---------------------------------------------------------------------
!J うるう年(1)か否(0)か
integer function libmxe_calendar__ileap_year( year )
  implicit none

  integer,intent(in) :: year

  if ( libmxe_calendar__l_leap_year(year) ) then
    libmxe_calendar__ileap_year = 1
  else
    libmxe_calendar__ileap_year = 0
  endif

end function libmxe_calendar__ileap_year


!---------------------------------------------------------------------
!J 年間通算日(通日)を計算する
integer function libmxe_calendar__day_of_year( date, l_leap )
  implicit none

  type(type_calendar),intent(in) :: date
  logical,optional,intent(in)    :: l_leap

  integer :: n
  logical :: lleap

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap

  n = date%day
  if ( date%month >= 2 ) n = n + sum( month(1:date%month-1) )
  if ( lleap .and.( date%month >= 3 ) ) &
   & n = n + libmxe_calendar__ileap_year( date%year )

  libmxe_calendar__day_of_year = n

end function libmxe_calendar__day_of_year
!---------------------------------------------------------------------


function libmxe_calendar__time_digit( timemode )
  implicit none

  integer                 :: libmxe_calendar__time_digit
  character(*),intent(in) :: timemode

  integer :: digit

  digit = 0

  select case ( timemode )
  case('second')
    digit = 14
  case('minute')
    digit = 12
  case('hour')
    digit = 10
  case('day')
    digit = 8
  case('month')
    digit = 6
  case('year')
    digit = 4
  end select

  libmxe_calendar__time_digit = digit

end function libmxe_calendar__time_digit
!---------------------------------------------------------------------


subroutine libmxe_calendar__seek( ncal, cal, cal_target, n_target, weight, l_leap )
  implicit none

  integer,intent(in)             :: ncal
  type(type_calendar),intent(in) :: cal(ncal), cal_target
  integer,intent(out)            :: n_target
  real(8),intent(out)            :: weight
  logical,optional,intent(in)    :: l_leap

  integer :: n, sec_b, sec_a, na, nb, nc
  logical :: lleap

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap

  weight = 0.d0
  if ( libmxe_calendar__diffsec( cal(1), cal_target ) < 0.d0 ) then
    n_target = 0
    return
  endif
  if ( libmxe_calendar__diffsec( cal(ncal), cal_target ) > 0.d0 ) then
    n_target = ncal
    return
  endif
  if ( ncal <= 1 ) then
    n_target = 1
    return
  endif

  !- Use dichotomy
  na = 1
  nb = ncal
  do n = 1, ncal-1
    nc = int( (na+nb) / 2 )
    sec_b = libmxe_calendar__diffsec( cal(nc), cal_target, lleap )
    sec_a = libmxe_calendar__diffsec( cal_target, cal(nc+1), lleap )
    if ( sec_b >= 0 .and. sec_a >= 0 ) then
      n_target = nc
      exit
    endif
    if ( sec_b < 0 ) then
      nb = nc
    else
      na = nc
    endif
  enddo

  weight = dble(sec_a) / dble( sec_b + sec_a )

end subroutine libmxe_calendar__seek
!---------------------------------------------------------------------


subroutine libmxe_calendar__verify_date( cal, l_leap )
  implicit none

  type(type_calendar),intent(in) :: cal
  logical,optional,intent(in)    :: l_leap

  logical :: lleap

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap

  if ( libmxe_calendar__is_date_wrong(cal,lleap) ) then
    write(6,*) 'ERROR: Wrong calendar date'
    call libmxe_calendar__display_num( cal )
    stop
  endif

end subroutine libmxe_calendar__verify_date
!---------------------------------------------------------------------

logical function libmxe_calendar__is_date_wrong( cal, l_leap )
  implicit none

  type(type_calendar),intent(in) :: cal
  logical,optional,intent(in)    :: l_leap

  logical :: lleap
  integer :: day_in_month

  day_in_month = month( cal%month )

  lleap = l_leap_default
  if ( present(l_leap) ) lleap = l_leap
  if ( lleap .and. ( cal%month == 2 ) ) &
    & day_in_month = day_in_month + libmxe_calendar__ileap_year( cal%year )

  if ( cal%year < 0 .or. cal%month < 1 .or. cal%month > 12 &
     & .or. cal%day < 0 .or. cal%day > day_in_month &
     & .or. cal%hour < 0 .or. cal%hour > 23 &
     & .or. cal%minute < 0 .or. cal%minute > 59 &
     & .or. cal%second < 0 .or. cal%second > 59 ) then
    libmxe_calendar__is_date_wrong = .true.
  else
    libmxe_calendar__is_date_wrong = .false.
  endif

end function libmxe_calendar__is_date_wrong


end module
