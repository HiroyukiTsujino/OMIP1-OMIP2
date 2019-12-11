! -*-F90-*-
!- display IN/OUT info in common format.
module libmxe_display
  implicit none
  private

  public :: libmxe_display__nml_block
  public :: libmxe_display__title
  public :: libmxe_display__section
  public :: libmxe_display__var
  public :: libmxe_display__input
  public :: libmxe_display__output
  public :: libmxe_display__file
  public :: libmxe_display__to_lower

  interface libmxe_display__var
    module procedure  libmxe_display__var_i
    module procedure  libmxe_display__var_r
    module procedure  libmxe_display__var_d
    module procedure  libmxe_display__var_l
    module procedure  libmxe_display__var_c
    module procedure  libmxe_display__var_i8
    module procedure  libmxe_display__arr_i
    module procedure  libmxe_display__arr_d
    module procedure  libmxe_display__arr_c
    module procedure  libmxe_display__arr_i8
  end interface

  private :: display_var


contains
!---------------------------------------------------------------------


subroutine libmxe_display__nml_block( block_name, file_name )
  implicit none

  character(*),intent(in)          :: block_name
  character(*),intent(in),optional :: file_name

  character(255) :: cfile

  if ( present(file_name) ) then
    cfile=' ('//file_name//')'
  else
    cfile=''
  endif

  write(6,*)
  write(6,'(3a)') '#### Namelist ',block_name,trim(cfile)

end subroutine libmxe_display__nml_block
!---------------------------------------------------------------------


subroutine libmxe_display__title( title_name )
  implicit none

  character(*),intent(in)          :: title_name

  write(6,'(a)') trim(title_name)
  write(6,'(a)') '========'

end subroutine libmxe_display__title
!---------------------------------------------------------------------


subroutine libmxe_display__section( section_name )
  implicit none

  character(*),intent(in)          :: section_name

  write(6,*)
  write(6,'(a)') trim(section_name)
  write(6,'(a)') '--------'

end subroutine libmxe_display__section
!-----------------------------------------------------------------


subroutine libmxe_display__var_i( input, default, input_name )
  implicit none

  integer(4),intent(in)   :: input
  integer(4),intent(in)   :: default
  character(*),intent(in) :: input_name

  logical    :: lspecified
  character(10) :: cval

  if ( input == default ) then
    lspecified = .false.
  else
    lspecified = .true.
  endif

  write(cval,'(i10)') input
  call display_var( input_name, cval, lspecified )

end subroutine libmxe_display__var_i
!-----------------------------------------------------------------


subroutine libmxe_display__arr_i( input, default, input_name )
  implicit none

  integer(4),intent(in)   :: input(:)
  integer(4),intent(in)   :: default
  character(*),intent(in) :: input_name

  integer    :: ncount, i
  logical    :: lspecified
  character(10) :: cval
  character(255) :: cvals

  ncount = size(input)
  cvals  = ''
  lspecified = .false.

  do i = 1, ncount
    write(cval,'(i10)') input(i)
    if ( i >= 2 ) then
      cvals = trim(cvals)//','//adjustl(cval)
    else
      cvals = adjustl(cval)
    endif
    if ( input(i) /= default ) then
      lspecified = .true.
    endif
  enddo
  call display_var( input_name, cvals, lspecified )

end subroutine libmxe_display__arr_i
!-----------------------------------------------------------------


subroutine libmxe_display__var_r( input, default, input_name, l_show_full )
  implicit none

  real(4),intent(in)      :: input
  real(4),intent(in)      :: default
  character(*),intent(in) :: input_name
  logical,intent(in),optional :: l_show_full

  logical    :: lspecified
  character(10) :: cval
  character(14) :: cval14

  if ( input == default ) then
    lspecified = .false.
  else
    lspecified = .true.
  endif

  if ( present(l_show_full) ) then
    if ( l_show_full ) then
      write(cval14,'(es14.7)') input
      call display_var( input_name, cval14, lspecified )
      return
    endif
  endif

  write(cval,'(es10.3)') input
  call display_var( input_name, cval, lspecified )

end subroutine libmxe_display__var_r
!-----------------------------------------------------------------


subroutine libmxe_display__var_d( input, default, input_name, l_show_full )
  implicit none

  real(8),intent(in)      :: input
  real(8),intent(in)      :: default
  character(*),intent(in) :: input_name
  logical,intent(in),optional :: l_show_full

  logical    :: lspecified
  character(10) :: cval
  character(20) :: cval20

  if ( input == default ) then
    lspecified = .false.
  else
    lspecified = .true.
  endif

  if ( present(l_show_full) ) then
    if ( l_show_full ) then
      write(cval20,'(es20.13)') input
      call display_var( input_name, cval20, lspecified )
      return
    endif
  endif

  write(cval,'(es10.3)') input
  call display_var( input_name, cval, lspecified )

end subroutine libmxe_display__var_d
!-----------------------------------------------------------------


subroutine libmxe_display__arr_d( input, default, input_name )
  implicit none

  real(8),intent(in)      :: input(:)
  real(8),intent(in)      :: default
  character(*),intent(in) :: input_name

  integer        :: ncount, i
  logical        :: lspecified
  character(10)  :: cval
  character(255) :: cvals

  ncount = size(input)
  cvals  = ''
  lspecified = .false.

  do i = 1, ncount
    write(cval,'(es10.3)') input(i)
    if ( i >= 2 ) then
      cvals = trim(cvals)//','//adjustl(cval)
    else
      cvals = adjustl(cval)
    endif
    if ( input(i) /= default ) then
      lspecified = .true.
    endif
  enddo
  call display_var( input_name, cvals, lspecified )

end subroutine libmxe_display__arr_d
!-----------------------------------------------------------------


subroutine libmxe_display__var_l( input, default, input_name )
  implicit none

  logical,intent(in)      :: input
  logical,intent(in)      :: default    !- dummy
  character(*),intent(in) :: input_name

  logical       :: lspecified
  character(10) :: cval

  if ( input .eqv. default ) then
    lspecified = .false.
  else
    lspecified = .true.
  endif

  write(cval,'(l10)') input
  call display_var( input_name, cval, lspecified )

end subroutine libmxe_display__var_l
!-----------------------------------------------------------------


subroutine libmxe_display__var_c( input, default, input_name )
  implicit none

  character(*),intent(in) :: input
  character(*),intent(in) :: default
  character(*),intent(in) :: input_name

  logical    :: lspecified

  if ( trim(input) == trim(default) ) then
    lspecified = .false.
  else
    lspecified = .true.
  endif

  call display_var( input_name, input, lspecified )

end subroutine libmxe_display__var_c
!-----------------------------------------------------------------


subroutine libmxe_display__arr_c( input, default, input_name )
  implicit none

  character(*),intent(in) :: input(:)
  character(*),intent(in) :: default
  character(*),intent(in) :: input_name

  integer        :: ncount, i
  logical        :: lspecified
  character(3)   :: cn

  ncount = size(input)
  lspecified = .false.

  do i = 1, ncount
    write(cn,'(i3)') i
    if ( input(i) /= default ) then
      lspecified = .true.
    endif
    call display_var( input_name//' ('//trim(adjustl(cn))//')', trim(input(i)), lspecified )
  enddo

end subroutine libmxe_display__arr_c
!-----------------------------------------------------------------


subroutine libmxe_display__var_i8( input, default, input_name )
  implicit none

  integer(8),intent(in)   :: input
  integer(8),intent(in)   :: default
  character(*),intent(in) :: input_name

  logical    :: lspecified
  character(10) :: cval

  if ( input == default ) then
    lspecified = .false.
  else
    lspecified = .true.
  endif

  write(cval,'(i10)') input
  call display_var( input_name, cval, lspecified )

end subroutine libmxe_display__var_i8
!-----------------------------------------------------------------


subroutine libmxe_display__arr_i8( input, default, input_name )
  implicit none

  integer(8),intent(in)   :: input(:)
  integer(8),intent(in)   :: default
  character(*),intent(in) :: input_name

  integer    :: ncount, i
  logical    :: lspecified
  character(10) :: cval
  character(255) :: cvals

  ncount = size(input)
  cvals  = ''
  lspecified = .false.

  do i = 1, ncount
    write(cval,'(i10)') input(i)
    if ( i >= 2 ) then
      cvals = trim(cvals)//','//adjustl(cval)
    else
      cvals = adjustl(cval)
    endif
    if ( input(i) /= default ) then
      lspecified = .true.
    endif
  enddo
  call display_var( input_name, cvals, lspecified )

end subroutine libmxe_display__arr_i8
!-----------------------------------------------------------------


subroutine display_var( input_name, cval, lspecified )
  implicit none

  character(*),intent(in) :: input_name
  character(*),intent(in) :: cval
  logical,intent(in)      :: lspecified

  character(20) :: ctemp20
  character(10) :: ctemp10
  character(256) :: cprint

  if ( len(input_name) <= 20 ) then
    ctemp20 = input_name
    cprint = ctemp20//' ='
  else
    cprint = input_name//' ='
  endif

  if ( lspecified ) then
    ctemp20 = ''
  else
    ctemp20 = ' (default)'
  endif

  if ( len(cval) <= 10 ) then
    ctemp10 = adjustl(cval)
    cprint = trim(cprint)//' '//ctemp10//trim(ctemp20)
  else
    cprint = trim(cprint)//' '//trim(cval)//trim(ctemp20)
  endif

  write(6,'(2x,2a)') '* ',trim(cprint)

end subroutine display_var
!---------------------------------------------------------------------


subroutine libmxe_display__input
  implicit none

  write(6,*)
  write(6,'(a)') '#### Input file'

end subroutine libmxe_display__input
!---------------------------------------------------------------------


subroutine libmxe_display__output
  implicit none

  write(6,*)
  write(6,'(a)') '#### Output file'

end subroutine libmxe_display__output
!-----------------------------------------------------------------


subroutine libmxe_display__file( var_name, file_name, file_format )
  implicit none

  character(*),intent(in) :: var_name
  character(*),intent(in) :: file_name
  character(*),intent(in) :: file_format

  character(255)          :: cformat, cprint
  character(10)           :: ctemp10

  select case( file_format )
  case('int')
    cformat='integer direct access'
  case('real')
    cformat='real(4) direct access'
  case('dble')
    cformat='real(8) direct access'
  case('sequential')
    cformat='sequential access'
  case default
    cformat=file_format
  end select

  if ( len(file_name) <= 10 ) then
    ctemp10 = adjustl(file_name)
    cprint = ctemp10//' ('//trim(cformat)//')'
  else
    cprint = trim(file_name)//' ('//trim(cformat)//')'
  endif

  call display_var( var_name, cprint, .true. )

end subroutine libmxe_display__file
!-----------------------------------------------------------------


function libmxe_display__to_lower( string_in ) result( string_out )
  implicit none

  character(*),intent(in)   :: string_in
  character(len(string_in)) :: string_out

  integer(4) :: i, ic

  string_out = string_in
  do i = 1, len(string_in)
    ic = ichar(string_in(i:i))
    if (ic >= 65 .and. ic <= 90) string_out(i:i) = char(ic+32)
  enddo

end function libmxe_display__to_lower


end module libmxe_display
