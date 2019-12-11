! -*-F90-*-
!- system dependent commands
module libmxe_system
  implicit none
  private

  public :: libmxe_system__ini_rewind_stdin
  public :: libmxe_system__rewind_stdin

  logical,save :: can_rewind_stdin = .false.

contains


subroutine libmxe_system__ini_rewind_stdin
  implicit none

  integer :: i

  rewind(5,iostat=i)

  if ( i == 0 ) then
    can_rewind_stdin = .true.
  else
    can_rewind_stdin = .false.
  endif

end subroutine libmxe_system__ini_rewind_stdin


subroutine libmxe_system__rewind_stdin
  implicit none

  if ( can_rewind_stdin ) rewind(5)

end subroutine libmxe_system__rewind_stdin


end module libmxe_system
