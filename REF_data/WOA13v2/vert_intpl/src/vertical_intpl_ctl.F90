! -*-F90-*-
program main

  use vertical_interpolation

  implicit none

  call ini
  do while ( has_next() )
    call read_and_write
    call next
  end do

end program main
