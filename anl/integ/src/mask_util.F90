! -*-F90-*-
!- Utility subroutines of "mask"
module mask_util
  implicit none
  private

  public  :: mask_util__read
  public  :: mask_util__write


contains


subroutine mask_util__read( imask, file_in )
  implicit none

  integer,intent(out)     :: imask(:,:)
  character(*),intent(in) :: file_in

  integer,parameter    :: lun = 10

  character(6) :: ctemp
  integer      :: im, jm, j
  integer,allocatable :: mask1d(:)

  im = size( imask, 1 )
  jm = size( imask, 2 )
  allocate( mask1d(im) )
  write(ctemp,'(i6)'),im
  open( lun, file=file_in, form='formatted', action='read', status='old' )
  do j = jm, 1, -1
    read(lun,'('//trim(adjustl(ctemp))//'i1)') mask1d
    imask(:,j) = mask1d(:)
  enddo
  close( lun )

  deallocate(mask1d)

end subroutine mask_util__read


subroutine mask_util__write( imask, file_out )
  implicit none

  integer,intent(in)      :: imask(:,:)
  character(*),intent(in) :: file_out

  integer,parameter    :: lun = 10

  character(6) :: ctemp
  integer      :: im, jm, j

  im = size( imask, 1 )
  jm = size( imask, 2 )
  write(ctemp,'(i6)'),im
  open( lun, file=file_out, form='formatted', action='write' )
  do j = jm, 1, -1
    write( lun, '('//trim(adjustl(ctemp))//'i1)' ) imask(:,j)
  enddo
  close( lun )

end subroutine mask_util__write


end module mask_util

