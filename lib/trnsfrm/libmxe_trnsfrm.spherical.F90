! -*-F90-*-
module libmxe_trnsfrm
  implicit none

  public :: mp2lp

contains

  subroutine set_abc ( nplat0, nplon0, splat0, splon0 )
    real(8),intent(in) :: nplat0, nplon0, splat0, splon0
  end subroutine set_abc

  subroutine mp2lp(lambda, phi, mu, psi )
    implicit none

    real(8), intent(out)    :: lambda, phi
    real(8), intent(in)     :: mu, psi

    lambda = mu
    phi    = psi

  end subroutine mp2lp

end module libmxe_trnsfrm
