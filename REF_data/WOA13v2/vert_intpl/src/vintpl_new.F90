! -*-F90-*-
!======================================================================
subroutine vintpl(depm,datm,mlyr,mav,depn,datn,nlyr,nav,rmiss)
!=====================================================================
! Information:
!     Square fitting toward the standard layer
!     Fitted data do not have extremum between original data points.
!----------------------------------------------------------------------
! DEPM, datm : model depth and model data
! depn, datn : new depth and data 
!
  integer(4),intent(in) :: mlyr, nlyr, mav
  integer(4),intent(out):: nav
  REAL(8),intent(in)    :: depm(mlyr),datm(mlyr)
  REAL(8),intent(out)   :: depn(nlyr),datn(nlyr)
  real(8),intent(in)    :: rmiss

  INTEGER(4),parameter :: limit = 50

  INTEGER(4) ::   lp
  REAL(8) ::     x0,x1,x2,y0,y1,y2
  REAL(8) ::     a1,b1,c1,a2,b2,c2
  REAL(8) ::     amean,bmean,cmean
!
!------------------------------------------------------------------------

  ierr = 0

  do k = 1, nlyr
    datn(k) = RMISS
    if (depn(k) < depm(1)) cycle
    if (depn(k) == depm(mav)) then
      datn(k) = datm(mav)
      nav = k
      cycle
    else if (depn(k) > depm(mav)) then
      cycle
    endif

    lp= 0
    a1= 0.0d0
    a2= 0.0d0
    b1= 0.0d0
    b2= 0.0d0
    c1= 0.0d0
    c2= 0.0d0

    do m = 1, mav

      if (depm(m) == depn(k)) then
        datn(k) = datm(m)
        nav = k
        ierr = 0
        exit
      else if ((depm(m) < depn(k)) .and. (nlyr > (k+limit)) &
           &    .and. ((m+1) <= mlyr) .and. (depn(k+limit) < depm(m+1))) then
        ierr = ierr + 1
        exit
      end if

      if ((ierr > 0).and.(depn(k-1) < depm(m)).and.(depm(m) < depn(k))) then
        ierr = 0
      end if

      if (depm(m) > depn(k)) then
        if (ierr > 0) exit
        if (mav >= 3) then
          if (m > 2) then
            x0=depm(m-2)
            x1=depm(m-1)
            x2=depm(m  )
            y0=datm(m-2)
            y1=datm(m-1)
            y2=datm(m  )
            call squarefit(x0,x1,x2,y0,y1,y2,A1,B1,C1)
            call hamidashi(x1,x2,y1,y2,A1,B1,C1)
            lp = lp + 1
            TMP = (a1*depn(k)+b1)*depn(k)+c1
          end if
          if (m < mav) then
            x0=depm(m-1)
            x1=depm(m  )
            x2=depm(m+1)
            y0=datm(m-1)
            y1=datm(m  )
            y2=datm(m+1)
            call squarefit(x0,x1,x2,y0,y1,y2,A2,B2,C2)
            call hamidashi(x0,x1,y0,y1,A2,B2,C2)
            lp = lp + 1
            TMP = (a2*depn(k)+b2)*depn(k)+c2
          endif
          amean = (a1+a2)/real(lp,8)
          bmean = (b1+b2)/real(lp,8)
          cmean = (c1+c2)/real(lp,8)
          datn(k) = (amean*depn(k)+bmean)*depn(k)+cmean
          nav = k
          exit
        else
          datn(k) = datm(m-1) + (datm(m)-datm(m-1)) &
               &    *(depn(k)-depm(m-1))/(depm(m)-depm(m-1)) 
          nav = k
          exit
        end if
      end if
    end do
  end do

  RETURN

END subroutine vintpl

!=========================================================================
subroutine squarefit(x0,x1,x2,y0,y1,y2,a,b,c)
!=========================================================================
  
  real(8) x0,x1,x2
  real(8) y0,y1,y2
  real(8) a,b,c
  real(8) W0,W1,W2
  
  W0 = y0/(x2-x0)/(x0-x1)
  W1 = y1/(x0-x1)/(x1-x2)
  W2 = y2/(x1-x2)/(x2-x0)

  A = -W0-W1-W2
  B = W0*(x1+x2)+W1*(x2+x0)+W2*(x0+x1)
  C = -W0*x1*x2-W1*x2*x0-W2*x0*x1

  RETURN
END subroutine squarefit

!=========================================================================
subroutine hamidashi(x1,x2,y1,y2,a,b,c)
!=========================================================================

  real(8) x1,x2
  real(8) y1,y2
  real(8) a,b,c
  real(8) e

  if (a.eq.0.0d0) goto 900
  e = -b/a/2.0d0
  if (((e-x1)*(e-x2)).GE.0.0d0) goto 900
  if ((e-x1).GE.(x2-e)) then
    a = (y1-y2)/(x1-x2)/(x1-x2)
    b = -2.0d0 * a * x2
    c = y2 + a * x2 * x2
  else
    a = (y2-y1)/(x2-x1)/(x2-x1)
    b = -2.0d0 * a * x1
    c = y1 + a * x1 * x1
  endif

900 continue

  RETURN
END subroutine hamidashi
