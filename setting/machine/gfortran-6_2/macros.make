# Compile setting for gfortran ver.6.2
#========

F90     = gfortran
LDFLAGS =
AR      = ar rv
INCLUDES=
FFLAGS  = -pedantic -Wconversion -Wunused -Wampersand -Wintrinsics-std -Wintrinsic-shadow -Waliasing -Wsurprising -Wc-binding-type -Wtabs -Wline-truncation -Wtarget-lifetime -Winteger-division -Wreal-q-constant -fbounds-check -O -ffpe-trap=invalid,zero,overflow -fbacktrace -fconvert=big-endian


# Unspecified options:
#  * -Wall              noisy ( due to uninitialized )
#  * -Wuninitialized    noisy
#  * -std=f2003         cannot use open( format= )
