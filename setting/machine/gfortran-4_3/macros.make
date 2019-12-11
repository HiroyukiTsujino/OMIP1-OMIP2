# Compile setting for gfortran ver.4.3
#========

F90     = gfortran
LDFLAGS =
AR      = ar rv
INCLUDES=
FFLAGS  =  -pedantic -std=f95 -Wall -fbounds-check -O -Wuninitialized -ffpe-trap=invalid,zero,overflow -fbacktrace -fconvert=big-endian
