# PGI Fortran at OGSV009
#========

# pgf90 17.4-0 64-bit target on x86-64 Linux -tp haswell at OGSV009

F90     = pgf90
FFLAGS  = -tp k8-64 -mcmodel=medium -O -byteswapio -Kieee
LDFLAGS = -mcmodel=medium -lblas -llapack -lfftw3
AR      = ar rv
INCLUDES=

#- For debug
#FFLAGS= -tp k8-64 -mcmodel=medium -O -byteswapio -Kieee -Mcache_align -Minfo -Mbounds
#- Do not use -fastsse, -O2

#- NetCDF Library in OGSV009
NETCDF_FFLAGS   = -DMXE_NETCDF
NETCDF_INCLUDES = `nf-config --fflags`
NETCDF_LDFLAGS  = `nf-config --flibs`
