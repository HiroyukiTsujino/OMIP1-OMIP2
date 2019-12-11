### Fortran 90 compiler for OCSV(linux) pgi fortran

F90     = pgf90
FFLAGS  = -tp k8-64 -mcmodel=medium -lacml -O -byteswapio -Kieee
LDFLAGS = -mcmodel=medium -lacml
AR      = ar rv
INCLUDES=

#- For debug
#FFLAGS="-tp k8-64 -mcmodel=medium -lacml -lacml_mv -O -byteswapio -Mcache_align -Minfo -Mbounds"
#- Do not use -fastsse, -O2

#- NetCDF Library in OGSV007
NETCDF_DIR      = /usr/local/netcdf-fortran-4.4.3
NETCDF_FFLAGS   = -DMXE_NETCDF
NETCDF_INCLUDES = `$(NETCDF_DIR)/bin/nf-config --fflags`
NETCDF_LDFLAGS  = `$(NETCDF_DIR)/bin/nf-config --flibs`
