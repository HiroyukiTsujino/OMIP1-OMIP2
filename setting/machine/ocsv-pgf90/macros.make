### Fortran 90 compiler for OCSV(linux) pgi fortran

F90     = pgf90
FFLAGS  = -tp k8-64 -mcmodel=medium -lacml -lacml_mv -O -byteswapio -Kieee
LDFLAGS = -mcmodel=medium -lacml -lacml_mv
AR      = ar rv
INCLUDES=

#- For debug
#FFLAGS="-tp k8-64 -mcmodel=medium -lacml -lacml_mv -O -byteswapio -Mcache_align -Minfo -Mbounds"
#- Do not use -fastsse, -O2
