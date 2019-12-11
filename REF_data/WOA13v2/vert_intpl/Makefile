#-*-Makefile-*-

targets := vertical_intpl_ctl

all: $(targets)
clean:
	rm -f *.o core $(targets) *.mod *.a *~ */*~ \
	  *_driver NAMELIST.TEST log.*

vpath %.F90 src

include ../../../setting/macros.make

LIB_DIR := ../../../lib
include ../../../lib/libmxe.make


### Local modules ###
lib := lib.a
objs_lib := netcdf_io.o io_interface.o vintpl_new.o
$(lib) : $(objs_lib)
	$(AR) $@ $^ 2>&1
netcdf_io.o: %.o: %.F90 $(LIBMXE)
	$(F90) -c $(FFLAGS) $(NETCDF_FFLAGS) $(INCLUDES) $(NETCDF_INCLUDES) -o $@ $<
io_interface.o: %.o: %.F90 netcdf_io.o $(LIBMXE)
	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<
vintpl_new.o: %.o: %.F90 $(LIBMXE)
	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<

#$(objs_lib): %.o: %.F90 $(LIBMXE)
#	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<

%.o: %.F90 $(lib) $(LIBMXE)
	$(F90) -c $(FFLAGS) $(NETCDF_FFLAGS) $(INCLUDES) $(NETCDF_INCLUDES) -o $@ $<

%: %.o
	$(F90) $^ $(lib) $(LDFLAGS) $(NETCDF_LDFLAGS) -o $@

vertical_intpl_ctl.o: vertical_intpl_ctl.F90 vertical_intpl.o
vertical_intpl_ctl: vertical_intpl_ctl.o vertical_intpl.o

#include test/test.make
