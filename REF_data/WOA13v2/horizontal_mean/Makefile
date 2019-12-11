### Target objects ###
targets := horizontal_mean


### Phony targets ###
all: $(targets)
clean:
	rm -f *.o core $(targets) *_driver log.* \
	 *.mod *.a *~ */*~ *.ctl *.d namelist.test

vpath %.F90 src

### Platform dependent settings ###
include ../../../setting/macros.make


### MXE library ###
LIB_DIR := ../../../lib
include ../../../lib/libmxe.make


### Local modules ###
#lib :=
lib := lib.a
objs_lib := netcdf_io.o io_interface.o
$(lib) : $(objs_lib)
	$(AR) $@ $^ 2>&1
netcdf_io.o: %.o: %.F90 $(LIBMXE)
	$(F90) -c $(FFLAGS) $(NETCDF_FFLAGS) $(INCLUDES) $(NETCDF_INCLUDES) -o $@ $<
io_interface.o: %.o: %.F90 netcdf_io.o $(LIBMXE)
	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<
#$(objs_lib): %.o: %.F90 $(LIBMXE)
#	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<


### Compile ###
%.o: %.F90 $(lib) $(LIBMXE)
	$(F90) $< -c $(FFLAGS) $(INCLUDES) -o $@

### Link ###
%: %.o
	$(F90) $^ $(lib) $(LDFLAGS) $(NETCDF_LDFLAGS) -o $@


### Dependencies ###
horizontal_mean: horizontal_mean.o structure_mxe.o param_mxe.o
horizontal_mean.o: horizontal_mean.F90 structure_mxe.o param_mxe.o


### Wrapper of ANL ###
include ../../../anl/wrapper_ANL/mkdep
