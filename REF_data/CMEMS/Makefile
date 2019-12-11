### Target objects ###
targets := generate_1x1_monthly

### Phony targets ###
all: $(targets)
clean:
	rm -f *.o $(targets) *_driver *.mod *.a *~ */*~

vpath %.F90 src

### Platform dependent settings ###
include ../../setting/macros.make

### MXE library ###
LIB_DIR := ../../lib
include ../../lib/libmxe.make

### Local modules ###
lib := lib.a
objs_lib := netcdf_io.o io_interface.o
$(lib) : $(objs_lib)
	$(AR) $@ $^
netcdf_io.o: %.o: %.F90 $(LIBMXE)
	$(F90) -c $(FFLAGS) $(NETCDF_FFLAGS) $(INCLUDES) $(NETCDF_INCLUDES) -o $@ $<
io_interface.o: %.o: %.F90 netcdf_io.o $(LIBMXE)
	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<
#$(objs_lib): %.o: %.F90 $(LIBMXE)ge	
#	$(F90) -c $(FFLAGS) $(CPPFLAGS) $(INCLUDES) -o $@ $<

####################

### Compile ###
%.o: %.F90 $(LIBMXE) $(lib)
	$(F90) $< -c $(FFLAGS) $(INCLUDES) $(CPPFLAGS) -o $@

### Link ###
%: %.o $(lib)
	$(F90) $^ $(LDFLAGS) $(NETCDF_LDFLAGS) -o $@

### Dependencies ###
