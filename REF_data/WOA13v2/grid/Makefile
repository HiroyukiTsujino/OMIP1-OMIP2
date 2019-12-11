
targets := dz_ctl

all: $(targets)
clean:
	rm -f *.o core $(targets) *.mod *.a *~ */*~ *.ctl \
	 NAMELIST.TEST *_driver log.*

vpath %.F90 src

include ../../../setting/macros.make

LIB_DIR := ../../../lib
include ../../../lib/libmxe.make


%.o: %.F90 $(LIBMXE)
	$(F90) -c $(FFLAGS) $(INCLUDES) -o $@ $<

%: %.o
	$(F90) $^ $(LDFLAGS) -o $@
