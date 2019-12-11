# Makefile to be included by each Makefile
#========
#- How to write Makefile: see anl/test_lib/Makefile
#- How to compile       : > make lib

LIBMXE := libmxe.a

vpath $(LIBMXE) $(LIB_DIR)
INCLUDES += -I $(LIB_DIR)
LDFLAGS  += -L $(LIB_DIR) -lmxe

.PHONY: lib
lib:
	$(MAKE) --directory=$(LIB_DIR)

$(LIBMXE):
	$(MAKE) --directory=$(LIB_DIR)
