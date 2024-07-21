.PHONY: all, clean
all: aes.elf


OPT_LEVEL := -O2

EXTRA_OBJ := aes_func.o

# EXTRA_CFLAGS := -DMASKING

clean:
	-rm -f *.{o,elf,disasm}
include  ${SDK_DIR}/etc/Makefile.common

