.PHONY: all, clean
all: aes.elf

aes_func.o: aes_func.c aes_func.h


OPT_LEVEL := -O2

EXTRA_OBJ := aes_func.o

# if you want to use the masking technique, uncomment the following line
EXTRA_CFLAGS := -DMASKING

clean:
	-$(RM) *.o *.elf *.disasm
include  ${SDK_DIR}/etc/Makefile.common

