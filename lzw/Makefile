CC = gcc
CFLAGS ?= -O2 -Wall -Wextra
OMP_FLAG ?= -fopenmp

# On macOS, Homebrew GCC provides OpenMP without extra linker setup.
ifeq ($(shell uname -s),Darwin)
  ifneq ($(shell command -v gcc-16 2>/dev/null),)
    CC = gcc-16
  else ifneq ($(shell command -v gcc-15 2>/dev/null),)
    CC = gcc-15
  else ifneq ($(shell command -v gcc-14 2>/dev/null),)
    CC = gcc-14
  endif
endif

all: lzw

lzw: lzw.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ $<

lzw_seq: lzw.c
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o $@ $<

clean:
	rm -f lzw lzw_seq

.PHONY: all clean
