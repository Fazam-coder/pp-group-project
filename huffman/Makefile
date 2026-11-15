CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
  ifneq ($(shell which gcc-16 2>/dev/null),)
    CC = gcc-16
  else ifneq ($(shell which gcc-15 2>/dev/null),)
    CC = gcc-15
  else ifneq ($(shell which gcc-14 2>/dev/null),)
    CC = gcc-14
  endif
endif

OMP_FLAG ?= -fopenmp

TARGETS = huffman
ifneq ($(wildcard LZ77.c),)
TARGETS += lz77
endif
ifneq ($(wildcard src/lz77.c),)
TARGETS += lz77
endif

all: $(TARGETS)

huffman: huffman.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ $<

lz77: $(firstword $(wildcard LZ77.c src/lz77.c))
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ $<

huffman_seq: huffman.c
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o $@ $<

lz77_seq: $(firstword $(wildcard LZ77.c src/lz77.c))
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o $@ $<

clean:
	rm -f huffman lz77 huffman_seq lz77_seq *.out *.huff *.lz77

.PHONY: all clean
