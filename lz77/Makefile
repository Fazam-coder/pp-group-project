CC = gcc
CFLAGS ?= -O2 -Wall -Wextra
OMP_FLAG ?= -fopenmp

# macOS: Homebrew GCC даёт OpenMP без дополнительной настройки линкера
ifeq ($(shell uname -s),Darwin)
  ifneq ($(shell command -v gcc-16 2>/dev/null),)
    CC = gcc-16
  else ifneq ($(shell command -v gcc-15 2>/dev/null),)
    CC = gcc-15
  else ifneq ($(shell command -v gcc-14 2>/dev/null),)
    CC = gcc-14
  endif
endif

# исходник ищем рядом с Makefile (LZ77.c / lz77.c) или в src/
SRC := $(firstword $(wildcard LZ77.c lz77.c src/LZ77.c src/lz77.c))

all: lz77

# параллельная версия (OpenMP)
lz77: $(SRC)
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ $<

# последовательная версия: тот же код без OpenMP
lz77_seq: $(SRC)
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o $@ $<

clean:
	rm -f lz77 lz77_seq *.out *.lz77

.PHONY: all clean
