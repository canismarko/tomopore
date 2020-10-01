# GCC for debugging
# OPTIMIZATIONS=-g -pg
# GCC for deployment
OPTIMIZATIONS=-O2
# CC=g++ $(OPTIMIZATIONS)
CC=nvcc $(OPTIMIZATIONS) --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0
LINK=-lhdf5 -lm -lpthread -lcuda -lcudart
INSTALLDIR=$(HOME)/bin

.phony: all, tests, install

all: tomopore

tests: tomopore_tests.out

install: tomopore
	cp tomopore.out $(INSTALLDIR)/tomopore

tomopore: src/tomopore.cpp src/filters.cpp src/filters_cuda.cu src/hdfhelpers.cpp src/matrix.cu
	$(CC) -o tomopore.out src/tomopore.cpp src/filters.cpp src/filters_cuda.cu src/matrix.cu src/hdfhelpers.cpp $(LINK)

tomopore_tests.out: tests/test_math.c
	$(CC) -o tomopore_tests.out tests/test_math.c $(LINK)
