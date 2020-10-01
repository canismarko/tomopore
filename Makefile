# GCC for debugging
# OPTIMIZATIONS=-g -pg
# GCC for deployment
OPTIMIZATIONS=-O2
CC=g++ $(OPTIMIZATIONS)
LINK=-lhdf5 -lm -lpthread
INSTALLDIR=$(HOME)/bin

.phony: all, tests, install

all: tomopore

tests: tomopore_tests.out

install: tomopore
	cp tomopore.out $(INSTALLDIR)/tomopore

tomopore: src/tomopore.c src/filters.c src/hdfhelpers.c src/config.cpp
	$(CC) -o tomopore.out src/tomopore.c src/filters.c src/hdfhelpers.c src/config.cpp $(LINK)

tomopore_tests.out: tests/test_math.c
	$(CC) -o tomopore_tests.out tests/test_math.c $(LINK)
