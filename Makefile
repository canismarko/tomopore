# GCC for debugging
# CC=gcc -g -pg
# GCC for deployment
CC=gcc -O2
LINK=-lhdf5 -lm -lpthread
INSTALLDIR=$(HOME)/bin

.phony: all, tests, install

all: tomopore

tests: tomopore_tests.out

install: tomopore
	cp tomopore.out $(INSTALLDIR)/tomopore

tomopore: src/tomopore.c src/filters.c src/hdfhelpers.c
	$(CC) -o tomopore.out src/tomopore.c src/filters.c src/hdfhelpers.c $(LINK)

tomopore_tests.out: tests/test_math.c
	$(CC) -o tomopore_tests.out tests/test_math.c $(LINK)
