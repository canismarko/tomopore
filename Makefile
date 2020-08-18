# GCC for debugging
# CC=gcc -g -pg
# GCC for deployment
CC=gcc -O2
LINK=-lhdf5 -lm
INSTALLDIR=$(HOME)/bin/

.phony: all, tests, install

all: tomopore

tests: tomopore_tests.out

install: tomopore
	cp tomopore $(INSTALLDIR)

tomopore: src/tomopore.c
	$(CC) -o tomopore src/tomopore.c $(LINK)

tomopore_tests.out: tests/test_math.c
	$(CC) -o tomopore_tests.out tests/test_math.c $(LINK)
