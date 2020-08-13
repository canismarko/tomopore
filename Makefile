CC=gcc
LINK=-lhdf5 -lpthread -lm
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
