# GCC for debugging
CXX_FLAGS=-g -pg -D TP_DEBUG
# GCC for deployment
CXX_FLAGS=-O2 -D TP_RELEASE

LINK=-lhdf5 -lm -lpthread
CC=g++ $(CXX_FLAGS) $(LINK)
INSTALLDIR=$(HOME)/bin

.phony: all, tests, install

all: tomopore

tests: tomopore_tests.out

install: tomopore
	cp tomopore.out $(INSTALLDIR)/tomopore

tomopore: src/tomopore.cpp src/filters.cpp src/hdfhelpers.cpp src/config.cpp src/matrix.cpp
	$(CC) -o tomopore.out src/tomopore.cpp src/filters.cpp src/hdfhelpers.cpp src/config.cpp src/matrix.cpp

tomopore_tests.out: tests/test_math.c
	$(CC) -o tomopore_tests.out tests/test_math.c
