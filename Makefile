CC=gcc -g -pg
LINK=-lhdf5 -lpthread -lm

.phony: all, tests

all: tomopore.out

tests: tomopore_tests.out

tomopore.out: src/tomopore.c
	$(CC) -o tomopore.out src/tomopore.c $(LINK)

tomopore_tests.out: tests/test_math.c
	$(CC) -o tomopore_tests.out tests/test_math.c $(LINK)
