CC=gcc -g
LINK=-lhdf5

.phony: all

all: tomopore

tomopore: src/tomopore.c
	$(CC) -o tomopore.out src/tomopore.c $(LINK)
