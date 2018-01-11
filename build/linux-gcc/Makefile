SRCROOT=../../src
OBJROOT=./obj
OPENBLASROOT=/home/anthonyaue/src/OpenBLAS/
INCS=/home/anthonyaue/src/OpenBLAS/
CC=g++
CFLAGS=-std=c++11
LFLAGS=-L $(OPENBLASROOT) -lopenblas

all: yantk

yantk: yantk.o
	$(CC) $(LFLAGS) -o yantk yantk.o $(OPENBLASROOT)/libopenblas.a -lpthread

yantk.o: $(SRCROOT)/yantk.cpp
	$(CC) $(LFLAGS) $(CFLAGS) -I$(INCS) -g -c $(SRCROOT)/yantk.cpp
