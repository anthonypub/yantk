SRCROOT=../../src
TESTROOT=../../test
OBJROOT=./obj
OPENBLASROOT=/home/anthonyaue/src/OpenBLAS/
IFLAGS=-I ../../src -I /home/anthonyaue/src/OpenBLAS/
CC=g++
CFLAGS=-std=c++11 
LFLAGS=-L $(OPENBLASROOT) -lopenblas

all: yantk

test: tester
	./tester

tester: tester.o
	$(CC) $(LFLAGS) -o tester tester.o $(OPENBLASROOT)/libopenblas.a -lpthread

tester.o: $(TESTROOT)/test.cpp $(SRCROOT)/matrix.h
	$(CC) $(LFLAGS) $(CFLAGS) $(IFLAGS) -g -c -o tester.o $(TESTROOT)/test.cpp

yantk: yantk.o
	$(CC) $(LFLAGS) -g -o yantk yantk.o $(OPENBLASROOT)/libopenblas.a -lpthread

yantk.o: $(SRCROOT)/yantk.cpp $(SRCROOT)/matrix.h
	$(CC) $(LFLAGS) $(CFLAGS) $(IFLAGS) -g -c $(SRCROOT)/yantk.cpp
