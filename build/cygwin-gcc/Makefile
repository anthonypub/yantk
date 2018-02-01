SRCROOT=../../src
TESTROOT=../../test
OBJROOT=./obj
IFLAGS=-I ../../src -I /home/anthonyaue/src/OpenBLAS/
CC=g++
CFLAGS=-std=c++11 
LFLAGS=

all: testnet


testnet: testnet.o
	$(CC) $(LFLAGS) -o testnet testnet.o 

testnet.o: $(SRCROOT)/testnet.cpp
	$(CC) $(LFLAGS) $(CFLAGS) $(IFLAGS) -g -c -o testnet.o $(SRCROOT)/testnet.cpp


