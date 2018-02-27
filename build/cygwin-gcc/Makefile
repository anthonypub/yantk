SRCROOT=../../src
IFROOT=../../if/cpp
TESTROOT=../../test
OBJROOT=./obj
IFLAGS=-I ../../src -I /home/anthonyaue/src/OpenBLAS/ -I $(IFROOT)
CC=g++
CFLAGS=-std=c++11 
LFLAGS=-L $(OPENBLASROOT) -lopenblas -lprotobuf -lpthread 

all: testnet


testnet: testnet.o net.pb.o weights.pb.o
	$(CC)  -o testnet testnet.o net.pb.o weights.pb.o $(LFLAGS)

testnet.o: $(SRCROOT)/testnet.cpp
	$(CC) $(CFLAGS) $(IFLAGS) -g -c -o testnet.o $(SRCROOT)/testnet.cpp

net.pb.o: $(IFROOT)/net.pb.cc
	$(CC)  $(CFLAGS) $(IFLAGS) -g -c -o net.pb.o $(IFROOT)/net.pb.cc

weights.pb.o: $(IFROOT)/weights.pb.cc
	$(CC) $(CFLAGS) $(IFLAGS) -g -c -o weights.pb.o $(IFROOT)/weights.pb.cc



