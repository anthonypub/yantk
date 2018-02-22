SRCROOT=../../src
IFROOT=../../if/cpp
TESTROOT=../../test
OBJROOT=./obj
OPENBLASROOT=/home/anthonyaue/src/OpenBLAS/
IFLAGS=-I ../../src -I ../../if/cpp -I /home/anthonyaue/src/OpenBLAS/ -I /usr/local/include
CC=g++
CFLAGS=-std=c++11
LFLAGS=-L $(OPENBLASROOT) -L /usr/local/lib -lopenblas -lprotobuf -lpthread 

all: yantk

test: tester
	./tester

testnet: testnet.o net.pb.o weights.pb.o
	$(CC)  -o testnet testnet.o net.pb.o weights.pb.o $(LFLAGS)

net.pb.o: $(IFROOT)/net.pb.cc
	$(CC)  $(CFLAGS) $(IFLAGS) -g -c -o net.pb.o $(IFROOT)/net.pb.cc

weights.pb.o: $(IFROOT)/weights.pb.cc
	$(CC) $(CFLAGS) $(IFLAGS) -g -c -o weights.pb.o $(IFROOT)/weights.pb.cc

testnet.o: $(SRCROOT)/testnet.cpp
	$(CC)  $(CFLAGS) $(IFLAGS) -g -c -o testnet.o $(SRCROOT)/testnet.cpp


tester: tester.o
	$(CC) $(LFLAGS) -o tester tester.o $(OPENBLASROOT)/libopenblas.a -lpthread

tester.o: $(TESTROOT)/test.cpp $(SRCROOT)/matrix.h
	$(CC) $(LFLAGS) $(CFLAGS) $(IFLAGS) -g -c -o tester.o $(TESTROOT)/test.cpp

yantk: yantk.o
	$(CC) $(LFLAGS) -g -o yantk yantk.o $(OPENBLASROOT)/libopenblas.a -lpthread

yantk.o: $(SRCROOT)/yantk.cpp $(SRCROOT)/matrix.h
	$(CC) $(LFLAGS) $(CFLAGS) $(IFLAGS) -g -c $(SRCROOT)/yantk.cpp
