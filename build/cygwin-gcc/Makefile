SRCROOT=../../src
IFROOT=../../if
IFCPP=$(IFROOT)/cpp
TESTROOT=../../test
OBJROOT=./obj
IFLAGS=-I ../../src -I /home/anthonyaue/src/OpenBLAS/ -I $(IFROOT)
CC=g++
CFLAGS=-std=c++11 
LFLAGS=-L $(OPENBLASROOT) -lopenblas -lprotobuf -lpthread 
PROTOC=protoc

all: testnet


testnet: testnet.o net.pb.o weights.pb.o
	$(CC)  -o testnet testnet.o net.pb.o weights.pb.o $(LFLAGS)

testnet.o: $(SRCROOT)/testnet.cpp
	$(CC) $(CFLAGS) $(IFLAGS) -g -c -o testnet.o $(SRCROOT)/testnet.cpp

net.pb.o: $(IFCPP)/net.pb.cc
	$(CC)  $(CFLAGS) $(IFLAGS) -g -c -o net.pb.o $(IFCPP)/net.pb.cc

weights.pb.o: $(IFCPP)/weights.pb.cc
	$(CC) $(CFLAGS) $(IFLAGS) -g -c -o weights.pb.o $(IFCPP)/weights.pb.cc

$(IFCPP)/weights.pb.cc $(IFCPP)/weights.pb.h: $(IFROOT)/weights.proto
	$(PROTOC) --proto_path=$(IFROOT) --cpp_out=$(IFCPP) $(IFROOT)/weights.proto

$(IFCPP)/net.pb.cc $(IFCPP)/net.pb.h: $(IFROOT)/net.proto
	$(PROTOC) --proto_path=$(IFROOT) --cpp_out=$(IFCPP) $(IFROOT)/net.proto




