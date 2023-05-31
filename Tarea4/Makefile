CPP=g++
CPPFLAGS=-O3 -fopenmp 

default: rt

all: rt

rt: rt.cpp Makefile
	$(CPP) $(CPPFLAGS) -o rt rt.cpp 
	./rt

clean:
	-rm rt

rtas: rtas.cpp Makefile
	$(CPP) $(CPPFLAGS) -o rtas rtas.cpp 
	./rtas
