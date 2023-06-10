CPP=g++
CPPFLAGS=-O3 -fopenmp 

default: rt

all: rt

rt: rt.cpp Makefile
	$(CPP) $(CPPFLAGS) -o rt rt.cpp 
	./rt



rtrec: rtrec.cpp Makefile
	$(CPP) $(CPPFLAGS) -o rtrec rtrec.cpp 
	./rtrec



rtinses: rtinses.cpp Makefile
	$(CPP) $(CPPFLAGS) -o rtinses rtinses.cpp 
	./rtinses

clean:
	-rm rtinses rt rtrec