CPP = /usr/local/opt/llvm/bin/clang
CPPFLAGS = -I/usr/local/opt/llvm/include -fopenmp
LDFLAGS = -L/usr/local/opt/llvm/lib

N = 1000000000

all: omp.x normal.x

test: all
	./omp.x ${N} 8
	./normal.x ${N}

omp.x: omp.c
	$(CPP) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

normal.x: normal.c
	gcc $^ -o $@

clean:
	rm -rf *.x