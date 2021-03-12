remote: build standard buffered

local: build standard_local buffered_local

build:
	mpicc -o standard.x standard.c
	mpicc -o buffered.x buffered.c

standard:
	mpiexec -machinefile ./allnodes -np 2 ./standard.x

buffered:
	mpiexec -machinefile ./allnodes -np 2 ./buffered.x

standard_local:
	mpiexec -np 2 ./standard.x

buffered_local:
	mpiexec -np 2 ./buffered.x

clean:
	rm -rf *.x

