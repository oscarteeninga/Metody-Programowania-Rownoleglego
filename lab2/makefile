all: build local one_node two_nodes_same_host two_nodes_other_host

build:
	mpicc -o standard.x standard.c
	mpicc -o buffered.x buffered.c

one_node:
	mpiexec -machinefile ./onenode -np 2 ./standard.x
	mpiexec -machinefile ./onenode -np 2 ./buffered.x

two_nodes_same_host:
	mpiexec -machinefile ./twonodessamehost -np 2 ./standard.x
	mpiexec -machinefile ./twonodessamehost -np 2 ./buffered.x

two_nodes_other_host:
	mpiexec -machinefile ./twonodesotherhost -np 2 ./standard.x
	mpiexec -machinefile ./twonodesotherhost -np 2 ./buffered.x

local:
	mpiexec -np 2 ./standard.x
	mpiexec -np 2 ./buffered.x

clean:
	rm -rf *.x
	