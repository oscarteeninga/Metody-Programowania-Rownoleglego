all: build two twobuff one onebuff

build:
	mpicc -o two.x two.c
	mpicc -o one.x one.c
	mpicc -o two_buff.x two_buff.c
	mpicc -o one_buff.x one_buff.c

two:
	mpiexec -machinefile ./allnodes -np 2 ./two.x

twobuff:
	mpiexec -machinefile ./allnodes -np 2 ./two_buff.x

one:
	mpiexec -machinefile ./allnodes -np 1 ./one.x

onebuff:
	mpiexec -machinefile ./allnodes -np 1 ./one_buff.x

clean:
	rm -rf *.x

