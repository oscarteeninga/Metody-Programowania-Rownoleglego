all: build twoshared oneshared

build:
	mpicc -o two_shared.x two_shared.c
	mpicc -o one_shared.x one_shared.c
	# mpicc -o two_network.x two_network.c
	# mpicc -o one_network.x one_network.c

twoshared:
	mpirun -np 2 two_shared.x

twonet:
	mpirun -np 2 two_network.x

oneshared:
	mpirun -np 1 one_shared.x

onenet:
	mpirun -np 1 one_network.x

clean:
	rm -rf *.x

