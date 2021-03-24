

build:
	gcc -o mc_sek.x mc_sek.c
	mpicc -o mc_par.x mc_par.c

seq:
	@./mc_sek.x $(n)

par:
	mpiexec -np $(p) ./mc_par.x $(n)

silne:
	mpiexec -machinefile ./allnodes -np 12 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 11 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 10 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 9 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 8 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 7 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 6 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 5 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 4 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 3 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 2 -ppn 1 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 1 -ppn 1 ./mc_par.x $(n)

slabe:
	mpiexec -machinefile ./allnodes -np 12 ./mc_par.x 12000000000
	mpiexec -machinefile ./allnodes -np 11 ./mc_par.x 11000000000
	mpiexec -machinefile ./allnodes -np 10 ./mc_par.x 10000000000
	mpiexec -machinefile ./allnodes -np 9 ./mc_par.x 9000000000
	mpiexec -machinefile ./allnodes -np 8 ./mc_par.x 8000000000
	mpiexec -machinefile ./allnodes -np 7 ./mc_par.x 7000000000
	mpiexec -machinefile ./allnodes -np 6 ./mc_par.x 6000000000
	mpiexec -machinefile ./allnodes -np 5 ./mc_par.x 5000000000
	mpiexec -machinefile ./allnodes -np 4 ./mc_par.x 4000000000
	mpiexec -machinefile ./allnodes -np 3 ./mc_par.x 3000000000
	mpiexec -machinefile ./allnodes -np 2 ./mc_par.x 2000000000
	mpiexec -machinefile ./allnodes -np 1 ./mc_par.x 1000000000

clean:
	rm -rf *.x

	