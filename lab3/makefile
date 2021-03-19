build:
	gcc -o mc_sek.x mc_sek.c
	mpicc -o mc_par.x mc_par.c

seq:
	./mc_sek.x $(n)

par:
	mpiexec -np $(p) ./mc_par.x $(4*$(*n))

silne:
	mpiexec -machinefile ./allnodes -np 12 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 11 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 10 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 9 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 8 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 7 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 6 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 5 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 4 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 3 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 2 ./mc_par.x $(n)
	mpiexec -machinefile ./allnodes -np 1 ./mc_par.x $(n)

slabe:
	mpiexec -machinefile ./allnodes -np 12 ./mc_par.x 1200000000
	mpiexec -machinefile ./allnodes -np 11 ./mc_par.x 1100000000
	mpiexec -machinefile ./allnodes -np 10 ./mc_par.x 1000000000
	mpiexec -machinefile ./allnodes -np 9 ./mc_par.x 900000000
	mpiexec -machinefile ./allnodes -np 8 ./mc_par.x 800000000
	mpiexec -machinefile ./allnodes -np 7 ./mc_par.x 700000000
	mpiexec -machinefile ./allnodes -np 6 ./mc_par.x 600000000
	mpiexec -machinefile ./allnodes -np 5 ./mc_par.x 500000000
	mpiexec -machinefile ./allnodes -np 4 ./mc_par.x 400000000
	mpiexec -machinefile ./allnodes -np 3 ./mc_par.x 300000000
	mpiexec -machinefile ./allnodes -np 2 ./mc_par.x 200000000
	mpiexec -machinefile ./allnodes -np 1 ./mc_par.x 100000000

clean:
	rm -rf *.x

	