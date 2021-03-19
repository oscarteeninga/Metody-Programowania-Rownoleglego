build:
	gcc -o mc_sek.x mc_sek.c
	mpicc -o mc_par.x mc_par.c

seq:
	./mc_sek.x $(n)

par:
	mpiexec -machinefile ./allnodes -np $(p) ./mc_par.x $(n)

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
	
clean:
	rm -rf *.x

	