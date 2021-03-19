build:
	gcc -o mc_sek.x mc_sek.c
	mpicc -o mc_par.x mc_par.c

seq:
	./mc_sek.x $(n)

par:
	mpiexec -np $(p) ./mc_par.x $(n)

clean:
	rm -rf *.x

	