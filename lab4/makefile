PC = $(shell seq 1 8)
ZEUS = $(shell seq 1 12)
TIMES = $(shell seq 1 4) 

build:
	mpicc -o montecarlo.x montecarlo.c

local:
	@echo Skalowanie silne [local n = $(n)]:
	@for time in $(TIMES); do \
		for p in $(PC); do \
			mpiexec -np $$p ./montecarlo.x $(n) ; \
		done ; \
		printf "\n" ; \
	done
	@printf "\n"

strong:
	@echo Skalowanie silne [n = $(n)]:
	@for time in $(TIMES); do \
		for p in $(ZEUS); do \
			mpiexec -machinefile ./allnodes -np $$p ./montecarlo.x $(n) ; \
		done ; \
	done
	@printf "\n"

weak:
	@echo Skalowanie słabe [n = $(n)]:
	@for time in $(TIMES); do \
		for p in $(ZEUS); do \
			mpiexec -machinefile ./allnodes -np $$p ./montecarlo.x $($$p * $(n)) ; \
		done ; \
	done
	@printf "\n"

clean:
	rm -rf *.x

	