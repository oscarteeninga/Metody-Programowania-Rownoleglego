PC = $(shell seq 1 8)
ZEUS = $(shell seq 1 12)
TIMES = $(shell seq 1 1) 

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
			mpiexec -np $$p ./montecarlo.x $(n) 2>/dev/null ; \
		done ; \
		printf "\n" ; \
	done
	@printf "\n"

weak:
	@echo Skalowanie slabe [n = $(n)]:
	@for time in $(TIMES); do \
		for p in $(ZEUS); do \
			mpiexec -np $$p ./montecarlo.x $$(( $$p * $(n) )) 2>/dev/null ; \
		done ; \
		printf "\n" ; \
	done
	@printf "\n"

clean:
	rm -rf *.x

	