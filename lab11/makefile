BLOCKS = $(shell seq 1 12)
GRIDS = $(shell seq 1 12)
NS = $(shell seq 1000000 1000000 10000000)

build:
	nvcc -I cuda-samples/Common/ vectors.cu -o vectors

test:
	@for n in $(NS); do \
		for grid in $(GRIDS); do \
			for grid in $(GRIDS); do \
				./vectors $$n $$grid $$block; \
			done ; \	
		done ; \
		printf "\n" ; \
	done
	@printf "\n"

clean:
	rm -rf *.x

	