BLOCKS = $(shell seq 1 2 4 8 16 32 64 128 256 512 1024 2048)
GRIDS =  $(shell seq 1 2 4 8 16 32 64 128 256 512 1024)
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

	