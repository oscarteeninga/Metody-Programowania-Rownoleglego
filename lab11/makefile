BLOCKS = $(shell seq 1 2 4 8 16 32 64 128 256 512 1024 2048)
GRIDS =  $(shell seq 1 2 4 8 16 32 64 128 256 512 1024)
CORES = $(shell seq 1 90 180 360 720 1440 2880)

build:
	nvcc -std=c++11 -I cuda-samples/Common/ vectors.cu -o vectors

test:
	@for core in $(CORES); do \
		for grid in $(GRIDS); do \
			for block in $(BLOCKS); do \
				./vectors $$($$n * $$grid * $$block) $$grid $$block; \
			done ; \	
		done ; \
		printf "\n" ; \
	done
	@printf "\n"

clean:
	rm -rf *.x

	