GRIDS = 1 2 4 8 16 32 64 128 256 512 1024 2048
BLOCKS =  1 2 4 8 16 32 64 128 256 512 1024

build:
	nvcc -std=c++11 -I cuda-samples/Common/ vectors.cu -o vectors

test:
	@for block in $(BLOCKS); do \
		for grid in $(GRIDS); do \
			./vectors $$grid $$block; \
		done ; \
		printf "\n" ; \
	done
	@printf "\n"

clean:
	rm -rf *.x

	