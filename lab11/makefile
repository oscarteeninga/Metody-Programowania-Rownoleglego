GRIDS = 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144
BLOCKS = 64 128 256 512 1024

build:
	nvcc -std=c++11 -I cuda-samples/Common/ vectors.cu -o vectors

test_gpu:
	@for block in $(BLOCKS); do \
		for grid in $(GRIDS); do \
			./vectors $$grid $$block 1 ; \
		done ; \
	done
	@printf "\n"

test_cpu:
	@for grid in $(GRIDS); do \
		./vectors $$grid 1024 0 ; \
	done
	@printf "\n"

test:
	@for block in $(BLOCKS); do \
		for grid in $(GRIDS); do \
			./vectors $$grid $$block 2 ; \
		done ; \
	done
	@printf "\n"

clean:
	rm -rf *.x

	