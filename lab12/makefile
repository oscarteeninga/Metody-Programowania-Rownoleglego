SIZE = 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144

build:
	nvcc -std=c++11 -I cuda-samples/Common/ vectors.cu -o vectors

cpu: 
	g++ -std=c++11 cpu.cpp -o cpu.x

gpu:
	nvcc -std=c++11 -I cuda-samples/Common/ gpu.cu -o gpu.x

test_cpu:
	@for N in $(SIZE); do \
		./cpu.x $$N ; \
	done
	@printf "\n"

test_gpu:
	@for N in $(SIZE); do \
		./gpu.x $$N ; \
	done
	@printf "\n"

test: test_cpu test_gpu

clean:
	rm -rf *.x temperature/*.txt

	