SIZE = 128 256 384 512 640 768 896 1024
CPU_SIZE = 16 32 64 80 96 112 128 144 160 176 192 208 224 240 256

all: test1 test2

test1: test01 test02

test2: test11 test12 test13

test01:
	g++ -std=c++11 cpu.cpp -o cpu
	@for N in $(CPU_SIZE); do \
		./cpu $$N ; \
	done
	@printf "\n"

test02:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_global.cu -o global
	@for N in $(CPU_SIZE); do \
		./global $$N ; \
	done
	@printf "\n"

test11:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_global.cu -o global
	@for N in $(SIZE); do \
		./global $$N ; \
	done
	@printf "\n"

test12:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_shared.cu -o shared
	@for N in $(SIZE); do \
		./shared $$N ; \
	done
	@printf "\n"

test13:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_texture_pad.cu -o texture
	@for N in $(SIZE); do \
		./texture $$N ; \
	done
	@printf "\n"