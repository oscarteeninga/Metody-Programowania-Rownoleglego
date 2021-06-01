SIZE = 128 256 384 512 640 768 896 1024
CPU_SIZE = 16 32 64 80 96 112 128 144 160 176 192 208 224 240 256
BLOCK_SIZE = 8 16 24 32 40 48 56 72 80 88 96 104 112 120 128
all: test0 test1 test2

test0: test01 test02

test1: test11 test12 test13

test2: test21 test22 test23

# Porównanie prędkości CPU i GPU
test01:
	g++ -std=c++11 cpu.cpp -o cpu
	@for N in $(CPU_SIZE); do \
		./cpu $$N ; \
	done
	@printf "\n"

test02:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_global.cu -o global
	@for N in $(CPU_SIZE); do \
		./global $$N 16 ; \
	done
	@printf "\n"

# Testy wpływu rozmiaru siatki na wydajność obliczeń dla różnych typów pamięci
test11:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_global.cu -o global
	@for N in $(SIZE); do \
		./global $$N 16 ; \
	done
	@printf "\n"

test12:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_shared.cu -o shared
	@for N in $(SIZE); do \
		./shared $$N 16 ; \
	done
	@printf "\n"

test13:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_texture_pad.cu -o texture
	@for N in $(SIZE); do \
		./texture $$N 16; \
	done
	@printf "\n"

# Testy wpływu wielkości bloku wątków na wydajność obliczeń
test21:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_global.cu -o global
	@for B in $(BLOCK_SIZE); do \
		./global 128 $$B; \
	done
	@printf "\n"

test22:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_shared.cu -o shared
	@for B in $(BLOCK_SIZE); do \
		./shared 128 $$B; \
	done
	@printf "\n"

test23:
	nvcc -I cuda-samples/Common/ lab3/FD_2D_texture_pad.cu -o texture
	@for B in $(BLOCK_SIZE); do \
		./texture 128 $$B; \
	done
	@printf "\n"