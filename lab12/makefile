SIZE = 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200

cpu: 
	g++ -std=c++11 cpu.cpp -o cpu.x

gpu:
	nvcc -std=c++11 -I cuda-samples/Common/ gpu.cu -o gpu.x

test_cpu: cpu
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

	