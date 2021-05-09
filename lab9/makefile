N = $(shell seq 200000 200000 800000) 
BUCKETS = $(shell seq 1 4)
THREADS = $(shell seq 1 16)
TIMES = $(shell seq 1 1)

all: compile test1

compile: macos

macos:
	clang++ -Xpreprocessor -fopenmp sort.cpp -o sort -lomp

unix:
	g++ -fopenmp sort.cpp -o sort

test1seq: compile
	@echo "size\tbuckets\tprepare\trand\tsplit\tsort\tconcat"
	@for n in $(N) ; do \
		for buckets in $(BUCKETS) ; do \
			for time in $(TIMES); do \
				./sort 1 $$n 1 $$buckets ; \
			done ; \
		done ; \
	done

test1speedup: compile
	@echo "size\tthreads\tprepare\trand\tsplit\tsort\tconcat"
	@for thread in $(THREADS) ; do \
		for time in $(TIMES); do \
			./sort 1 $$n $$thread $$thread ; \
		done ; \
	done
