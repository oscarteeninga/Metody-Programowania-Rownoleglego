N = $(shell seq 200000 200000 800000) 
BUCKETS = $(shell seq 1 4)
THREADS = $(shell seq 1 16)
TIMES = $(shell seq 1 1)

all: compile test1

macos:
	clang++ -Xpreprocessor -fopenmp sort.cpp -o sort -lomp
	clang++ -Xpreprocessor -fopenmp random.cpp -o random -lomp

unix:
	g++ -fopenmp sort.cpp -o sort

test2seq: macos
	@echo "size\tbuckets\tprepare\trand\tsplit\tsort\tconcat"
	@for n in $(N) ; do \
		for buckets in $(BUCKETS) ; do \
			for time in $(TIMES); do \
				./sort 2 $$n 1 $$buckets ; \
			done ; \
		done ; \
	done

test2speedup: macos
	@echo "size\tthreads\tprepare\trand\tsplit\tsort\tconcat"
	@for thread in $(THREADS) ; do \
		for time in $(TIMES); do \
			./sort 2 $$n $$thread $$thread ; \
		done ; \
	done

testrandomize: macos
	@echo "size\tbuckets\tmaxdiff\terror"
	@for thread in $(THREADS) ; do \
		for time in $(TIMES); do \
			./random $$n $$thread ; \
		done ; \
	done

