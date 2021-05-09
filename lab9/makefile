N = $(shell seq 100000 100000 900000) 
BUCKETS = $(shell seq 1 8)

all: compile test1

compile: macos

macos:
	clang++ -Xpreprocessor -fopenmp sort.cpp -o sort -lomp

unix:
	g++ -fopenmp sort.cpp -o sort

test1: compile
	@echo "size\tbuckets\tprepare\trand\tsplit\tsort\tconcat"
	@for n in $(N) ; do \
		for buckets in $(BUCKETS) ; do \
			./sort 1 $$n 1 $$buckets ; \
		done ; \
	done