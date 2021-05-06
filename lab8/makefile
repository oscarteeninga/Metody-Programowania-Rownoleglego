all: macos unix

macos:
	clang++ -Xpreprocessor -fopenmp sort.cpp -o sort -lomp

unix:
	g++ -fopenmp sort.cpp -o sort