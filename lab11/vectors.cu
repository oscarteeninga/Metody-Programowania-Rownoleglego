#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <helper_timer.h>
#include <iostream>
#include <chrono>

using namespace std;


__global__ void add(int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}



int** cuda(int n, int gridSize, int blockSize){

    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    int * a = new int[n];
    int * b = new int[n];
    int * c = new int[n];
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **) &dev_a, n * sizeof(int));
    cudaMalloc((void **) &dev_b, n * sizeof(int));
    cudaMalloc((void **) &dev_c, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, n * sizeof(int), cudaMemcpyHostToDevice);
    add <<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, n);
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    cout << "gpu\t" << n << "\t" << time << "\t" << gridSize << "\t" << blockSize << "\t" << endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    int ** result = new int*[3];
    result[0] = a;
    result[1] = b;
    result[2] = c;
    return result;
}

int **cpu(int n){
    int * a = new int[n];
    int * b = new int[n];
    int * c = new int[n];

    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> time = end_time - start_time;

    printf("cpu\t%d\t%.3f\n", n, time.count());

    int ** result = new int*[3];
    result[0] = a;
    result[1] = b;
    result[2] = c;
    return result;
}

int checkResults(int** cuda, int** cpp, int n){

    for(int i = 0; i < 3; ++i){
        for (int j = 0; j < n; ++j){
            if(cpp[i][j] != cuda[i][j]){
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {

    if (argc < 4) {
        cout << "Proper format is: <grid_size> <block_size> <method(gpu:1, cpu:0, both:2)>" << endl;
        return 1;
    }

    int gridSize = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    int method = atoi(argv[3]);
    int n = gridSize*blockSize;

    if (method == "gpu") {
        int** cudaResult = cuda(n, gridSize, blockSize);
    } else if (method == "cpu") {
        int** cpuResult = cpu(n);
    } else {
        int** cudaResult = cuda(n, gridSize, blockSize);
        int** cpuResult = cpu(n);
        if(!checkResults(cudaResult, cpuResult, n)) {
           cout << "Results are NOT equal!" << endl;
        }
    }

    return 0;
}