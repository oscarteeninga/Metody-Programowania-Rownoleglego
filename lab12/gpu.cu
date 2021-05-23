/*** Calculating a derivative with CD ***/
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include "util.cuh"

// GPU kernel
__global__ void copy_array(float *u, float *u_prev, int N, int BSZ) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    int I = blockIdx.y * BSZ * N + blockIdx.x * BSZ + j * N + i;
    if (I >= N * N) { return; }
    u_prev[I] = u[I];
}

__global__ void update(float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ) {
    // Setting up indices
    int i = threadIdx.x;
    int j = threadIdx.y;
    int I = blockIdx.y * BSZ * N + blockIdx.x * BSZ + j * N + i;

    if (I >= N * N) { return; }

    // if not boundary do
    if ((I > N) && (I < N * N - 1 - N) && (I % N != 0) && (I % N != N - 1)) {
        float add = (alpha*dt)/(h*h)*(u_prev[I+N] + u_prev[I-N] + u_prev[I-1] + u_prev[I+1] - 4*u_prev[I])
        atomicAdd(&u[I], add);
    }

    // Boundary conditions are automatically imposed
    // as we don't touch boundaries
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Not enough arguments. Required: <N>.");
        return -1;
    }
    
    // Allocate in CPU
    int N = atoi(argv[1]);
    int BLOCKSIZE = 16;

    cudaSetDevice(0);

    float xmin = 0.0f;
    float xmax = 10.0f;
    float ymin = 0.0f;
    float h = (xmax - xmin) / (N - 1);
    float dt = 0.00001f;
    float alpha = 0.645f;
    float time = 0.4f;

    int steps = ceil(time / dt);
    int I;

    float *x = new float[N * N];
    float *y = new float[N * N];
    float *u = new float[N * N];
    float *u_prev = new float[N * N];


    // Generate mesh and intial condition
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            I = N * j + i;
            x[I] = xmin + h * i;
            y[I] = ymin + h * j;
            u[I] = 0.0f;
            if ((i == 0) || (j == 0)) { u[I] = 200.0f; }
        }
    }

    // Allocate in GPU
    float *u_d, *u_prev_d;

    cudaMalloc((void **) &u_d, N * N * sizeof(float));
    cudaMalloc((void **) &u_prev_d, N * N * sizeof(float));

    // Copy to GPU
    cudaMemcpy(u_d, u, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Loop
    dim3 dimGrid(int((N - 0.5) / BLOCKSIZE) + 1, int((N - 0.5) / BLOCKSIZE) + 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    double start = get_time();
    for (int t = 0; t < steps; t++) {
        copy_array <<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, BLOCKSIZE);
        update <<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);

    }
    double stop = get_time();
    checkErrors("update");

    double elapsed = stop - start;
    std::cout << "gpu" << "\t" << N << "\t" << elapsed << std::endl;

    // Copy result back to host
    cudaMemcpy(u, u_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream temperature("temperature/temperature_gpu.txt");
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            I = N * j + i;
            //	std::cout<<u[I]<<"\t";
            temperature << u[I] << "\t";
        }
        temperature << "\n";
        //std::cout<<std::endl;
    }

    temperature.close();

    // Free device
    cudaFree(u_d);
    cudaFree(u_prev_d);
}
