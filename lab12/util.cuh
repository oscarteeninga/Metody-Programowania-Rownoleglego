//
// Created by Roman Wiatr on 18/05/2021.
//

#ifndef CUDA_INTRODUCTION_UTIL_CUH
#define CUDA_INTRODUCTION_UTIL_CUH

void checkErrors(char *label) {
// we need to synchronise first to catch errors due to
// asynchroneous operations that would otherwise
// potentially go unnoticed
    cudaError_t err;
    err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
        char *e = (char *) cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        char *e = (char *) cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
    }
}

double get_time() {
    struct timeval tim;
    cudaThreadSynchronize();
    gettimeofday(&tim, NULL);
    return (double) tim.tv_sec + (tim.tv_usec / 1000000.0);
}

#endif //CUDA_INTRODUCTION_UTIL_CUH
