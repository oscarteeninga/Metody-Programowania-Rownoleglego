#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Tab size required!\n");
        return -1;
    }
    int n = atoi(argv[1]);

    struct timeval start, end;
    long *tab = (long*) malloc(n * sizeof(long));

    gettimeofday(&start, NULL);
    
    for (int i = 0; i < n; i++) {
        tab[i] = rand();
    }
    gettimeofday(&end, NULL);

    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
            end.tv_usec - start.tv_usec) / 1.e6;
    
    printf("%.6f\n", delta);

    return 0;
}