#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Array size and threads num are required!\n");
        return -1;
    }
    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);

    omp_set_num_threads(threads);

    long *a = (long*) malloc(n * sizeof(long));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    #pragma omp parallel default(none) shared(a, n)
    {
        int i;
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(i = 0; i < n; i++) {
            a[i] = rand_r(&myseed);
        }   
    }

    gettimeofday(&end, NULL);

    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
            end.tv_usec - start.tv_usec) / 1.e6;
    
    printf("%.6f\n", delta);

    return 0;
}