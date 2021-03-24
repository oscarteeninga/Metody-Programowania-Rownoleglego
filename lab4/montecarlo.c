#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

unsigned long long global_in;

int inside(double a, double b) {
    if (a*a+b*b <= 1)
        return 1;
    else
        return 0;
}

double rand_double() {
    return (double) rand() / RAND_MAX;
}

unsigned long long montecarlo(unsigned long begin, unsigned long long end) {
    register unsigned long long i, in = 0;
    register double a, b;
    for (i = begin; i < end; i++) {
        a = rand_double();
        b = rand_double();
        in += inside(a, b);
    }
    return in;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Require n numer of points\n");
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    
    unsigned long long n = strtoull(argv[1], NULL, 0);
    unsigned long long size = n/num_proc;
    srand(time(NULL));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double start = MPI_Wtime();
        unsigned long long in = montecarlo(0, size);
        MPI_Reduce(&in, &global_in, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("%lf\t", MPI_Wtime() - start);
    } else {
        unsigned long long in = montecarlo(size*rank, size*(rank+1));
        MPI_Reduce(&in, &global_in, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}