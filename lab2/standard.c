#include "parameters.h"

double sender(int size) {
    char *buff = malloc(size);
    int i;
    double start = MPI_Wtime();
    for (i = 0; i < N; i++) {
        MPI_Send(buff, size, MPI_BYTE, RECEIVER, 0, MPI_COMM_WORLD);
        MPI_Recv(buff, size, MPI_BYTE, RECEIVER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return MPI_Wtime() - start;
}

void receiver(int size) {
    char *buff = malloc(size);
    int i;
    for (i = 0; i < N; i++) {
        MPI_Recv(buff, size, MPI_BYTE, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buff, size, MPI_BYTE, SENDER, 0, MPI_COMM_WORLD);
    }
}

void test(int rank) {
    int size;
    for (size = 1; size <= MAX_SIZE; size *= 2) {
        if (rank == SENDER) {
            double time = sender(size);
            printf("%.5f\n", N*size/time/1000000*2);
        } else if (rank == RECEIVER) {
            receiver(size);
        }
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test(rank);

    MPI_Finalize();
    return 0;
}