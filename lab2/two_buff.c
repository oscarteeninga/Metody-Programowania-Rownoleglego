#include "parameters.h"

double sender_buffered(int size) {
    char *buff = malloc(size);
    double start = MPI_Wtime();
    for (int i = 0; i < N; i++) {
        MPI_Bsend(buff, size, MPI_BYTE, RECEIVER, 0, MPI_COMM_WORLD);
        MPI_Recv(buff, size, MPI_BYTE, RECEIVER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return MPI_Wtime() - start;
}

void receiver_buffered(int size) {
    char *buff = malloc(size);
    for (int i = 0; i < N; i++) {
        MPI_Recv(buff, size, MPI_BYTE, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Bsend(buff, size, MPI_BYTE, SENDER, 0, MPI_COMM_WORLD);
    }
}

void test_buffered(int rank) {
    char buff[1];
    double start_latency = MPI_Wtime();

    if (rank == SENDER) {
        MPI_Bsend(buff, 1, MPI_BYTE, RECEIVER, 0, MPI_COMM_WORLD);
    } else if (rank == RECEIVER) {
        MPI_Recv(buff, 1, MPI_BYTE, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double latency = MPI_Wtime()*1000-start_latency*1000;
        printf("====Buffered====\n");
        printf("Latency: %f ms\n", latency);
    }

    for (int size = 1; size <= MAX_SIZE; size *= 2) {
        if (rank == SENDER) {
            double time = sender_buffered(size);
            printf("%.2f, %d\n", N*size/time/1000000*2, size);
        } else if (rank == RECEIVER) {
            receiver_buffered(size);
        }
    }
}


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_buffered(rank);

    MPI_Finalize();
    return 0;
}