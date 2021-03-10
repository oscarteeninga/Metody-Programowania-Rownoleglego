#include "parameters.h"

double send_recv_buffered(int size) {
    char *buff = malloc(size);
    int i;
    double start = MPI_Wtime();
    for (i = 0; i < N; i++) {
        MPI_Bsend(buff, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(buff, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return MPI_Wtime() - start;
}

void test_buffered(int rank) {

    MPI_Buffer_attach(malloc(1), 1);
    
    char buff[1];
    printf("====Buffered====\n");
    double start_latency = MPI_Wtime();
    MPI_Bsend(buff, 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(buff, 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double end_latency = MPI_Wtime();
    printf("Latency: %f ms\n", (end_latency-start_latency)*1000);

    int size;
    for (size = 1; size <= MAX_SIZE; size *= 2) {
        MPI_Buffer_attach(malloc(size), size);
        double time = send_recv_buffered(size);
        printf("%.2f, %d\n", N*size/time/1000000*2, size);
    }
}



int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    test_buffered(rank);

    MPI_Finalize();
    return 0;
}