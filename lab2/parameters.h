#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define SENDER 1
#define RECEIVER 0
#define N 1000
#define MAX_SIZE 10000000

//MPIR_CVAR_CH3_NOLOCAL = 1;