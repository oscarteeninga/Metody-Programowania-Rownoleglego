#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define SENDER 1
#define RECEIVER 0
#define N 100000
#define MAX_SIZE 8096

MPIR_CVAR_CH3_NOLOCAL = 1;
