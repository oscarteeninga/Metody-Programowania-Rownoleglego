#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int inside(double a, double b) {
    if (a*a+b*b <= 1)
        return 1;
    else
        return 0;
}

double rand_double() {
    return (double) rand() / RAND_MAX;
}

unsigned long long montecarlo(unsigned long long n) {
    register unsigned long long i, in = 0;
    register double a, b;
    for (i = 0; i < n; i++) {
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
    
    unsigned long long n = strtoull(argv[1], NULL, 0);
    srand(time(NULL));
    printf("n = %llu, π = %'.10Lf\n", n, (long double) montecarlo(n)/n);
}