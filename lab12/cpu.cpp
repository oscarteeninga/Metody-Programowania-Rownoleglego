#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include <chrono>
#include <iomanip>
#include <string> 

using namespace std;

void copy_array(int N, float *u, float *u_prev) {
    for (int i = 0; i < N*N; i++) {
        u_prev[i] = u[i];
    }
}

int main(int argc, char *argv[]) {
    
    if (argc < 2) {
        printf("Not enough arguments. Required: <N>.");
        return -1;
    }

    int N = atoi(argv[1]);

    float xmin = 0.0f;
    float xmax = 10.0f;
    float ymin = 0.0f;
    float h = (xmax - xmin) / (N - 1);
    float dt = 0.00001f;
    float alpha = 0.645f;
    float time = 0.4f;

    int steps = ceil(time / dt);
    int I;

    int STOP = steps;
    if (argv[2]) {
        STOP /= atoi(argv[2]);
    }

    float *x = new float[N * N];
    float *y = new float[N * N];
    float *u = new float[N * N];
    float *u_prev = new float[N * N];


    // Generate mesh and intial condition
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            I = N * j + i;
            x[I] = xmin + h * i;
            y[I] = ymin + h * j;
            u[I] = 0.0f;
            if ((i == 0) || (j == 0)) { u[I] = 200.0f; }
        }
    }

    auto start_time = chrono::high_resolution_clock::now();

    for (int t = 0; t < steps; t++) {
        copy_array(N, u, u_prev);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                I = N * j + i;
                if ((I > N) && (I < N * N - 1 - N) && (I % N != 0) && (I % N != N - 1)) {
                    int add = (alpha*dt)/(h*h)*(u_prev[I+N] + u_prev[I-N] + u_prev[I-1] + u_prev[I+1] - 4*u_prev[I]);
                    u[I] += add;
                }

            }
        }
        if ((STOP != steps && (t % STOP == 0 || t == steps-1))) {
            ofstream temperature("temperature/temperature_" + to_string(t) + ".txt");
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    I = N * j + i;
                    // cout << setprecision(3) << u[I] << "\t";
                    temperature << u[I] << "\t";
                }
                temperature << "\n";
                // cout << endl;
            }
            temperature.close();
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> cpu_time = end_time - start_time;

    printf("cpu\t%d\t%.5f\n", N, cpu_time.count());

    delete u;
    delete u_prev;
    delete x;
    delete y;

    return 0;
}
