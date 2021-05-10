#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

using namespace std;

vector<int> getRandomVector(int N, int MAX, int MIN){
    vector<int> a;
     #pragma omp parallel
    {
        vector<int> vec_private;
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for nowait
        for(int i = 0; i < N; i++) {
            vec_private.push_back(rand_r(&myseed) % (MAX - MIN) + MIN);
        }
        #pragma omp critical
        a.insert(a.end(), vec_private.begin(), vec_private.end());
    }
    return a;
}

void test_randomization(int N, int buckets, int MAX) {
    double bucket_interval = MAX / buckets;
    vector <int> random = getRandomVector(N, MAX, 0);
    vector <int> counts(buckets, 0);
    for (int j = 0; j < random.size(); j++) {
        int index = random[j] / bucket_interval;
        counts[index]++;
    }
    double max = 0.0;
    double exp = 1.0 / buckets;
    for (int j = 0; j < counts.size(); j++) {
        double change = (double) counts[j]/N; 
        if (abs(change-exp) > max) max = abs(change-exp);
    }
    cout << N << "\t" << buckets << "\t" << max << "\t" << max/exp << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Prawidlowe uruchomienie: ./sort <rozmiar problemu> <liczba_kubelkow> " << endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int buckets = atoi(argv[2]);
    omp_set_num_threads(8);

    cout.precision(10);

    test_randomization(N, buckets, 1000);

    return 0;
}