#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

int N;
#define MAX 1000
#define MIN 0

using namespace std;

vector<vector<vector<int > > > getBuckets(int threads, int buckets, int n) {
    vector<int> vector3(n, 0);
    vector<vector<int > > vector2(buckets, vector3);
    vector<vector<vector<int > > > vector1(threads, vector2);
    return vector1;
}

// https://www.thepolyglotdeveloper.com/2019/04/sort-vector-integers-quicksort-algorithm-cpp/
int partition(vector<int> &values, int left, int right) {
    int pivotIndex = left + (right - left) / 2;
    int pivotValue = values[pivotIndex];
    int i = left, j = right;
    int temp;
    while(i <= j) {
        while(values[i] < pivotValue) {
            i++;
        }
        while(values[j] > pivotValue) {
            j--;
        }
        if(i <= j) {
            temp = values[i];
            values[i] = values[j];
            values[j] = temp;
            i++;
            j--;
        }
    }
    return i;
}

// https://www.thepolyglotdeveloper.com/2019/04/sort-vector-integers-quicksort-algorithm-cpp/
void quicksort(vector<int> &values, int left, int right) {
    if(left < right) {
        int pivotIndex = partition(values, left, right);
        quicksort(values, left, pivotIndex - 1);
        quicksort(values, pivotIndex, right);
    }
}

bool sorted(vector<int> v) {
    for (int i = 1; i < v.size(); ++i)
        if (v[i-1] > v[i])
           return false;
    return true;
}

bool between(int a, int j, int interval) {
    return a >= j*interval && a < (j+1)*interval;
}


vector<int> getRandomVector(){
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

vector<int> bucketsort1(int threads, int buckets_count) {
    double prepare_start, rand_start, split_start, sort_start, concat_start;

    // Przygotowania
    prepare_start = omp_get_wtime();
    vector<int> r;
    vector<vector<vector<int> > > buckets = getBuckets(1, buckets_count, N);
    vector<int> indexes(buckets_count, 0);
    int interval = (MAX-MIN) / buckets_count;
    cout << "prepare:" << (omp_get_wtime() - prepare_start) << " s" << endl;

    // Losowanie warto??ci wektora https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
    rand_start = omp_get_wtime();
    vector<int> a = getRandomVector();
    cout << "rand:\t" << (omp_get_wtime() - rand_start) << " s" << endl;

    // Podzia?? na kube??ki
    split_start = omp_get_wtime();
    #pragma omp parallel for shared(a, buckets, interval) schedule(static, N/threads)
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < buckets_count; j++) {
            int value = a[(i + omp_get_thread_num()) % N];
            if (between(value, j, interval)) {
                buckets[0][j][indexes[j]++] = value;
                break;
            }
        }
    }
    cout << "split:\t" << (omp_get_wtime() - split_start) << " s" << endl;
    
    // Sortowanie kube??kow
    sort_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < buckets_count; i++) {
        quicksort(buckets[0][i], 0, indexes[i]-1);
    }
    cout << "sort:\t" << (omp_get_wtime() - sort_start) << " s" << endl;
    
    // ????czenie kube??kow
    concat_start = omp_get_wtime();
    for (int i = 0; i < buckets_count; i++) {
        r.insert(r.end(), buckets[0][i].begin(), buckets[0][i].begin() + indexes[i]-1);
    }
    cout << "concat:\t" << (omp_get_wtime() - concat_start) << " s" << endl;

    return r;
}

vector<int> bucketsort2(int threads, int buckets_count) {

    double prepare_start, rand_start, split_start, sort_start, concat_start;

    // Przygotowania
    prepare_start = omp_get_wtime();
    vector<vector<vector<int > > > buckets = getBuckets(threads, buckets_count, 0);
    vector<int> result;
    double bucket_interval = 100.0 / buckets_count;
    cout << "prepare:" << (omp_get_wtime() - prepare_start) << " s" << endl;

    // Losowanie warto??ci wektora https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
    rand_start = omp_get_wtime();
    vector<int> vector = getRandomVector();
    cout << "rand:\t" << (omp_get_wtime() - rand_start) << " s" << endl;

    // Podzia?? na kube??ki
    split_start = omp_get_wtime();


    #pragma omp parallel for shared(buckets, vector, bucket_interval) schedule(dynamic)
    for (int i = 0; i < buckets_count; i++) {
        for (int j : vector) {
            if (j >= i*bucket_interval && j <= (i+1)*bucket_interval) {
                buckets[0][i].push_back(j);
            }
        }
    }

    cout << "split:\t" << (omp_get_wtime() - split_start) << " s" << endl;


    // Sortowanie kube??kow
    sort_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < buckets_count; i++) {
        quicksort(buckets[0][i], 0, buckets[0][i].size()-1);
    }
    cout << "sort:\t" << (omp_get_wtime() - sort_start) << " s" << endl;

    // ????czenie kube??kow
    concat_start = omp_get_wtime();

    for (int i = 0; i < buckets_count; i++) {
        result.insert(result.end(), buckets[0][i].begin(), buckets[0][i].end());
    }
    cout << "concat:\t" << (omp_get_wtime() - concat_start) << " s" << endl;

    return result;
}

vector<int> bucketsort3(int threads, int buckets_count) {
    
    double prepare_start, rand_start, split_start, sort_start, concat_start, concat_threads_buckets_start;

    // Przygotowania
    prepare_start = omp_get_wtime();
    vector<vector<vector<int > > > buckets = getBuckets(threads, buckets_count, 0);
    vector<int> result;
    double bucket_interval = 100.0 / buckets_count;
    cout << "prepare:" << (omp_get_wtime() - prepare_start) << " s" << endl;

    // Losowanie warto??ci wektora https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
    rand_start = omp_get_wtime();
    vector<int> vector = getRandomVector();
    cout << "rand:\t" << (omp_get_wtime() - rand_start) << " s" << endl;

    // Podzia?? na kube??ki
    split_start = omp_get_wtime();


    #pragma omp parallel for shared(buckets, vector, bucket_interval) schedule(static, vector.size()/threads)
    for (int i : vector) {
        for (int j = 0; j < buckets_count; j++) {
            if (i >= j*bucket_interval && i <= (j+1)*bucket_interval) {
                buckets[omp_get_thread_num()][j].push_back(i);

                break;
            }
        }
    }

    cout << "split:\t" << (omp_get_wtime() - split_start) << " s" << endl;


    concat_threads_buckets_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < buckets_count; i++) {
        for (int j = 1; j < threads; j++) {
            buckets[0][i].insert(buckets[0][i].end(), buckets[j][i].begin(), buckets[j][i].end());
        }
    }
    cout << "concat threads:\t" << (omp_get_wtime() - concat_threads_buckets_start) << " s" << endl;

    // Sortowanie kube??kow
    sort_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < buckets_count; i++) {
        quicksort(buckets[0][i], 0, buckets[0][i].size()-1);
    }
    cout << "sort:\t" << (omp_get_wtime() - sort_start) << " s" << endl;

    // ????czenie kube??kow
    concat_start = omp_get_wtime();

    for (int i = 0; i < buckets_count; i++) {
        result.insert(result.end(), buckets[0][i].begin(), buckets[0][i].end());
    }
    cout << "concat:\t" << (omp_get_wtime() - concat_start) << " s" << endl;

    return result;


}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Prawidlowe uruchomienie: ./sort <liczba_watkow> <liczba_kubelkow> <rozmiar problemu>" << endl;
        return 1;
    }

    int threads = atoi(argv[1]);
    int buckets_count = atoi(argv[2]);
    N = atoi(argv[3]);

    omp_set_num_threads(threads);
    double start;

    start = omp_get_wtime();
    cout << "sortowanie 1" << endl;
    vector<int> result1 = bucketsort1(threads, buckets_count);
    cout << "sum:\t" << (omp_get_wtime() - start) << " s" << endl;
    cout << (sorted(result1) ? "" : "UPS, TABLICA NIEPOSORTOWANA!!!!") << endl;

     start = omp_get_wtime();
     cout << "sortowanie 2" << endl;
     vector<int> result2 = bucketsort2(threads, buckets_count);
     cout << "sum:\t" << (omp_get_wtime() - start) << " s " << endl;
     cout << (sorted(result2) ? "" : "UPS, TABLICA NIEPOSORTOWANA!!!!") << endl;

     start = omp_get_wtime();
     cout << "sortowanie 3" << endl;
     vector<int> result3 = bucketsort3(threads, buckets_count);
     cout << "sum:\t" << (omp_get_wtime() - start) << " s " << endl;
     cout << (sorted(result3) ? "" : "UPS, TABLICA NIEPOSORTOWANA!!!!") << endl;

    return 0;
}