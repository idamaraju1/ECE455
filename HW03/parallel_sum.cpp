/*
Task: Convert a sequential sum of an array into an OpenMP parallel version using #pragma omp
parallel for. Use an array of size 1,000,000 and compute the sum of all elements
*/

#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    const int size = 1000000;
    std::vector<int> arr(size);
    
    // Initialize the array with some values
    for (int i = 0; i < size; ++i) {
        arr[i] = i + 1; // Fill with values 1 to 1,000,000
    }

    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }

    std::cout << "Sum of array elements: " << sum << std::endl;

    return 0;
}