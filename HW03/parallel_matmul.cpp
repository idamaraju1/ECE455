/*
Task: Parallelize a matrix multiplication using OpenMP. Multiply two square matrices of size
512x512
*/

#include <iostream>
#include <vector>
#include <omp.h>

const int N = 512;
using Matrix = std::vector<std::vector<int>>;
void initializeMatrix(Matrix &mat) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = i + j; // Simple initialization
}

int main() {
    Matrix A(N, std::vector<int>(N));
    Matrix B(N, std::vector<int>(N));
    Matrix C(N, std::vector<int>(N, 0));

    initializeMatrix(A);
    initializeMatrix(B);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    std::cout << " C [0][0] = " << C[0][0] << std::endl;

    return 0;
}