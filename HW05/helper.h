#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <cuda_runtime.h>

// Macro wrapper: converts the CUDA call `val` into a string (#val)
// and passes it with file and line info to `check`.
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

// Checks the return code of CUDA API calls.
// If an error occurs prints file/line, readable error and aborts.
void check(cudaError_t err, const char* func, const char* file, int line);

// Create a vector filled with random integers in [-256, 256].
template <typename T>
std::vector<T> create_rand_vector(size_t n);

// Naive triple-loop matrix multiplication: C = A * B
// A: m x n, B: n x p, C: m x p
template <typename T>
void mm(const T* A, const T* B, T* C, size_t m, size_t n, size_t p);

// Compare two vectors elementwise within an absolute tolerance.
template <typename T>
bool allclose(const std::vector<T>& a, const std::vector<T>& b, T abs_tol);

// Run one randomized test comparing CPU vs GPU results.
template <typename T>
bool random_test_mm_cuda(size_t m, size_t n, size_t p);

// Run multiple random tests in a loop (for stress testing).
// Note: MAT_DIM must be defined elsewhere.
template <typename T>
bool random_multiple_test_mm_cuda(size_t num_tests);

// Measure average runtime of mm_cuda using CUDA events.
// Returns average runtime per test in milliseconds.
template <typename T>
float measure_latency_mm_cuda(size_t m, size_t n, size_t p,
                              size_t num_tests, size_t num_warmups);

// Forward declaration of CUDA kernel function
// This function should be implemented elsewhere (e.g., in a .cu file)
template <typename T>
void mm_cuda(const T* d_A, const T* d_B, T* d_C, size_t m, size_t n, size_t p);

#endif // HELPER_H
