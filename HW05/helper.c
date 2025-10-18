// helper.c (formatted)
// Note: This file uses C++ features and CUDA runtime API.

#include <vector>
#include <random>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Macro wrapper: converts the CUDA call `val` into a string (#val)
// and passes it with file and line info to `check`.
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

// Checks the return code of CUDA API calls.
// If an error occurs prints file/line, readable error and aborts.
inline void check(cudaError_t err, const char* func, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Create a vector filled with random integers in [-256, 256].
template <typename T>
std::vector<T> create_rand_vector(size_t n)
{
    std::random_device rd;                       // Non-deterministic seed
    std::default_random_engine eng(rd());        // Random engine
    std::uniform_int_distribution<int> dist(-256, 256);

    std::vector<T> vec(n);
    for (size_t i = 0; i < n; ++i) {
        vec[i] = static_cast<T>(dist(eng));
    }
    return vec;
}

// Naive triple-loop matrix multiplication: C = A * B
// A: m x n, B: n x p, C: m x p
template <typename T>
void mm(const T* A, const T* B, T* C, size_t m, size_t n, size_t p)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T acc = 0;
            for (size_t k = 0; k < n; ++k) {
                acc += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = acc;
        }
    }
}

// Compare two vectors elementwise within an absolute tolerance.
template <typename T>
bool allclose(const std::vector<T>& a, const std::vector<T>& b, T abs_tol)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > abs_tol) {
            std::cout << a[i] << " " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Run one randomized test comparing CPU vs GPU results.
template <typename T>
bool random_test_mm_cuda(size_t m, size_t n, size_t p)
{
    // Allocate and initialize host matrices
    const std::vector<T> mat1_vec = create_rand_vector<T>(m * n);
    const std::vector<T> mat2_vec = create_rand_vector<T>(n * p);
    std::vector<T> mat3_vec(m * p); // CPU result
    std::vector<T> mat4_vec(m * p); // GPU result

    const T* mat1 = mat1_vec.data();
    const T* mat2 = mat2_vec.data();
    T* mat3 = mat3_vec.data();
    T* mat4 = mat4_vec.data();

    // Compute reference result on CPU
    mm(mat1, mat2, mat3, m, n, p);

    // Allocate GPU memory
    T *d_mat1 = nullptr, *d_mat2 = nullptr, *d_mat4 = nullptr;
    checkCuda(cudaMalloc(&d_mat1, sizeof(T) * mat1_vec.size()));
    checkCuda(cudaMalloc(&d_mat2, sizeof(T) * mat2_vec.size()));
    checkCuda(cudaMalloc(&d_mat4, sizeof(T) * mat4_vec.size()));

    // Copy input matrices to device
    checkCuda(cudaMemcpy(d_mat1, mat1, sizeof(T) * mat1_vec.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat2, mat2, sizeof(T) * mat2_vec.size(), cudaMemcpyHostToDevice));

    // Launch CUDA kernel (user-defined function)
    // Assumes mm_cuda is defined elsewhere.
    mm_cuda(d_mat1, d_mat2, d_mat4, m, n, p);
    cudaDeviceSynchronize();

    // Copy result back to host
    checkCuda(cudaMemcpy(mat4, d_mat4, sizeof(T) * mat4_vec.size(), cudaMemcpyDeviceToHost));

    // Free device memory
    checkCuda(cudaFree(d_mat1));
    checkCuda(cudaFree(d_mat2));
    checkCuda(cudaFree(d_mat4));

    // Compare CPU vs GPU results
    return allclose<T>(mat3_vec, mat4_vec, static_cast<T>(1e-4));
}

// Run multiple random tests in a loop (for stress testing).
// Note: MAT_DIM must be defined elsewhere.
template <typename T>
bool random_multiple_test_mm_cuda(size_t num_tests)
{
    size_t m { MAT_DIM }, n { MAT_DIM }, p { MAT_DIM };
    for (size_t i = 0; i < num_tests; ++i) {
        if (!random_test_mm_cuda<T>(m, n, p)) {
            return false;
        }
    }
    return true;
}

// Measure average runtime of mm_cuda using CUDA events.
// Returns average runtime per test in milliseconds.
template <typename T>
float measure_latency_mm_cuda(size_t m, size_t n, size_t p,
                              size_t num_tests, size_t num_warmups)
{
    cudaEvent_t startEvent, stopEvent;
    float time_ms = 0.0f;

    // Create CUDA events for timing
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    // Allocate device matrices once
    T *d_mat1 = nullptr, *d_mat2 = nullptr, *d_mat4 = nullptr;
    checkCuda(cudaMalloc(&d_mat1, sizeof(T) * m * n));
    checkCuda(cudaMalloc(&d_mat2, sizeof(T) * n * p));
    checkCuda(cudaMalloc(&d_mat4, sizeof(T) * m * p));

    // Warm-up runs (not timed)
    for (size_t i = 0; i < num_warmups; ++i) {
        mm_cuda(d_mat1, d_mat2, d_mat4, m, n, p);
    }
    cudaDeviceSynchronize();

    // Timed runs using CUDA events
    checkCuda(cudaEventRecord(startEvent, 0));
    for (size_t i = 0; i < num_tests; ++i) {
        mm_cuda(d_mat1, d_mat2, d_mat4, m, n, p);
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&time_ms, startEvent, stopEvent)); // time in ms

    // Free device memory and events
    checkCuda(cudaFree(d_mat1));
    checkCuda(cudaFree(d_mat2));
    checkCuda(cudaFree(d_mat4));
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));

    // Return average runtime per test (milliseconds)
    return time_ms / static_cast<float>(num_tests);
}