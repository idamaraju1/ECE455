/*
Task. Spawn N threads. Each thread prints "Hello from thread X of N" where X is the thread’s ID (0-based). Join all threads.

Hints
Pass the thread ID as a function argument; store threads in a std::vector<std::thread> and call join() on each.
*/

#include <iostream>
#include <thread>
#include <vector>

int main() {
    const int N = 10; // Number of threads
    std::vector<std::thread> threads;

    // Lambda function to be executed by each thread
    auto print_hello = [](int thread_id, int total_threads) {
        std::cout << "Hello from thread " << thread_id << " of " << total_threads << std::endl;
    };

    // Spawn N threads
    for (int i = 0; i < N; ++i) {
        threads.emplace_back(print_hello, i, N);
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    return 0;
}


