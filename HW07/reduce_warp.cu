/*
Task: Implement a warp-level reduction using the CUDA intrinsic __shfl_down_sync() to exchange
data between threads in the same warp. This version avoids shared-memory synchronization and
reduces intra-warp latency
*/