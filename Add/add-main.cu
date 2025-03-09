#include <iostream>
#include <cassert>
#include <ctime>
#include <chrono>

//----------------------------------------------------------------
__global__
void
add(const float *const a, const float *const b, float *const c)
{
    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;
    unsigned int index = y * blockDim.x + x;

    c[index] = a[index] + b[index];
}


//----------------------------------------------------------------
int
main(int argc, char **argv)
{
    cudaError_t cu_err;
    const unsigned int mem_size = sizeof(float) * 1024;
    float *managed_a, *managed_b, *managed_c;

    cu_err = cudaMallocManaged(&managed_a, mem_size);
    assert(cu_err == cudaSuccess);

    cu_err = cudaMallocManaged(&managed_b, mem_size);
    assert(cu_err == cudaSuccess);

    cu_err = cudaMallocManaged(&managed_c, mem_size);
    assert(cu_err == cudaSuccess);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        add<<<1, 1024>>>(managed_a, managed_b, managed_c);
    }
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << ms.count() << "ms" << std::endl;

    cudaFree(managed_a);
    cudaFree(managed_b);
    cudaFree(managed_c);
}
