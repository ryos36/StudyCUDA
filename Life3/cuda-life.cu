#include <iostream>
#include <chrono>
#include <cassert>

#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

#include <curand_kernel.h>
#include <stdio.h>

const unsigned int THREAD_X = 32;
const unsigned int THREAD_Y = 32;

const unsigned int WIDTH = 32 * 1024;
const unsigned int HIGHT = 32 * 1024;
const unsigned int LIFE_SIZE = WIDTH * HIGHT;

const unsigned int INIT_THREAD_N = 1024;

//----------------------------------------------------------------
__global__
void life(const bool *const current, bool *const next)
{
    unsigned int x = blockIdx.x * THREAD_X + threadIdx.x;
    unsigned int y = blockIdx.y * THREAD_Y + threadIdx.y;
    if (( x == 0 ) && ( y == 0 )) {
        printf("life game start!!\n");
    }

    unsigned int n = 0;
    for (int yd = -1; yd <= 1; yd++) {
        for (int xd = -1; xd <= 1; xd++) {
            if (yd == 0 && xd == 0) {
                continue;
            }
            int target_x = x + xd;
            int target_y = y + yd;
            if (( target_x < 0 ) ||
                ( target_x >= WIDTH ) || 
                ( target_y < 0 ) ||
                ( target_y >= WIDTH )) {
                continue;
            }
            n += current[target_y * WIDTH + target_x];
        }
    }

    unsigned int center_index = y * WIDTH + x;

    if (current[center_index]) {
        next[center_index] = (n == 2 || n == 3);
    } else {
        next[center_index] = (n == 3);
    }
}

//----------------------------------------------------------------
__global__
void init_life(bool *const life_board, unsigned int seed, unsigned int *const count)
{
    curandState state;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((unsigned long long)seed, id, 0, &state);

    unsigned int v0 = curand_uniform(&state);
    unsigned int v1 = curand_uniform(&state);

    life_board[id] = (v1 < (v0 / 4));
}

//----------------------------------------------------------------
void
show_life(const bool *const life)
{
    int rv;
    static int age;

    rv = system("/usr/bin/clear");
    assert(rv == 0);
    std::cout << "age:" << age << std::endl;
    for (unsigned int y = 0; y < 20; ++y) {
        for (unsigned int x = 0; x < 70; ++x) {
            std::cout << (life[y*WIDTH+x] ? "#" : " ");
        }
        std::cout << std::endl;
    }
    age++;
}


//----------------------------------------------------------------
int
main(int argc, char **argv)
{
    cudaError_t cu_err;
    bool *current, *next;
    const unsigned int mem_size_for_life = LIFE_SIZE * sizeof(bool);

    cu_err = cudaMallocManaged(&current, mem_size_for_life);
    assert(cu_err == cudaSuccess);

    cu_err = cudaMallocManaged(&next, mem_size_for_life);
    assert(cu_err == cudaSuccess);

    //----------------------------------------------------------------
    cudaStream_t stream;
    cudaEvent_t event;

    cu_err = cudaStreamCreate(&stream);
    assert(cu_err == cudaSuccess);
    cu_err = cudaEventCreate(&event);
    assert(cu_err == cudaSuccess);

    //----------------------------------------------------------------
    unsigned int *count;
    cu_err = cudaMallocManaged(&count, sizeof(unsigned int));
    assert(cu_err == cudaSuccess);

    std::cout << "処理開始" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    unsigned int seed = 817;
    seed = time(nullptr);

    nvtxRangePush("Init Life");
    init_life<<<LIFE_SIZE/INIT_THREAD_N, INIT_THREAD_N, 0, stream>>>(current, seed, count);

    //----------------------------------------------------------------
    assert((THREAD_X * THREAD_Y) <= 1024);
    dim3 blocks(WIDTH / THREAD_X, HIGHT / THREAD_Y);
    dim3 threads(THREAD_X, THREAD_Y);

    cudaEventRecord(event, stream);
    cudaStreamWaitEvent(stream, event, 0);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "初期化終了: " << ms.count() << "ms" << std::endl;
    cudaStreamAttachMemAsync(stream, current, cudaMemAttachSingle);
    
    cudaStreamSynchronize(stream);
    nvtxRangePop();

    end_time = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "処理開始: " << ms.count() << "ms" << std::endl;

    life<<<blocks, threads>>>(current, next);

    end_time = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "処理終了: " << ms.count() << "ms" << std::endl;
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "最終シンクロ終了: " << ms.count() << "ms" << std::endl;

    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    cudaFree(next);
    cudaFree(current);
}
