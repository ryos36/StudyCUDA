#include <iostream>
#include <chrono>
#include <cassert>

#include <unistd.h>

const unsigned int THREAD_X = 32;
const unsigned int THREAD_Y = 32;

const unsigned int WIDTH = 2 * 1024;
const unsigned int HIGHT = 2 * 1024;
const unsigned int LIFE_SIZE = WIDTH * HIGHT;

//----------------------------------------------------------------
__global__
void life(const bool *const current, bool *const next)
{
    unsigned int x = blockIdx.x * THREAD_X + threadIdx.x;
    unsigned int y = blockIdx.y * THREAD_Y + threadIdx.y;

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

    srand(time(NULL));
    for (unsigned int i = 0; i < LIFE_SIZE; ++i) {
        current[i] = rand() < (rand() / 3);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    assert((THREAD_X * THREAD_Y) <= 1024);
    dim3 blocks(WIDTH / THREAD_X, HIGHT / THREAD_Y);
    dim3 threads(THREAD_X, THREAD_Y);

    for (int i = 0; i < 1000; ++i) {
#ifdef SHOW_LIFE
        cudaDeviceSynchronize();
        show_life(current);
        sleep(1);
#endif
        life<<<blocks, threads>>>(current, next);
        std::swap(current, next);
    }

    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << ms.count() << "ms" << std::endl;

    cudaFree(next);
    cudaFree(current);
}
