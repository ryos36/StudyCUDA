#include <cuda_runtime.h>
#include <cassert>
#include <stdio.h>

// カーネル1
__global__ void kernel1(int *array) {
    if (threadIdx.x == 0) {
        array[0] = 0;
        printf("Kernel 1 executing...\n");
    }
    if (threadIdx.x == 1023) {
        printf("Kernel 1 almoust end...\n");
        while ( array[0] == 0 ) {
        }
        printf("Kernel 1 almoust end...\n");
    }
}

// カーネル2
__global__ void kernel2(int *array) {
    if (threadIdx.x == 0) {
        printf("Kernel 2 executing...\n");
    }
    if (threadIdx.x == 1023) {
        array[0] = 1;
        printf("Kernel 2 almoust end...\n");
    }
}

// カーネル3
__global__ void kernel3() {
    printf("Kernel 3 executing after Kernel 1 and Kernel 2...\n");
}

int main()
{
    cudaError_t cu_err;
    int *array;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Concurrent execution supported: %s\n", prop.concurrentKernels ? "Yes" : "No");

    cu_err = cudaMalloc((void **)&array, 1024 * sizeof(int));
    assert(cu_err == cudaSuccess);

    // ストリームの作成
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    //cudaStreamAttachMemAsync(stream1, array);
    //cudaStreamAttachMemAsync(stream2, array);

    // イベントの作成
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    // カーネル1をストリーム1で非同期実行
    kernel1<<<1, 1024, 0, stream1>>>(array);
    // カーネル1の終了時点でイベント1を記録
    cudaEventRecord(event1, stream1);

    printf("gogo\n");
    // カーネル2をストリーム2で非同期実行
    kernel2<<<1, 1024, 0, stream2>>>(array);
    // カーネル2の終了時点でイベント2を記録
    cudaEventRecord(event2, stream2);
    printf("gogo\n");

    // メインデフォルトストリーム上でイベント1とイベント2を待機
    cudaEventSynchronize(event2);
    cudaEventSynchronize(event1);

    // カーネル3をメインデフォルトストリームで実行
    kernel3<<<1, 1>>>();

    // ストリームとイベントを破棄
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);

    // デバイスの同期
    cudaDeviceSynchronize();
    return 0;
}

