#include <cuda_runtime.h>
#include <cstdio>

// カーネル関数（float4 の加算）
__global__ void add_float4(const float4* a, const float4* b, float4* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // float4 の各成分を加算
        c[idx].x = a[idx].x + b[idx].x;
        c[idx].y = a[idx].y + b[idx].y;
        c[idx].z = a[idx].z + b[idx].z;
        c[idx].w = a[idx].w + b[idx].w;
    }
}

int main() {
    const int N = 1024;  // float4 配列のサイズ
    float4 h_a[N], h_b[N], h_c[N];

    // データ初期化
    for (int i = 0; i < N; i++) {
        h_a[i] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        h_b[i] = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    }

    // デバイスメモリ確保
    float4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float4));
    cudaMalloc(&d_b, N * sizeof(float4));
    cudaMalloc(&d_c, N * sizeof(float4));

    // データを GPU にコピー
    cudaMemcpy(d_a, h_a, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float4), cudaMemcpyHostToDevice);

    // カーネルを起動
    add_float4<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    // 結果をコピー
    cudaMemcpy(h_c, d_c, N * sizeof(float4), cudaMemcpyDeviceToHost);

    // 結果を表示
    for (int i = 0; i < 5; i++) {
        printf("Result: (%f, %f, %f, %f)\n", h_c[i].x, h_c[i].y, h_c[i].z, h_c[i].w);
    }

    // メモリを解放
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
