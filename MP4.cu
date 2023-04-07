/*
Name: Brant Li
Student #: 20212040
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <map>

#define TOLERANCE 1.0E-8f

__global__ void matMulGPU2(float* A, float* B, float* C, const int N) {
    __shared__ float sFA[2][2];
    __shared__ float sFB[2][2];

    int thrx = threadIdx.x, thry = threadIdx.y;
    int c = blockIdx.x * blockDim.x + thrx;
    int r = blockIdx.y * blockDim.y + thry;
    float s = 0.0;

    for (int i = 0; i < (N + 2 - 1) / 2; i++) {
        if (r < N && i * 2 + thrx < N) {
            sFA[thry][thrx] = A[r * N + i * 2 + thrx];
        } else {
            sFA[thry][thrx] = 0.0;
        }

        if (c < N && i * 2 + thry < N) {
            sFB[thry][thrx] = B[(i * 2 + thry) * N + c];
        } else {
            sFB[thry][thrx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < 2; j++) {
            s += sFA[thry][j] * sFB[j][thrx];
        }

        __syncthreads();
    }

    if (r < N && c < N) {
        C[r * N + c] = s;
    }
}

__global__ void matMulGPU5(float* A, float* B, float* C, const int N) {
    __shared__ float sFA[5][5];
    __shared__ float sFB[5][5];

    int thrx = threadIdx.x, thry = threadIdx.y;
    int c = blockIdx.x * blockDim.x + thrx;
    int r = blockIdx.y * blockDim.y + thry;
    float s = 0.0;

    for (int i = 0; i < (N + 5 - 1) / 5; i++) {
        if (r < N && i * 5 + thrx < N) {
            sFA[thry][thrx] = A[r * N + i * 5 + thrx];
        } else {
            sFA[thry][thrx] = 0.0;
        }

        if (c < N && i * 5 + thry < N) {
            sFB[thry][thrx] = B[(i * 5 + thry) * N + c];
        } else {
            sFB[thry][thrx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < 5; j++) {
            s += sFA[thry][j] * sFB[j][thrx];
        }

        __syncthreads();
    }

    if (r < N && c < N) {
        C[r * N + c] = s;
    }
}

__global__ void matMulGPU10(float* A, float* B, float* C, const int N) {
    __shared__ float sFA[10][10];
    __shared__ float sFB[10][10];

    int thrx = threadIdx.x, thry = threadIdx.y;
    int c = blockIdx.x * blockDim.x + thrx;
    int r = blockIdx.y * blockDim.y + thry;
    float s = 0.0;

    for (int i = 0; i < (N + 10 - 1) / 10; i++) {
        if (r < N && i * 10 + thrx < N) {
            sFA[thry][thrx] = A[r * N + i * 10 + thrx];
        } else {
            sFA[thry][thrx] = 0.0;
        }

        if (c < N && i * 10 + thry < N) {
            sFB[thry][thrx] = B[(i * 10 + thry) * N + c];
        } else {
            sFB[thry][thrx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < 10; j++) {
            s += sFA[thry][j] * sFB[j][thrx];
        }

        __syncthreads();
    }

    if (r < N && c < N) {
        C[r * N + c] = s;
    }
}

__global__ void matMulGPU20(float* A, float* B, float* C, const int N) {
    __shared__ float sFA[20][20];
    __shared__ float sFB[20][20];

    int thrx = threadIdx.x, thry = threadIdx.y;
    int c = blockIdx.x * blockDim.x + thrx;
    int r = blockIdx.y * blockDim.y + thry;
    float s = 0.0;

    for (int i = 0; i < (N + 20 - 1) / 20; i++) {
        if (r < N && i * 20 + thrx < N) {
            sFA[thry][thrx] = A[r * N + i * 20 + thrx];
        } else {
            sFA[thry][thrx] = 0.0;
        }

        if (c < N && i * 20 + thry < N) {
            sFB[thry][thrx] = B[(i * 20 + thry) * N + c];
        } else {
            sFB[thry][thrx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < 20; j++) {
            s += sFA[thry][j] * sFB[j][thrx];
        }

        __syncthreads();
    }

    if (r < N && c < N) {
        C[r * N + c] = s;
    }
}

__global__ void matMulGPU25(float* A, float* B, float* C, const int N) {
    __shared__ float sFA[25][25];
    __shared__ float sFB[25][25];

    int thrx = threadIdx.x, thry = threadIdx.y;
    int c = blockIdx.x * blockDim.x + thrx;
    int r = blockIdx.y * blockDim.y + thry;
    float s = 0.0;

    for (int i = 0; i < (N + 25 - 1) / 25; i++) {
        if (r < N && i * 25 + thrx < N) {
            sFA[thry][thrx] = A[r * N + i * 25 + thrx];
        } else {
            sFA[thry][thrx] = 0.0;
        }

        if (c < N && i * 25 + thry < N) {
            sFB[thry][thrx] = B[(i * 25 + thry) * N + c];
        } else {
            sFB[thry][thrx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < 25; j++) {
            s += sFA[thry][j] * sFB[j][thrx];
        }

        __syncthreads();
    }

    if (r < N && c < N) {
        C[r * N + c] = s;
    }
}

void matMulCPU(float* A, float* B, float* C, const int N){
    #pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) 
				C[i * N + j] += A[i * N + k] * B[k * N + j];
		}
	}
}

void initializeMatrix(float* M, const int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate_n(M, N*N, [&dis, &gen]() { return dis(gen); });
}

void testResult(float* CPU, float* GPU, const int N) {
    float epsilon = TOLERANCE;
    auto max_error = std::max_element(CPU, CPU + (N*N), [&](float a, float b){ return fabs(a - b) < epsilon; });
    if (fabsf(*max_error - *(GPU + std::distance(CPU, max_error))) > epsilon)
        std::cout << "Arrays are mismatched. Maximum error is " << *max_error << ".\n\n";
    else
        std::cout << "Test passed.\n\n";
}

void testArray(float* M, const int N) {
    std::cout << "[";
    std::copy(M, M + N*N - 1, std::ostream_iterator<float>(std::cout, ", "));
    std::cout << M[N*N - 1] << "]\n";
}

void testGPU(float* CPU_A, float* CPU_B, float* resultCPU, const int sizeTile, const int N){
	float *GPU_A, *GPU_B, *GPU_C, *resultGPU;
	cudaMalloc(&GPU_A, N * N * sizeof(float));
	cudaMalloc(&GPU_B, N * N * sizeof(float));
	cudaMalloc(&GPU_C, N * N * sizeof(float));
	resultGPU = (float*)malloc(N * N * sizeof(float));

	cudaMemcpy(GPU_A, CPU_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_B, CPU_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
    cudaEventRecord(start);

    dim3 block(sizeTile, sizeTile, 1);
    dim3 grid((int)ceil(N / (float)block.x), (int)ceil(N / (float)block.y), 1);

    switch (sizeTile) {
        case 2:
            matMulGPU2 <<<grid, block>>> (GPU_A, GPU_B, GPU_C, N);
            break;
        case 5:
            matMulGPU5 <<<grid, block>>> (GPU_A, GPU_B, GPU_C, N);
            break;
        case 10:
            matMulGPU10 <<<grid, block>>> (GPU_A, GPU_B, GPU_C, N);
            break;
        case 20:
            matMulGPU20 <<<grid, block>>> (GPU_A, GPU_B, GPU_C, N);
            break;
        case 25:
            matMulGPU25 <<<grid, block>>> (GPU_A, GPU_B, GPU_C, N);
            break;
        default:
            std::cout << "Invalid tile size.\n";
            break;
    }

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaMemcpy(resultGPU, GPU_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	float timeDuration;
	cudaEventElapsedTime(&timeDuration, start, end);

    std::cout << "GPU Duration: " << timeDuration << " ms Tile Size:" << sizeTile << ".\n\n";
	
    testResult(resultCPU, resultGPU, N);

    cudaFree(GPU_A);
	cudaFree(GPU_B);
	cudaFree(GPU_C);
	free(resultGPU);
}

void completeMat(const int N) {
	std::cout << N << " by " << N << " (Tiled Matrix Multiplication)\n\n";
	
    float* CPU_A = new float[N*N];
	float* CPU_B = new float[N*N];
	float* CPU_C = (float*)calloc(N*N, sizeof(float));

	initializeMatrix(CPU_A, N);
	initializeMatrix(CPU_B, N);

    matMulCPU(CPU_A, CPU_B, CPU_C, N);

    testGPU(CPU_A, CPU_B, CPU_C, 2, N);
    testGPU(CPU_A, CPU_B, CPU_C, 5, N);
    testGPU(CPU_A, CPU_B, CPU_C, 10, N);
    testGPU(CPU_A, CPU_B, CPU_C, 20, N);
    testGPU(CPU_A, CPU_B, CPU_C, 25, N);

    free(CPU_A);
	free(CPU_B);
	free(CPU_C);
	cudaDeviceReset();
}

int main() {
    completeMat(125);
    completeMat(250);
    completeMat(500);
    completeMat(1000);
    completeMat(2000);
}