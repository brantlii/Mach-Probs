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

__global__ void matMulGPU(float* A, float* B, float* C, const int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < N && c < N) {
        float s = 0.0;
        for (int i = 0; i < N; i++) {
            s += A[r * N + i] * B[i * N + c];
        }
        C[r * N + c] = s;
    }
}

__global__ void matMulGPU1(float* A, float* B, float* C, const int N) {
	for (int r = 0; r < N; r++) {
		for (int c = 0; c < N; c++) {
			if (r < N && c < N) {
				C[r * N + c] = 0.0;
				for (int k = 0; k < N; k++)
					C[r * N + c] += A[r * N + k] * B[k * N + c];
			}
		}
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
        std::cout << "Test passed!\n\n";
}

void testArray(float* M, const int N) {
    std::cout << "[";
    std::copy(M, M + N*N - 1, std::ostream_iterator<float>(std::cout, ", "));
    std::cout << M[N*N - 1] << "]\n";
}

float testGPU(float* CPU_A, float* CPU_B, float* resultCPU, const int sizeBlock, const int N) {
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

	if (sizeBlock == 0) {
		matMulGPU1 <<<1, 1>>> (GPU_A, GPU_B, GPU_C, N);
	}
	else {
		dim3 block(sizeBlock, sizeBlock, 1);
		dim3 grid((int)ceil(N / (float)block.x), (int)ceil(N / (float)block.y), 1);
		matMulGPU <<<grid, block>>> (GPU_A, GPU_B, GPU_C, N);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaMemcpy(resultGPU, GPU_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	float durationTIme;
	cudaEventElapsedTime(&durationTIme, start, end);

	testResult(resultCPU, resultGPU, N);

	cudaFree(GPU_A);
	cudaFree(GPU_B);
	cudaFree(GPU_C);
	free(resultGPU);
	return durationTIme;
}

void completeMat(const int N) {
	std::cout << N << " by " << N << " (Matrix Multiplication)\n\n";

	float* CPU_A = new float[N*N];
	float* CPU_B = new float[N*N];
	float* CPU_C = (float*)calloc(N*N, sizeof(float));

	initializeMatrix(CPU_A, N);
	initializeMatrix(CPU_B, N);

    auto cstart = std::chrono::high_resolution_clock::now();
    matMulCPU(CPU_A, CPU_B, CPU_C, N);
    auto cend = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000.0;
    std::cout << "CPU Duration: " << elapsedTime << " ms\n\n";

    float durationTIme;
    durationTIme = testGPU(CPU_A, CPU_B, CPU_C, 0, N);
    std::cout << "GPU Duration " << durationTIme << " ms Block Size: " << 0 << ".\n\n";
    durationTIme = testGPU(CPU_A, CPU_B, CPU_C, 2, N);
    std::cout << "GPU Duration " << durationTIme << " ms Block Size: " << 2 << ".\n\n";
    durationTIme = testGPU(CPU_A, CPU_B, CPU_C, 4, N);
    std::cout << "GPU Duration " << durationTIme << " ms Block Size: " << 4 << ".\n\n";
    durationTIme = testGPU(CPU_A, CPU_B, CPU_C, 10, N);
    std::cout << "GPU Duration " << durationTIme << " ms Block Size: " << 10 << ".\n\n";
    durationTIme = testGPU(CPU_A, CPU_B, CPU_C, 20, N);
    std::cout << "GPU Duration " << durationTIme << " ms Block Size: " << 20 << ".\n\n";
    durationTIme = testGPU(CPU_A, CPU_B, CPU_C, 25, N);
    std::cout << "GPU Duration " << durationTIme << " ms Block Size: " << 25 << ".\n";

    free(CPU_A);
	free(CPU_B);
	free(CPU_C);
    cudaDeviceReset();
}

void calculateTime(const int N) {
    float *CPU_A, *CPU_B, *GPU_A, *GPU_B;

    CPU_A = (float*)malloc(N * N * sizeof(float));
    CPU_B = (float*)malloc(N * N * sizeof(float));
    cudaMalloc((void**)&GPU_A, N * N * sizeof(float));
    cudaMalloc((void**)&GPU_B, N * N * sizeof(float));

    initializeMatrix(CPU_A, N);
    initializeMatrix(CPU_B, N);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float time = 0;

    cudaEventRecord(start);
    cudaMemcpy(GPU_A, CPU_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_B, CPU_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    std::cout << "\n" << N << " by " << N << " matrix, CPU -> GPU Duration: " << time << "ms\n";

    cudaEventRecord(start);
    cudaMemcpy(CPU_A, GPU_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(CPU_B, GPU_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    std::cout << "\n" << N << " by " << N << " matrix, GPU -> CPU Duration: " << time << "ms\n\n";

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(GPU_A);
    cudaFree(GPU_B);
    free(CPU_A);
    free(CPU_B);
}

int main(){
    completeMat(125);
    calculateTime(125);
    completeMat(250);
    calculateTime(250);
    completeMat(500);
    calculateTime(500);
    completeMat(1000);
    calculateTime(1000);
    completeMat(2000);
    calculateTime(2000);
	return 0;
}