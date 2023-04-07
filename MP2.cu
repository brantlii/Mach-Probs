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

#define BLOCK 16
#define TOLERANCE 1.0E-8f

__global__ void matAdd(float* A, float* B, float* C, const int N) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	if (c < N && r < N) {
		int i = r * N + c;
		C[i] = A[i] + B[i];
	}
}

__global__ void matAddRow(float* A, float* B, float* C, const int N) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int s = gridDim.x * blockDim.x;
	for (int r = t; r < N; r += s) {
		for (int i = 0; i < N; i++) {
			C[r * N + i] = A[r * N + i] + B[r * N + i];
		}
	}
}

__global__ void matAddCol(float* A, float* B, float* C, const int N) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int s = gridDim.x * blockDim.x;
	for (int c = t; c < N; c += s) {
		for (int i = 0; i < N; i++) {
			C[i * N + c] = A[i * N + c] + B[i * N + c];
		}
	}
}

void testResult(float* CPU, float* GPU, const int N) {
    float epsilon = TOLERANCE;
    auto max_error = std::max_element(CPU, CPU + (N*N), [&](float a, float b){ return fabs(a - b) < epsilon; });
    if (fabsf(*max_error - *(GPU + std::distance(CPU, max_error))) > epsilon)
        std::cout << "Arrays are mismatched. Maximum error is " << *max_error << ".\n\n";
    else
        std::cout << "Test passed!\n\n";
}

void matSum(float* A, float* B, float* C, const int N) {
    #pragma omp parallel for
	for (int i = 0; i < (N*N); i++)
		C[i] = A[i] + B[i];
}

void initializeMatrix(float* M, const int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate_n(M, N*N, [&dis, &gen]() { return dis(gen); });
}

void testArray(float* M, const int N) {
    std::cout << "[";
    std::copy(M, M + N*N - 1, std::ostream_iterator<float>(std::cout, ", "));
    std::cout << M[N*N - 1] << "]\n";
}

void completeMat(const int N) {
	std::cout << N << " by " << N << " (Matrix Addition)\n\n";

	float* CPU_A = new float[N*N];
	float* CPU_B = new float[N*N];
	float* CPU_C = (float*)calloc(N*N, sizeof(float));
	float* CPU_C1 = (float*)calloc(N*N, sizeof(float));

	initializeMatrix(CPU_A, N);
	initializeMatrix(CPU_B, N);

	float* GPU_A, *GPU_B, *GPU_C, *GPU_C1, *GPU_C2;
	cudaMalloc((void**)&GPU_A, N*N*sizeof(float));
	cudaMalloc((void**)&GPU_B, N*N*sizeof(float));
	cudaMalloc((void**)&GPU_C, N*N*sizeof(float));
	cudaMalloc((void**)&GPU_C1, N*N*sizeof(float));
	cudaMalloc((void**)&GPU_C2, N*N*sizeof(float));

	cudaMemcpy(GPU_A, CPU_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_B, CPU_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

	auto cstart = std::chrono::high_resolution_clock::now();
    matSum(CPU_A, CPU_B, CPU_C, N);
    auto cend = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000.0 ;
    std::cout << "CPU Duration: " << elapsedTime << " ms\n\n";

	dim3 block(BLOCK, BLOCK);
	dim3 thread((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	
    float durationTime;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	matAdd <<<thread, block >>> (GPU_A, GPU_B, GPU_C, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&durationTime, start, end);
	std::cout << "GPU Duration: " << durationTime << " ms (Thread/Element)\n\n";

	cudaMemcpy(CPU_C1, GPU_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	testResult(CPU_C, CPU_C1, N);

	dim3 blockA(BLOCK);
	dim3 threadA((N + blockA.x - 1) / blockA.x);

	cudaEventRecord(start);
	matAddRow <<<threadA, blockA >>>(GPU_A, GPU_B, GPU_C1, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&durationTime, start, end);
	std::cout << "GPU Duration: " << durationTime << " ms (Thread/Row)\n\n";

	cudaMemcpy(CPU_C1, GPU_C1, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    testResult(CPU_C, CPU_C1, N);

	dim3 blockB(BLOCK);
	dim3 threadB((N + blockB.x - 1) / blockB.x);

	cudaEventRecord(start);
	matAddCol <<<threadB, blockB >>>(GPU_A, GPU_B, GPU_C2, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&durationTime, start, end);
	std::cout << "GPU Duration: " << durationTime << " ms (Thread/Column)\n\n";

	cudaMemcpy(CPU_C1, GPU_C2, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	testResult(CPU_C, CPU_C1, N);

	cudaFree(GPU_A);
	cudaFree(GPU_B);
	cudaFree(GPU_C);
	cudaFree(GPU_C1);
	cudaFree(GPU_C2);
	free(CPU_A);
	free(CPU_B);
	free(CPU_C);
	free(CPU_C1);
	cudaDeviceReset();
}

int main() {
	completeMat(125);
	completeMat(250);
	completeMat(500);
	completeMat(1000);
	completeMat(2000);
	return 0;
}