#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int getGPUCores(const cudaDeviceProp& deviceProp)
{
	int mp = deviceProp.multiProcessorCount;
	int cores_mp;

	switch (deviceProp.major)
	{
	case 2:
		cores_mp = (deviceProp.minor == 1) ? 48 : 32;
		break;
	case 3:
		cores_mp = 192;
		break;
	case 5:
		cores_mp = 128;
		break;
	case 6:
        	if (deviceProp.minor == 0) cores_mp = 64;
        	else if (deviceProp.minor == 1 || deviceProp.minor == 2) cores_mp = 128;
        	else cores_mp = -1;
        	break;
	case 7:
		cores_mp = (deviceProp.minor == 0 || deviceProp.minor == 5) ? 64 : -1;
		break;
	default:
		cores_mp = -1;
		break;
	}
	return (cores_mp == -1) ? -1 : (mp * cores_mp);
}

void deviceInfo(const cudaDeviceProp& deviceProp)
{
	printf("\tDevice Name: %s\n", deviceProp.name);
	printf("\tClock Rate: %d kHz\n", deviceProp.clockRate);
	printf("\tNumber of Streaming Multiprocessors: %d\n", deviceProp.multiProcessorCount);
	printf("\tNumber of GPU Cores: %d\n", getGPUCores(deviceProp));
	printf("\tWarp Size: %d\n", deviceProp.warpSize);
	printf("\tAmount of Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
	printf("\tAmount of Constant Memory: %lu bytes\n", deviceProp.totalConstMem);
	printf("\tAmount of Shared Memory per Block: %d bytes\n", deviceProp.sharedMemPerBlock);
	printf("\tNumber of Registers per Block: %d\n", deviceProp.regsPerBlock);
	printf("\tMaximum Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("\tMaximum Size of Each Dimension per Block:\n\tX: %d Y: %d Z: %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("\tMaximum Size of Each Dimension of Grid:\n\tX: %d Y: %d Z: %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
}

int main()
{
	int devices;
	cudaGetDeviceCount(&devices);
	printf("Number of GPU Devices: %d\n", devices);
	for (int id = 0; id < devices; id++)
	{
		printf("Device %d:\n", id);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, id);
		deviceInfo(deviceProp);
		printf("\n");
	}
	return 0;
}
