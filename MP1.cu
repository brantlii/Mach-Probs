/*
Name: Brant Li
Student #: 20212040
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

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
	std::cout << "\tDevice Name: " << deviceProp.name << std::endl;
	std::cout << "\tClock Rate: " << deviceProp.clockRate << " kHz" << std::endl;
	std::cout << "\tNumber of Streaming Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
	std::cout << "\tNumber of GPU Cores: " << getGPUCores(deviceProp) << std::endl;
	std::cout << "\tWarp Size: " << deviceProp.warpSize << std::endl;
	std::cout << "\tAmount of Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
	std::cout << "\tAmount of Constant Memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
	std::cout << "\tAmount of Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
	std::cout << "\tNumber of Registers per Block: " << deviceProp.regsPerBlock << std::endl;
	std::cout << "\tMaximum Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
	std::cout << "\tMaximum Size of Each Dimension per Block:" << std::endl;
	std::cout << "\tX: " << deviceProp.maxThreadsDim[0] << " Y: " << deviceProp.maxThreadsDim[1] << " Z: " << deviceProp.maxThreadsDim[2] << std::endl;
	std::cout << "\tMaximum Size of Each Dimension of Grid:" << std::endl;
	std::cout << "\tX: " << deviceProp.maxGridSize[0] << " Y: " << deviceProp.maxGridSize[1] << " Z: " << deviceProp.maxGridSize[2] << std::endl;

}

int main()
{
	int devs;
	cudaGetDeviceCount(&devs);
	std::cout << "Number of GPU Devices: " << devs << std::endl;
	for (int id = 0; id < devs; id++)
	{
		std::cout << "Device " << id << ":" << std::endl;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, id);
		deviceInfo(deviceProp);
		std::cout << std::endl;
	}
	return 0;
}
