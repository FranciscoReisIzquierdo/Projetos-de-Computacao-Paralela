#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>

cudaEvent_t start, stop;
using namespace std;


/*Structure for each point */
typedef struct Point{
	float x;
	float y;
}*point;



typedef struct ClusterInfoParcial{
	float xSum;
	float ySum;
	int elements;
}*clusterInfoParcial;


void recalculateCentroids();
void clean();
void kmeans();
void alloc();
void initialize();
void input(int Points, int Clusters);
void output();

void checkCUDAError (const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		cerr << "Cuda error: " << msg << ", " << cudaGetErrorString( err) << endl;
		exit(-1);
	}
}


void startKernelTime (void) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
}

void stopKernelTime (void) {
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << endl << "Basic profiling: " << milliseconds << " ms have elapsed for the kernel execution" << endl << endl;
}


#endif

/* --------------------------------------------------------- */
