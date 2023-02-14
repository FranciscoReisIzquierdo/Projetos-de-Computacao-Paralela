#include <stdlib.h>
#include "utils.h"
#include <stdio.h>


using namespace std;

/* --------------------------------------------------------- */

					/* GLOBAL VARIABLES */

int N, K, NUM_THREADS_PER_BLOCK, BLOCKS;
int interations = 0;

int *elemsTotal;

point sample;
point centroids;
clusterInfoParcial infos;



/* Recalculate and clean function in the GPU */
__global__
	void kmeansKernel(Point *sampleGPU, Point *centroidsGPU, ClusterInfoParcial *infosGPU, int N, int K, int number_threads_per_block, int blocks){

		int id = blockIdx.x * blockDim.x + threadIdx.x; 
		int totalThreads = number_threads_per_block * blocks;

		for(int i = id; i < N; i+= totalThreads){

			Point p = sampleGPU[i];
			Point auxCentroid = centroidsGPU[0];
			int cluster = 0;

			float distance = (auxCentroid.x - p.x) * (auxCentroid.x - p.x) + (auxCentroid.y- p.y) * (auxCentroid.y- p.y);


			for(int j= 1; j< K; j++){

			Point tempCentroid = centroidsGPU[j];
			float tempDistance = (tempCentroid.x - p.x) * (tempCentroid.x - p.x) + (tempCentroid.y- p.y) * (tempCentroid.y- p.y);
			//float distanceAux = (centroidxJ - x) * (centroidxJ - x) + (centroidyJ- y) * (centroidyJ- y);
			
			if(tempDistance < distance){
				distance = tempDistance;
				cluster = j;
				}
			}
			infosGPU[K * id + cluster].xSum += p.x;
			infosGPU[K * id + cluster].ySum += p.y;
			infosGPU[K * id + cluster].elements ++;
		} 
}


/* K-means function in the GPU */
__global__
	void recalculateCentroidsKernel(Point *centroidsGPU, ClusterInfoParcial *infosGPU, int *elemsTotalGPU, int K, int size){
		
		int id = blockIdx.x * blockDim.x + threadIdx.x;

		centroidsGPU[id].x = 0;
		centroidsGPU[id].y = 0;
		elemsTotalGPU[id] = 0;

		for(int i = id; i < size; i+= K){

			centroidsGPU[id].x += infosGPU[i].xSum;
			infosGPU[i].xSum = 0;
			centroidsGPU[id].y += infosGPU[i].ySum;
			infosGPU[i].ySum = 0;
			elemsTotalGPU[id] += infosGPU[i].elements;
			infosGPU[i].elements = 0;
		}

		centroidsGPU[id].x /=  elemsTotalGPU[id];
		centroidsGPU[id].y /= elemsTotalGPU[id];
}


/* Function to launch the kernels */
void launchKMeansKernel(){

	Point *sampleGPU, *centroidsGPU; 
	ClusterInfoParcial *infosGPU;

	int *elemsTotalGPU;
	int size = K * BLOCKS * NUM_THREADS_PER_BLOCK;

	cudaMalloc(&sampleGPU, N * sizeof(struct Point));
	cudaMalloc(&centroidsGPU, K * sizeof(struct Point));
	cudaMalloc(&elemsTotalGPU, K * sizeof(int));
	cudaMalloc(&infosGPU, size * sizeof(struct ClusterInfoParcial));
	checkCUDAError("mem allocation");


	cudaMemcpy(sampleGPU, sample, N * sizeof(struct Point), cudaMemcpyHostToDevice);
	cudaMemcpy(centroidsGPU, centroids, K * sizeof(struct Point), cudaMemcpyHostToDevice);
	cudaMemcpy(elemsTotalGPU, elemsTotal, K * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(infosGPU, infos, size * sizeof(struct ClusterInfoParcial), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy h->d");


	for(; interations < 21; interations++){

		kmeansKernel<<< BLOCKS, NUM_THREADS_PER_BLOCK >>>(sampleGPU, centroidsGPU, infosGPU, N, K, NUM_THREADS_PER_BLOCK, BLOCKS);
		checkCUDAError("kernel invocation");

		recalculateCentroidsKernel<<< 1, K >>>(centroidsGPU, infosGPU, elemsTotalGPU, K, K * NUM_THREADS_PER_BLOCK * BLOCKS);
		checkCUDAError("kernel invocation");
	}	

		cudaMemcpy(elemsTotal, elemsTotalGPU, K * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(centroids, centroidsGPU, K * sizeof(struct Point), cudaMemcpyDeviceToHost);
		cudaFree(sampleGPU);
		cudaFree(centroidsGPU);
		cudaFree(infosGPU);
		cudaFree(elemsTotalGPU);
		checkCUDAError("mem free");
}


/* Function to allocate memory for the globals vectors */
void alloc(){

	sample = (Point *) malloc(N * sizeof(struct Point));
	centroids = (Point *) malloc(K * sizeof(struct Point));
	elemsTotal = (int *) malloc(K * sizeof(int));
	infos = (ClusterInfoParcial *) malloc(K * BLOCKS * NUM_THREADS_PER_BLOCK * sizeof(struct ClusterInfoParcial));
}


/* Function to initialize the clusters and the points*/
void initialize(){
	srand(10);
 	
 	for(int i = 0; i < N; i++){
 		sample[i].x = (float) rand() / RAND_MAX;
 		sample[i].y = (float) rand() / RAND_MAX;
 	}

 	for(int i = 0; i < K; i++) {
 		centroids[i].x = sample[i].x;
 		centroids[i].y = sample[i].y;
 		elemsTotal[i] = 0;
 	}

 	int size = K * BLOCKS * NUM_THREADS_PER_BLOCK;
 	for(int i = 0; i < size; i++){
 		infos[i].xSum = 0.0;
 		infos[i].ySum = 0.0;
 		infos[i].elements = 0;
	}
}


/* Function for the output */
void output(clock_t begin, clock_t end){
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("N = %d, K = %d\n", N, K);
	for(int i= 0; i< K; i++){
		printf("Center: (%.3f, %.3f) : Size: %d\n", centroids[i].x, centroids[i].y, elemsTotal[i]);
	}
	printf("Iterations: %d\n", interations);
	printf("Execution time: %f\n", time_spent);
}


int main(int argc, char *argv[]){

	N = atoi(argv[1]);
	K = atoi(argv[2]);
	BLOCKS = atoi(argv[3]);
	NUM_THREADS_PER_BLOCK = atoi(argv[4]);

	clock_t begin = clock();
	alloc();
	initialize();
	launchKMeansKernel();
	clock_t end = clock();
	output(begin, end);
}