#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "../includes/utils.h"


/* --------------------------------------------------------- */

					/* GLOBAL VARIABLES */

/* Variables of the number of points, clusters and threads */
int N, K, Threads;

/*Variable that counts the number of iterations until the algorithm converged */
int interations = 0;


/*Global vector with all clusters */
float *xCentroid;
float *yCentroid;
int *elements;
float *xSum;
float *ySum;
int *clAux;



/* Global vector with all points */
//point vector;
float *xPoints;
float *yPoints;



/* Function that recalculates the new centroids for each cluster */
void recalculateCentroids(){
	for(int i= 0; i< K; i++){
		xCentroid[i] = xSum[i] / elements[i];
		yCentroid[i] = ySum[i] / elements[i];
	}
}


/* Function that "cleans" the information of each clusters after each iteration */
void clean(){
	for(int h= 0; h< K; h++){
		elements[h] = 0;
		xSum[h] = 0.0;
		ySum[h] = 0.0;
	}
}


/* K-means function with unrolling-loop (4 times) */
void kmeans(){

	while(true){
		
		float centroidx0 = xCentroid[0];
		float centroidy0 = yCentroid[0];
		
		clean();
		
		#pragma omp parallel num_threads(Threads)
		#pragma omp for
		for(int i= 0; i< N; i++){

			int cl1 = 0;
			float x1 = xPoints[i];
			float y1 = yPoints[i];
			
			float dist11 = (centroidx0 - x1) * (centroidx0 - x1) + (centroidy0 - y1) * (centroidy0 - y1);
			
			for(int j= 1; j< K; j++){

				float centroidxJ = xCentroid[j];
				float centroidyJ = yCentroid[j];

				float dist21 = (centroidxJ - x1) * (centroidxJ - x1) + (centroidyJ- y1) * (centroidyJ- y1);
				
				if(dist21 < dist11){
					dist11 = dist21;
					cl1 = j;
				}
			}

			clAux[i] = cl1;
		}

		for(int i = 0; i < N; i++){
			int cl1 = clAux[i];
			elements[cl1]++;
			xSum[cl1] += xPoints[i];
			ySum[cl1] += yPoints[i];
		}

		if(interations == 20) break;
		recalculateCentroids();
		interations++;
	}

}


/* Function to allocate memory for the globals vectors */
void alloc(){
	xCentroid = malloc(K * sizeof(float));
	yCentroid = malloc(K * sizeof(float));
	elements = malloc(K * sizeof(int));
	xSum = malloc(K * sizeof(float));
	ySum = malloc(K * sizeof(float));
	xPoints = malloc(N * sizeof(float));
	yPoints = malloc(N * sizeof(float));
	clAux = malloc(N * sizeof(int));
}


/* Function to initialize the clusters and the points*/
void initialize(){
	srand(10);
 	
 	for(int i = 0; i < N; i++){
 		xPoints[i] = (float) rand() / RAND_MAX;
 		yPoints[i] = (float) rand() / RAND_MAX;
 	}


 	for(int i = 0; i < K; i++) {
 		xCentroid[i] = xPoints[i];
 		yCentroid[i] = yPoints[i];
 		elements[i] = 0;
 		xSum[i] = 0.0;
 		ySum[i] = 0.0;
	}
}


/* Function for the input */
void input(int Points, int Clusters, int NumberThreads){
	N = Points;
	K = Clusters;
	Threads = NumberThreads;
}


/* Function for the output */
void output(){
	printf("N = %d, K = %d\n", N, K);
	for(int i= 0; i< K; i++){
		printf("Center: (%.3f, %.3f) : Size: %d\n", xCentroid[i], yCentroid[i], elements[i]);
	}
	printf("Iterations: %d\n", interations);
}