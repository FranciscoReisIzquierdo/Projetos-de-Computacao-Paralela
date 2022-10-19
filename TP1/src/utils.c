#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "../includes/utils.h"


/* --------------------------------------------------------- */

					/* GLOBAL VARIABLES */

/* Number of points */
#define N 10000000

/* Number of clusters */
#define K 4

/*Variable that counts the number of iterations until the algorithm converged*/
int interations = 0;


clock_t begin;
clock_t end;



/*Global vector with all clusters */
cluster clusters;



/* Global vector with all points */
point vector;


/* Function that recalculates the new centroids for each cluster */
void recalculateCentroids(){
	for(int i= 0; i< K; i++){
		clusters[i].centroid.x = clusters[i].xsum / clusters[i].elements;
		clusters[i].centroid.y = clusters[i].ysum / clusters[i].elements;
	}
}


/* Function that "cleans" the information of each clusters after each iteration */
void clean(){
	for(int h= 0; h< K; h++){
			clusters[h].elements = 0;
			clusters[h].xsum = 0.0;
			clusters[h].ysum = 0.0;
		}
}


/* K-means function */
void kmeans(){
	begin = clock();

	while(true){
		clean();
		bool converged = true;

		for(int i= 0; i< N; i++){

			int cl = 0;
			float x = vector[i].x;
			float y = vector[i].y;

			float dist1 = (clusters[0].centroid.x - x) * (clusters[0].centroid.x - x) + (clusters[0].centroid.y- y) * (clusters[0].centroid.y- y);

			for(int j= 1; j< K; j++){
				float dist2 = (clusters[j].centroid.x- x) * (clusters[j].centroid.x- x) + (clusters[j].centroid.y- y) * (clusters[j].centroid.y- y);

				if(dist2 < dist1){
					dist1 = dist2;
					cl = j;
				}
			}

			if(vector[i].cluster != cl){
				vector[i].cluster = cl; 
				converged = false;
			}

			clusters[cl].elements++;
			clusters[cl].xsum += vector[i].x;
			clusters[cl].ysum += vector[i].y;
		}

		if(converged) break;
		recalculateCentroids();
		interations++;
	}

}



/* Function to allocate memory for the globals vectors */
void alloc(){
	vector = malloc(N* sizeof(struct Point));
	clusters = malloc(K* sizeof(struct Cluster));
}


/* Function to initialize the clusters and the points*/
void initialize(){
	srand(10);
 
 	for(int i = 0; i < N; i++){
		vector[i].x = (float) rand() / RAND_MAX;
		vector[i].y = (float) rand() / RAND_MAX;
		vector[i].cluster = -1;
 	}


 	for(int i = 0; i < K; i++) {
		clusters[i].centroid.x = vector[i].x;
		clusters[i].centroid.y = vector[i].y;
		clusters[i].xsum = 0.0;
		clusters[i].ysum = 0.0;
		clusters[i].elements = 0;
	}

}

void output(){
	end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("N = %d, K = %d\n", N, K);
	for(int i= 0; i< K; i++){
		printf("Center: (%.3f, %.3f) : Size: %d\n", clusters[i].centroid.x, clusters[i].centroid.y, clusters[i].elements);
	}
	printf("Iterations: %d\n", interations);
	printf("Execution time: %f\n", time_spent);
}
