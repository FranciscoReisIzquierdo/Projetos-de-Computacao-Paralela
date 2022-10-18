#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

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

/*Structure for each point */
typedef struct Point{
	float x;
	float y;
	int cluster;
}*point;

typedef struct Cluster{
	struct Point centroid;
	int elements;
	float xsum;
	float ysum;
}*cluster;



/*Each value in index i it's the centroid of the clusters i */
cluster clusters;



/* Global vector with all points */
point vector;

/* --------------------------------------------------------- */


/* Function that recalculates the new centroids for each cluster */
void recalculateCentroids(){
	for(int i= 0; i< K; i++){
		clusters[i].centroid.x = clusters[i].xsum / clusters[i].elements;
		clusters[i].centroid.y = clusters[i].ysum / clusters[i].elements;
	}
}


/* K-means function */
void kmeans(){

	while(true){
		for(int h= 0; h< K; h++){
			clusters[h].elements = 0;
			clusters[h].xsum = 0.0;
			clusters[h].ysum = 0.0;
		}

		bool converged = true;
		for(int i= 0; i< N; i++){
			//euclidian distance

			float x = vector[i].x;
			float y = vector[i].y;

			float dist1 = (float) sqrt(pow((clusters[0].centroid.x - x), 2) + pow((clusters[0].centroid.y- y), 2));
			int cl = 0;

			//printf("Dist1-> %f\n", dist1);

			for(int j= 1; j< K; j++){

				float dist2 = (float) sqrt(pow((clusters[j].centroid.x- x), 2) + pow((clusters[j].centroid.y- y), 2));

				if(dist2 < dist1){
					dist1 = dist2;
					cl = j;
				}
			}

			if(vector[i].cluster != cl){
				vector[i].cluster = cl;
				clusters[cl].elements++;
				clusters[cl].xsum += vector[i].x;
				clusters[cl].ysum += vector[i].y;  
				converged = false;
			}
			else{
				clusters[vector[i].cluster].elements++;
				clusters[vector[i].cluster].xsum += vector[i].x;
				clusters[vector[i].cluster].ysum += vector[i].y;
			}
		}

		if(converged) break;
		recalculateCentroids();
		interations++;
	}

}



/* Function to initialize the vector */
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


int main(){
	begin = clock();
	alloc();
	initialize();
	kmeans();
	output();
}