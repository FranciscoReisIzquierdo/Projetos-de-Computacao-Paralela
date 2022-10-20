#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
#include <time.h>
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


/* K-means function with unrolling-loop (4 times) */
void kmeans(){
	begin = clock();


	while(true){
		
		float centroidx0 = clusters[0].centroid.x;
		float centroidy0 = clusters[0].centroid.y;
		
		clean();
		bool converged = true;

		for(int i= 0; i< N; i+=4){


			int cl1 = 0;
			float x1 = vector[i].x;
			float y1 = vector[i].y;

			int cl2 = 0;
			float x2 = vector[i+1].x;
			float y2 = vector[i+1].y;

			int cl3 = 0;
			float x3 = vector[i+2].x;
			float y3 = vector[i+2].y;

			int cl4 = 0;	
			float x4 = vector[i+3].x;
			float y4 = vector[i+3].y;

			float dist11 = (centroidx0 - x1) * (centroidx0 - x1) + (centroidy0- y1) * (centroidy0- y1);
			float dist12 = (centroidx0 - x2) * (centroidx0 - x2) + (centroidy0- y2) * (centroidy0- y2);
			float dist13 = (centroidx0 - x3) * (centroidx0 - x3) + (centroidy0- y3) * (centroidy0- y3);
			float dist14 = (centroidx0 - x4) * (centroidx0 - x4) + (centroidy0- y4) * (centroidy0- y4);

			for(int j= 1; j< K; j++){

				float centroidxJ = clusters[j].centroid.x;
				float centroidyJ = clusters[j].centroid.y;

				float dist21 = (centroidxJ - x1) * (centroidxJ - x1) + (centroidyJ- y1) * (centroidyJ- y1);
				float dist22 = (centroidxJ - x2) * (centroidxJ - x2) + (centroidyJ- y2) * (centroidyJ- y2);
				float dist23 = (centroidxJ - x3) * (centroidxJ - x3) + (centroidyJ- y3) * (centroidyJ- y3);
				float dist24 = (centroidxJ - x4) * (centroidxJ - x4) + (centroidyJ- y4) * (centroidyJ- y4);


				if(dist21 < dist11){
					dist11 = dist21;
					cl1 = j;
				}

				if(dist22 < dist12){
					dist12 = dist22;
					cl2 = j;
				}

				if(dist23 < dist13){
					dist13 = dist23;
					cl3 = j;
				}

				if(dist24 < dist14){
					dist14 = dist24;
					cl4 = j;
				}

			}

			if(vector[i].cluster != cl1 || vector[i+1].cluster != cl2 || vector[i+2].cluster != cl3 || vector[i+3].cluster != cl4) converged = false;
		
			vector[i].cluster = cl1; 
			vector[i+1].cluster = cl2; 
			vector[i+2].cluster = cl3;
			vector[i+3].cluster = cl4; 

			clusters[cl1].elements++;
			clusters[cl1].xsum += x1;
			clusters[cl1].ysum += y1;

			clusters[cl2].elements++;
			clusters[cl2].xsum += x2;
			clusters[cl2].ysum += y2;

			clusters[cl3].elements++;
			clusters[cl3].xsum += x3;
			clusters[cl3].ysum += y3;

			clusters[cl4].elements++;
			clusters[cl4].xsum += x4;
			clusters[cl4].ysum += y4;
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
