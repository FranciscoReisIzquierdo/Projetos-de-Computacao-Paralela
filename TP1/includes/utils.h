#ifndef UTILS_H
#define UTILS_H



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



void recalculateCentroids();
void clean();
void kmeans();
void alloc();
void initialize();
void output();


#endif

/* --------------------------------------------------------- */
