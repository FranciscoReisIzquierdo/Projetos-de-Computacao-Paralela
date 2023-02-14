#include <stdlib.h>
#include "../includes/utils.h"

int main(int argc, char *argv[]){

	int N = atoi(argv[2]);
	int K = atoi(argv[3]);
	int Threads;
	if(argc < 5) Threads = 1;
	else Threads = atoi(argv[4]);

	input(N, K, Threads);
	alloc();
	initialize();
	kmeans();
	output();
}
