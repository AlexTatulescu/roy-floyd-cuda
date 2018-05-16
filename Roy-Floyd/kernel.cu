
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 5
#define  INF 9999

int costMatrixGraph[N][N] = {
	{ 0, 10, INF, 5, INF },
	{ INF, 1, 6, INF, 7 },
	{ INF, INF, 0, 9, INF },
	{ INF, 9, INF, 0, INF },
	{ INF, INF, INF, 3, 0 }
};

__global__ void RoyFloyd(int costMatrixGraph[N][N], int k)
{

	int i = threadIdx.x;
	int j = threadIdx.y;

	if (costMatrixGraph[i][k] + costMatrixGraph[k][j] < costMatrixGraph[i][j]) {
		costMatrixGraph[i][j] = costMatrixGraph[i][k] + costMatrixGraph[k][j];
	}
}

int main()
{
	int *matrix;
	int* d_k;
	int numBlocks = 1;

	cudaMalloc(&matrix, N*N * sizeof(int));
	cudaMemcpy(matrix, costMatrixGraph, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_k, sizeof(int));
	for (int k = 0; k < N; ++k)
	{
		cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
		dim3 threadsPerBlock(N, N);
		RoyFloyd<<<numBlocks, threadsPerBlock>>>(matrix, k);
	}
	cudaMemcpy(costMatrixGraph, matrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)
		{
			if (costMatrixGraph[i][j] == INF)
				std::cout << "inf ,";
			else
				std::cout << costMatrixGraph[i][j] << ", ";
		}
		std::cout << std::endl;
	}

	cudaFree(costMatrixGraph);
	cudaFree(matrix);
	system("pause");
	return 0;

}
