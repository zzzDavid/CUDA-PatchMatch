#include "cuda_search.h"


__global__ void integer_patch_match(unsigned char* in0_gpu, int width, int height, int blockw, int blockh, int regionw, int regionh, float** motion_x, float** motion_y) 
{
	int x_index = threadIdx.x;
	int y_index = threadIdx.y;

	// get block head pointer
	unsigned char* block_head = in0_gpu + 2 * y_index * width + 2 * x_index;

}

__global__ void interpolate_patch_match()
{

}


int cuda_full_search(unsigned char *out_gpu, unsigned char *in0_gpu, unsigned char *in1_gpu, int width, int height) {

	/* Initialize Parameters */
	int blockw = 16;				// search block width 
	int blockh = 16;				// search block height
	int bnumx = width / blockw;		// block number in x direction
	int bnumy = height / blockh;	// block number in y direction
	int regionw = 16 * 3;			// block search region width
	int regionh = 16 * 3;			// block search region heigt

	int blockwf = 2;				// search block width of interpolation search
	int blockhf = 2;				// search block height of interpolation search
	int interprate = 2;				// interpolate rate
	int regionwf = 16;				// search region width of interpolation search
	int regionhf = 16;				// search region height of interpolation search

	/* Motion Vectors Initialization */
	float **motion_x = new float*[bnumy];				// x motion vector for each block,  2D float array
	for (int i = 0; i < bnumy; i++)
		motion_x[i] = new float[bnumx];
	float **motion_y = new float*[bnumy];				// y motion vector for each block,  2D float array
	for (int i = 0; i < bnumy; i++)
		motion_y[i] = new float[bnumx];					

	/* Parallel Integer Patch Match */
	dim3 threadsPerBlock(bnumy, bnumx);
	integer_patch_match <<<1, threadsPerBlock>>> (in0_gpu, width, height, blockw, blockh, regionw, regionh, motion_x, motion_y);

	interpolate_patch_match <<<1, 1>>>();

	// explicitly free resources
	cudaDeviceReset();
}

