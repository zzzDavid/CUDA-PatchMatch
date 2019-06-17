#include "cuda_search.h"


__global__ void integer_patch_match(unsigned char* in0_gpu, const int width, const int height, const int blockw,
	const int blockh, const int regionw, const int regionh, float** motion_x, float** motion_y) 
{
	int block_x_idx = threadIdx.x;
	int block_y_idx = threadIdx.y;

	// get block head pointer
	unsigned char* block_head = in0_gpu + 2 * block_y_idx * blockh * width + 2 * block_x_idx * blockw;
	
	// get region head pointer
	int startIdx_x = block_x_idx * blockw - (regionw - blockw) / 2;					// start pixel index of search region
	int startIdx_y = block_y_idx * blockh - (regionh - blockh) / 2;					
	int endIdx_x = block_x_idx * blockw + blockw + (regionw - blockw) / 2;			// end pixel index of search region
	int endIdx_y = block_y_idx * blockh + blockw + (regionh - blockh) / 2;
	startIdx_x = startIdx_x > 0 ? startIdx_x : 0;
	startIdx_y = startIdx_y > 0 ? startIdx_y : 0;
	endIdx_x = endIdx_x < width ? endIdx_x : width;
	endIdx_y = endIdx_y < height ? endIdx_y : height;
	unsigned char* region_head = in0_gpu + 2 * startIdx_y * width + 2 * startIdx_x;
	int region_actual_w = endIdx_x - startIdx_x + 1;
	int region_actual_h = endIdx_y - startIdx_y + 1;
	// block location relative to region (in pixel)
	int relative_x = block_x_idx * blockw - startIdx_x;
	int relative_y = block_y_idx * blockh - startIdx_y;

	// full search
	float min_sad = 0;
	int x = 0;
	int y = 0;
	for (int j = 0; j < region_actual_h - blockh; j++)
	{
		for (int i = 0; i < region_actual_w - blockw; i++)
		{
			unsigned char* patch_head = region_head + 2 * j * width + 2 * i;
			// calculate SAD
			float sad = SAD(patch_head, block_head, blockw, blockh, width, height);
			if (i == 0 && j == 0)
				min_sad = sad;
			else if (sad < min_sad)
			{
				x = i - relative_x;
				y = j - relative_y;
				min_sad = sad;
			}
		}
	}
	
	motion_x[block_y_idx][block_x_idx] = x;
	motion_y[block_y_idx][block_x_idx] = y;
}

__global__ void interpolate_patch_match(unsigned char* in0_gpu, const int width, const int height, const int blockw,
	const int blockh, const int regionw, const int regionh, float** motion_x, float** motion_y, int interpolate)
{

}

__device__ float SAD(unsigned char* patch_header, unsigned char* block_header, int blockw, int blockh, int width, int height)
{
	float sum = 0;
	for (int j = 0; j < blockh; j++)
	{
		for (int i = 0; i < blockw; i++)
		{
			int num1 = (int)*(patch_header + 2 * width * j + 2 * i);
			int num2 = (int)*(block_header + 2 * width * j + 2 * i);
			float temp = float(num1 - num2);
			sum += temp > 0 ? temp : -temp;
		}
	}
	return sum;
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
	int interpolate = 2;				// interpolate rate
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

	/* Parallel Interpolate Patch Match */
	interpolate_patch_match <<<1, threadsPerBlock>>>(in0_gpu, width, height, blockwf, blockhf, regionwf, regionhf, motion_x, motion_y, interpolate);

	// explicitly free resources
	cudaDeviceReset();

	return 0;
}

