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
	for (int j = 0; j < region_actual_h - blockh + 1; j++)
	{
		for (int i = 0; i < region_actual_w - blockw + 1; i++)
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
	const int blockh, const int blockwf, const int blockhf, const int regionwf, const int regionhf, float** motion_x, float** motion_y, int interpolate)
{
	int block_x_idx = threadIdx.x;
	int block_y_idx = threadIdx.y;

	// get integer motion vector
	float mv_x = motion_x[block_y_idx][block_x_idx];
	float mv_y = motion_y[block_y_idx][block_x_idx];

	// get block head pointer
	int bx = block_x_idx * blockw + (blockw - blockwf) / 2;
	int by = block_y_idx * blockh + (blockh - blockhf) / 2;
	unsigned char* block_head = in0_gpu + 2 * by * width + 2 * bx;

	// get region head pointer
	int startIdx_x = bx + mv_x - (regionwf - blockwf) / 2;							// start pixel index of search region
	int startIdx_y = by + mv_y - (regionhf - blockhf) / 2;
	int endIdx_x = startIdx_x + regionwf;											// end pixel index of search region
	int endIdx_y = startIdx_y + regionhf;
	// restrain the region inside the frame
	startIdx_x = startIdx_x > 0 ? startIdx_x : 0;
	startIdx_y = startIdx_y > 0 ? startIdx_y : 0;
	endIdx_x = endIdx_x < width ? endIdx_x : width;
	endIdx_y = endIdx_y < height ? endIdx_y : height;
	unsigned char* region_head = in0_gpu + 2 * startIdx_y * width + 2 * startIdx_x; // the head pointer of search region
	int region_actual_w = endIdx_x - startIdx_x + 1;
	int region_actual_h = endIdx_y - startIdx_y + 1;
	// block location relative to region (in pixel)
	int relative_x = bx - startIdx_x;
	int relative_y = by - startIdx_y;

	// full search with interpolation
	float min_sad = 0;
	float x = 0;
	float y = 0;
	int patch_num_x = interpolate * region_actual_w - interpolate * blockwf + 1;
	int patch_num_y = interpolate * region_actual_h - interpolate * blockhf + 1;

	for (int j = 0; j < patch_num_y; j++)
	{
		for (int i = 0; i < patch_num_x; i++)
		{
			// calculate SAD
			float sad = SAD_interpolate(block_head, blockwf, blockhf, region_head, region_actual_w, region_actual_h, j, i, interpolate);

			// update min_sad and motion vector
			if (i == 0 && j == 0)
				min_sad = sad;
			else if (sad < min_sad)
			{
				x = (i - relative_x) / 2;
				y = (j - relative_y) / 2;
				min_sad = sad;
			}
		}
	}

	motion_x[block_y_idx][block_x_idx] += x;
	motion_y[block_y_idx][block_x_idx] += y;
}

__global__ void move_pixel(unsigned char* prev_frame, unsigned char* curr_frame, const int blockw, const int blockh, const int width, const int height, int r, float** motion_x, float** motion_y)
{
	// the block that this thread is responsible for
	int block_x_idx = threadIdx.x;
	int block_y_idx = threadIdx.y;
	// destination block pixel index
	int dst_x = block_x_idx * blockw;
	int dst_y = block_y_idx * blockh;

	// get integer motion vector
	float mv_x = motion_x[block_y_idx][block_x_idx];
	float mv_y = motion_y[block_y_idx][block_x_idx];
	// source block pixel index
	float src_x = dst_x + mv_x;
	float src_y = dst_y + mv_y;

	// move pixels
	for (int j = 0; j < blockh; j++)
	{
		for (int i = 0; i < blockw; i++)
		{
			*(curr_frame + 2 * dst_y * width + 2 * dst_x) = (unsigned char)get_interpolated_pixel(prev_frame, r, width, height, src_x, src_y);
		}
	}
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

__device__ float SAD_interpolate(unsigned char* block_header, const int blockwf, const int blockhf, unsigned char* region_header, 
	const int region_wf, const int region_hf, int y_index_f, int x_index_f, int interpolate)
{
	// y_index_f is the y index of interpolated patch
	// x_index_f is the x index of interpolated patch
	float sum = 0;
	for (int j = 0; j < blockhf * interpolate; j++)
	{
		for (int i = 0; i < blockwf * interpolate; i++)
		{
			int num1 = get_interpolated_pixel(block_header, interpolate, blockwf, blockhf, j, i);
			int num2 = get_interpolated_pixel(region_header, interpolate, region_wf, region_hf, y_index_f + j, x_index_f + i);
			float temp = float(num1 - num2);
			sum += temp > 0 ? temp : -temp;
		}
	}
	return sum;
}

__device__ float get_interpolated_pixel(unsigned char* im_header, int r, int width, int height, int y, int x)
{
	float pixel = 0; // interpolated pixel
	// actual (original) pixel index
	int h = int(y / r);
	int w = int(x / r);

	if (x % r == 0 && y % r == 0)
	{
		pixel = (int)*(im_header + 2 * h * width + w * 2);
	}
	else if (x % 2 == 0 && y % 2 != 0)
	{
		pixel += (int)*(im_header + 2 * h * width + w * 2);
		pixel += (h + 1 >= height) ? 0 : (int)*(im_header + 2 * (h+1) * width + w * 2);
		pixel = (int)(pixel * 0.5);
	}
	else if (x % 2 != 0 && y % 2 == 0)
	{
		pixel += (int)*(im_header + 2 * h * width + w * 2);
		pixel += (w + 1 >= width) ? 0 : (int)*(im_header + 2 * h * width + (w+1) * 2);
		pixel = (int)(pixel * 0.5);
	}
	else
	{
		pixel += (int)*(im_header + 2 * h * width + w * 2);
		pixel += (h + 1 >= height) ? 0 : (int)*(im_header + 2 * (h + 1) * width + w * 2);
		pixel += (w + 1 >= width) ? 0 : (int)*(im_header + 2 * h * width + (w + 1) * 2);
		pixel += (h + 1 >= height || w + 1 >= width) ? 0 : (int)*(im_header + 2 * (h+1) * width + (w+1)* 2 );
		pixel = (int)(pixel * 0.25);
	}

	return pixel;
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
	interpolate_patch_match <<<1, threadsPerBlock>>>(in0_gpu, width, height, blockw, blockh, blockwf, blockhf,
		regionwf, regionhf, motion_x, motion_y, interpolate);

	/* Move pixels from prev frame to current frame */


	// explicitly free resources
	cudaDeviceReset();

	return 0;
}

