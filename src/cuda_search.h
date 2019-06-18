#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

__device__ float SAD(unsigned char* patch_header, unsigned char* block_header, int blockw, int blockh, int width, int height);

__device__ float SAD_interpolate(unsigned char* block_header, const int blockwf, const int blockhf, unsigned char* region_header,
	const int region_w, const int region_h, int y_index_f, int x_index_f, int interpolate);

__device__ float get_interpolated_pixel(unsigned char* im_header, int intp_rate, int width, int height, int yIdx, int xIdx);

int cuda_full_search(unsigned char *out_gpu, unsigned char *in0_gpu, unsigned char *in1_gpu, int width, int height);