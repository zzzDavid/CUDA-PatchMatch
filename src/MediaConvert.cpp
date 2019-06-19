/*
	Copyright (C) 2015, Liang Fan
	All rights reverved.
*/

/** \file		MediaConvert.cpp
    \brief		Implements convert functions.
*/

#include <string.h>
#include <math.h>
#include "MediaConvert.h"
#include <iostream>
#include "cuda_search.h"
#include "cuda_runtime.h"

using namespace std;

//
#define Clip3(a, b, c) ((c < a) ? a : ((c > b) ? b : c))
#define Clip1(x)       Clip3(0, 255, x)
#define R 48

//
int yuv422_y_to_rgb24(unsigned char *out, unsigned char *in0, int width, int height)
{
	int i, j;
	int r0, g0, b0;
	int y0;
	unsigned char *ip0, *op;

	// initialize buf pointer
	// note: output RGB24 is bottom line first.
	op = out + width * (height - 1) * 3;
	ip0 = in0;

	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			// get YUV data
			y0 = (int)*(ip0);

			// convert YUV to RGB
			r0 = 255 - y0;
			g0 = 255 - y0;
			b0 = 255 - y0;

			// RGB24
			// unsigned char: B0, G0, R0, B1, G1, R1, ...
			// note: RGB24 is bottom line first.
			// first pixel line
			*op = (unsigned char)Clip1(b0);
			*(op + 1) = (unsigned char)Clip1(g0);
			*(op + 2) = (unsigned char)Clip1(r0);

			//
			op += 3;
			ip0 += 2;
		}
		op -= width * 3 * 2;
	}

	return 0;
}



int full_search(unsigned char *out, unsigned char *in0, unsigned char *in1, int width, int height)
{
	// Allocate vectors in device memory
	size_t size = width * height * 2 * sizeof(unsigned char);
	unsigned char* out_gpu;
	cudaMalloc(&out_gpu, size);
	unsigned char* in0_gpu;
	cudaMalloc(&in0_gpu, size);
	unsigned char* in1_gpu;
	cudaMalloc(&in1_gpu, size);

	// Copy data from host memory to device memory
	cudaMemcpy(in0_gpu, in0, size, cudaMemcpyHostToDevice);
	cudaMemcpy(in1_gpu, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(out_gpu, out, size, cudaMemcpyHostToDevice);

	cuda_full_search(out_gpu, in0_gpu, in1_gpu, width, height);

	// Copy data from device memory to host memory
	cudaMemcpy(out, out_gpu, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(out_gpu);
	cudaFree(in0_gpu);
	cudaFree(in1_gpu);

	return 0;
}