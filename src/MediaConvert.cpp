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

int motion_search(unsigned char *out, unsigned char *in0, unsigned char *in1, int width, int height)
{
	int i, j, i0, j0;
	unsigned char *op, *sp;
	int *motion_vector = new int[2];
	float *fraction_vector = new float[2];
	

	// search region
	int **region = new int*[R];
	for (int i = 0; i < R; i++)
	{
		region[i] = new int[R];
	}
	// search block
	int **block = new int*[16];
	for (int i = 0; i < 16; i++)
	{
		block[i] = new int[16];
	}

	// search block for 1/2 motion search
	int **blockf = new int*[2];
	for (int i = 0; i < 2; i++)
	{
		blockf[i] = new int[2];
	}
	// search region for 1/2 motion search
	int **temp = new int*[16];
	for (int i = 0; i < 16; i++)
	{
		temp[i] = new int[16];
	}

	//
	for (j = 0; j < height; j += 16)
	{
		for (i = 0; i < width; i += 16)
		{
			// get block data
			sp = in0 + j * width * 2 + i * 2;  // starting point
			for (j0 = 0; j0 < 16; j0++)
				for (i0 = 0; i0 < 16; i0++)
					block[j0][i0] = (int)*(sp + j0 * 2 * width + 2 * i0);
				
			for (j0 = 0; j0 < 2; j0++)
				for (i0 = 0; i0 < 2; i0++)
					blockf[j0][i0] = block[7 + j0][7 + i0];

			// get region data
			sp = in1 + (j - (R-16)/2) * width * 2 + (i - (R-16)/2) * 2;
			for (j0 = 0; j0 < R; j0++)
			{
				for (i0 = 0; i0 < R; i0++)
				{
					if (j - (R-16)/2 < 0 || i - (R-16)/2 < 0 || j - (R-16)/2 + R > height || i - (R - 16) / 2 + R > width)
						region[j0][i0] = 0;
					else
						region[j0][i0] = Clip1((int)*(sp + j0 * 2 * width + 2 * i0));
				}
			}

			// integer block search
			motion_vector = block_search(block, region);
			int dy = motion_vector[0];
			int dx = motion_vector[1];

			// 
			sp = in1 + j * width * 2 + i * 2;  // previous frame
			for (j0 = 0; j0 < 16; j0++)
			{
				for (i0 = 0; i0 < 16; i0++)
				{
					int y_index = j + j0 + dy;
					int x_index = i + i0 + dx;

					if (y_index < 0 || y_index > height || x_index < 0 || x_index > width)
						temp[j0][i0] = 0;
					else
					{
						temp[j0][i0] = Clip1(
							*(sp + (j0 + dy) * 2 * width + 2 * (i0 + dx)));
					}
				}
			}

			// 1/2 block search
			fraction_vector = fraction_block_search(blockf, temp);
			float dyf = fraction_vector[0] + dy;
			float dxf = fraction_vector[1] + dx;

			// set output image
			op = out + j * width * 2 + i * 2;  // output pointer
			for (j0 = 0; j0 < 16; j0++)
				for (i0 = 0; i0 < 16; i0++)
				{
					float y_index = j + j0 + dyf;
					float x_index = i + i0 + dxf;

					if (y_index < 0 || y_index > height || x_index < 0 || x_index > width)
						*(op + 2 * j0 * width + 2 * i0) = *(op + 2 * j0 * width + 2 * i0);
					else
					{
						*(op + 2 * j0 * width + 2 * i0) = (unsigned char) Clip1(getInterpolatePixel(in1, width, height, y_index, x_index));
					}
				}


		}
	}

	return 0;
}

int* block_search(int** block, int** region)
{
	int *motion_vector = new int[2];
	motion_vector[0] = 0;
	motion_vector[1] = 0;

	float min = 0;
	
	int **searchBlock = new int*[16];
	for (int i = 0; i < 16; i++)
	{
		searchBlock[i] = new int[16];
	}


	for (int i = 0; i < R-16+1; i+= 1)
	{
		for (int j = 0; j < R-16+1; j+= 1)
		{
			for (int i0 = 0; i0 < 16; i0++)
			{
				for (int j0 = 0; j0 < 16; j0++)
				{
					searchBlock[i0][j0] = region[i+i0][j+j0];
				}
			}
			
			float sad = SAD(block, searchBlock, 16, 16);
			
			if (i == 0 && j == 0)
				min = sad;
			else if (sad < min)
			{
				motion_vector[0] = i - (R-16)/2;
				motion_vector[1] = j - (R-16)/2;
				min = sad;
			}
		}
	}
	return motion_vector;
}

float* fraction_block_search(int** block, int** searchArea)
{
	float *motion_vector = new float[2];
	motion_vector[0] = 0;
	motion_vector[1] = 0;

	float min = 0;

	// initiate searchBlock and blockf
	// searchBlock: the block to compare
	// blockf: the original block to compare with
	int **searchBlock = new int*[4];
	for (int i = 0; i < 4; i++)
	{
		searchBlock[i] = new int[4];
	}
	int **blockf = new int*[4];
	for (int i = 0; i < 4; i++)
	{
		blockf[i] = new int[4];
	}
	
	// interpolate block
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			blockf[i][j] = getInterpolatePixel(block, 2, 2, float(i) / 2, float(j) / 2);

	// full-search
	for (int i = 0; i < 32 - 4 + 1; i += 2)
	{
		for (int j = 0; j < 32 - 4 + 1; j += 2)
		{
			// interpolate searchBlock
			for (int i0 = 0; i0 < 4; i0++)
				for (int j0 = 0; j0 < 4; j0++)
					searchBlock[i0][j0] = getInterpolatePixel(searchArea, 2, 2, float(i + i0) / 2, float(j + j0) / 2);

			float sad = SAD(blockf, searchBlock, 4, 4);

			if (i == 0 && j == 0)
				min = sad;
			else if (sad < min)
			{
				motion_vector[0] = float(i) / 2 - 7;
				motion_vector[1] = float(j) / 2 - 7;
				min = sad;
			}
		}
	}
	return motion_vector;
}

int getInterpolatePixel(unsigned char* prevFrame, int width, int height, float y_index, float x_index)
{
	int y = int(y_index * 2);
	int x = int(x_index * 2);
	int h = y / 2;
	int w = x / 2;
	int pixel = 0;

	if (x % 2 == 0 && y % 2 == 0)
	{
		pixel = (int)*(prevFrame + 2 * h * width + 2 * w);
	}
	else if (x % 2 == 0 && y % 2 != 0)
	{
		pixel += (int)(prevFrame + 2 * h * width + 2 * w);
		pixel += (int)(prevFrame + 2 * (h + 1) * width + 2 * w);
		pixel = (int)(pixel * 0.5);
	}
	else if (x % 2 != 0 && y % 2 == 0)
	{
		pixel += (int)(prevFrame + 2 * h * width + 2 * w);
		pixel += (int)(prevFrame + 2 * h * width + 2 * (w + 1));
		pixel = (int)(pixel * 0.5);
	}
	else
	{
		pixel += (int)(prevFrame + 2 * h * width + 2 * w);
		pixel += (int)(prevFrame + 2 * (h + 1) * width + 2 * w);
		pixel += (int)(prevFrame + 2 * h * width + 2 * (w + 1));
		pixel += (int)(prevFrame + 2 * (h + 1) * width + 2 * (w + 1));
		pixel = (int)(pixel * 0.25);
	}

	return pixel;
}

int getInterpolatePixel(int** im, int width, int height, float y_index, float x_index)
{
	int y = int(y_index * 2);
	int x = int(x_index * 2);
	int h = y / 2;
	int w = x / 2;
	int pixel = 0;

	if (x % 2 == 0 && y % 2 == 0)
	{
		pixel = im[h][w];
	}
	else if (x % 2 == 0 && y % 2 != 0)
	{
		pixel += im[h][w];
		pixel += (h+1 >= height) ? 0 : im[h+1][w];
		pixel = (int)(pixel * 0.5);
	}
	else if (x % 2 != 0 && y % 2 == 0)
	{
		pixel += im[h][w];
		pixel += (w+1 >= width) ? 0 : im[h][w+1];
		pixel = (int)(pixel * 0.5);
	}
	else
	{
		pixel += im[h][w];
		pixel += (h + 1 >= height) ? 0 : im[h + 1][w];
		pixel += (w + 1 >= width) ? 0 : im[h][w + 1];
		pixel += (h + 1 >= height || w + 1 >= width ) ? 0 : im[h+1][w+1];
		pixel = (int)(pixel * 0.25);
	}

	return pixel;
}

float SAD(int** currentBlock, int** searchBlock, int width, int height)
{
	int i, j;
	float sum = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			float temp = float(currentBlock[i][j] - searchBlock[i][j]);
			sum += temp > 0 ? temp : -temp;
		}
	}
	return sum;
}