/*
	Copyright (C) 2015, Liang Fan
	All rights reverved.
*/

/** \file		MediaConvert.h
    \brief		Header file of media convert functions.
*/

#ifndef __MEDIACONVERT_H__
#define __MEDIACONVERT_H__

int yuv420_to_rgb24(unsigned char *out, unsigned char *in, int width, int height);
int yuv422_to_rgb24(unsigned char *out, unsigned char *in, int width, int height);

int Compress(unsigned char *out, unsigned char *in, int width, int height, int compressX);
void DCT(int ** input, int ** output, int row, int col);
void IDCT(int ** input, int ** output, int row, int col);

int Difference(unsigned char *out, unsigned char *in0, unsigned char *in1, int width, int height);

int yuv422_y_to_rgb24(unsigned char *out, unsigned char *in0, int width, int height);

int motion_search(unsigned char *out, unsigned char *in0, unsigned char *in1, int width, int height);

int* block_search(int** block, int** region);
float* fraction_block_search(int** blockf, int** searchArea);

int getInterpolatePixel(unsigned char* previousFrame, int width, int height, float y_index, float x_index);
int getInterpolatePixel(int** im, int width, int height, float y_index, float x_index);

float SAD(int** currentBlock, int** searchBlock, int width, int height);

const double PI = 3.14159265358979;
const int Zig_Zag[8][8] = {
	{ 0,1,5,6,14,15,27,28 },
	{ 2,4,7,13,16,26,29,42 },
	{ 3,8,12,17,25,30,41,43 },
	{ 9,11,18,24,31,40,44,53 },
	{ 10,19,23,32,39,45,52,54 },
	{ 20,22,33,38,46,51,55,60 },
	{ 21,34,37,47,50,56,59,61 },
	{ 35,36,48,49,57,58,62,63 }
};

#endif // __MEDIACONVERT_H__
