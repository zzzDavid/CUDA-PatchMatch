#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

__device__ float SAD(unsigned char* patch_header, unsigned char* block_header, int blockw, int blockh, int width, int height);

