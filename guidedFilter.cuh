#pragma once

#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "config.h"

#include <algorithm>


/* input initial width and height.*/
__global__ void ScaleAndGray(unsigned char* orig, unsigned char* gray, unsigned width, unsigned height, int scaleFactor);

/* calculate ak and bk for each window wk.*/
__global__ void linearPara(unsigned char* filteringP, unsigned char* guidedI, float* ab, int width, int height, float epsilon);

/* do filtering for each ak and bk.
(This function is bad. Maybe we shouldn't change multiple pixels in one thread?)*/
__global__ void doFiltering(float* ab, unsigned* guidedI, float* outputQ, int width, int height);

/* do filtering for each qi*/
__global__ void doFiltering_new(float* ab, unsigned char* guidedI, float* outputQ, int width, int height);

/* image normalize and show result.*/
__global__ void imNormalizing(float* outputQ, int width, int height);
