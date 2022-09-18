#pragma once

#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "config.h"

#include <algorithm>

/* get the dark channel.*/
__global__ void darkChannel(unsigned char* orig, unsigned char* dark, unsigned width, unsigned height);

/* calculate a sub histogram of one block.
each thread calculates each column.*/
__global__ void subHist(unsigned char* dark, unsigned* subHist, unsigned width, unsigned height);
__global__ void subHist_2(unsigned char* dark, unsigned* subHist, unsigned width, unsigned height);

/* sum the sub histograms*/
__global__ void sumHist(unsigned* subHist, unsigned* hist, unsigned subHistSize);

/* get the AcRow for each column.*/
__global__ void getAcRow(unsigned char* orig, unsigned char* Idark, unsigned char* AcRow, unsigned char colorThresh, unsigned width, unsigned height);

/* get the Ac.*/
__global__ void getAc(unsigned char* AcRow, unsigned char* Ac, unsigned width);

/* get the t_tilde.*/
__global__ void getttilde(unsigned char* orig, float* ttilde, unsigned char* Ac, unsigned width, unsigned height);

/* input initial width and height.*/
__global__ void ScaleAndGray(unsigned char* orig, unsigned char* gray, unsigned width, unsigned height, int scaleFactor);

/* calculate ak and bk for each window wk.*/
__global__ void linearPara(float* filteringP, unsigned char* guidedI, float* ab, int width, int height, float epsilon);

/* do filtering for each qi*/
__global__ void doFiltering_new(float* ab, unsigned char* guidedI, float* outputQ, int width, int height);

///* do filtering for each ak and bk.
//(This function is bad. Maybe we shouldn't change multiple pixels in one thread?)*/
//__global__ void doFiltering(float* ab, unsigned* guidedI, float* outputQ, int width, int height);
//
///* image normalize and show result.*/
//__global__ void imNormalizing(float* outputQ, int width, int height);




