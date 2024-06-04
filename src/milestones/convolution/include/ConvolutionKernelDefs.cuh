//*********************************************************************************
// Filename    : ConvolutionKernelDefs.cuh
// Course      : ECE 569 
// Semester    : Spring 2024
// Group       : Group 7
// Authors     : Mason Edgar, Angel Silva, Peter Rulkov
// Description : 
//*********************************************************************************
#ifndef INCLUDED_CONVOLUTION_KERNELDEFS_CUH_
#define INCLUDED_CONVOLUTION_KERNELDEFS_CUH_

// Project Includes 
#include "ProjectDefs.h"

#define MASK_WIDTH 5
#define MASK_OFFSET ( MASK_WIDTH >> 1 )


__global__ void ErosionKernel(const float* __restrict__ srcImg, 
                              float*       __restrict__ dstImg, 
                              const float* __restrict__ mask, 
                              int                       width, 
                              int                       height);


__global__ void ConvolutionKernel(const float* __restrict__ in, 
                                  const float* __restrict__ mask, 
                                  float*       __restrict__ out, 
                                  int                       width, 
                                  int                       height);


#endif /* INCLUDED_CONVOLUTION_KERNELDEFS_CUH_ */
