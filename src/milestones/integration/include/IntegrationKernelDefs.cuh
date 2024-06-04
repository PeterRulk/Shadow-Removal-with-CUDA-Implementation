//*********************************************************************************
// Filename    : IntegrationKernelDefs.cuh
// Authors     : Angel Silva
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Header file for Integration Kernels
//*********************************************************************************
#ifndef INCLUDED_INTEGRATION_KERNELDEFS_CUH_
#define INCLUDED_INTEGRATION_KERNELDEFS_CUH_

#include "ProjectDefs.h"


/**
*  Insert Definitions/Function Declarations/Etc 
*/

__global__ void ImageMappingKernel(float* input_a, float* input_b, float* red_sum, float* green_sum, float* blue_sum, int size, int channels);
__global__ void ImageMaskCombinationKernel(float* input, float* sum, int size);
__global__ void ResultImageKernel(float* input, float* smoothmask, float* result, float* ratio_red, float* ratio_green, float* ratio_blue, int imageWidth, int imageHeight);





#endif /* INCLUDED_INTEGRATION_KERNELDEFS_CUH_ */

