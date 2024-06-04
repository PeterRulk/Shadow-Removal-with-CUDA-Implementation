//*********************************************************************************
// Filename    : ThresholdingKernelDefs.cuh
// Authors     : Mason W Edgar
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : 
//*********************************************************************************
#ifndef INCLUDED_THRESHOLDING_KERNELDEFS_CUH_
#define INCLUDED_THRESHOLDING_KERNELDEFS_CUH_

#include "ProjectDefs.h"


/**
*  Insert Definitions/Function Declarations/Etc 
*/

__global__ void HistogramKernel(float* in1, float* histo, long size);
__global__ void ImageBinarizationLight(float* image, float* image_binarized, int img_width, int img_height, int threshold);
__global__ void ImageBinarizationShadow(float* image, float* image_binarized, int img_width, int img_height, int threshold);
int calculate_otsu_threshold(float histogram[], int num_bins);













#endif /* INCLUDED_THRESHOLDING_KERNELDEFS_CUH_ */

