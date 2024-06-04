//*********************************************************************************
// Filename    : ColorspaceKernelDefs.cuh
// Authors     : Mason W Edgar
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : 
//*********************************************************************************
#ifndef INCLUDED_COLORSPACE_KERNELDEFS_CUH_
#define INCLUDED_COLORSPACE_KERNELDEFS_CUH_

#include "ProjectDefs.h"


/**
 *  Insert Definitions/Function Declarations/Etc 
 */

__global__ void colorspaceKernel(float* in1, float* uout, float* ciout, float* gout, int width, int height, int channels);















#endif /* INCLUDED_COLORSPACE_KERNELDEFS_CUH_ */

