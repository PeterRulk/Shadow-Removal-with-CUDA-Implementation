//**************************************************************************************
// Filename    : ImageMaskCombinationKernel.cu
// Authors     : Angel Silva
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Used to find the summation of the eroded gray shadow mask calculation
// that will be used to calculate the average channel value in shadow and light areas.
//**************************************************************************************
#include "ProjectDefs.h"
#include "IntegrationKernelDefs.cuh"

/**
 *   @brief Kernel that calculates the summation of the eroded grayscale shadowmask image
 *
 *   @param[in]  input is the eroded grayscale shadowmask image
 *   @param[in]  size is the number of elements of eroded grayscale shadow mask
 *   @param[out] sum is the total sum of pixel values for the eroded grayscale shadowmask image
 *
 */

__global__ void ImageMaskCombinationKernel(float* input, float* sum, int size)
{
    __shared__ float sum_private[1];
    if (threadIdx.x == 0) {
        sum_private[0] = 0; // Only first thread in block initializes to zero.
    }
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        float value = input[i]; //Get pixel value from input image
        atomicAdd(&sum_private[0], value); //Atomically add value to private sum
        i += stride;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&sum[0], sum_private[0]); //Atomically add private sum to global memory output
    }
}