//*********************************************************************************
// Filename    : ImageMappingKernel.cu
// Authors     : Angel Silva
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Used to find the summation of the eroded gray shadow mask calculation
// that will be used to calculate the average channel value in shadow and light areas.
//*********************************************************************************
#include "ProjectDefs.h"
#include "IntegrationKernelDefs.cuh"

/**
 *   @brief Calculates the numerator used for finding the average channel values
 *
 *   @param[in]  input_a   Original input image
 *   @param[in]  input_b   Eroded grayscale shadow mask
 *   @param[in]  size      Number of elements of eroded grayscale shadow mask
 *   @param[in]  channels  Number of channels for the original input image
 *   @param[out] red_sum   Total sum for the red channel values multiplied with the respective eroded grayscale value
 *   @param[out] green_sum Total sum for the green channel values multiplied with the respective eroded grayscale value
 *   @param[out] blue_sum  Total sum for the blue channel values multiplied with the respective eroded grayscale value
 */
__global__ 
void ImageMappingKernel( float* input_a, float* input_b, float* red_sum, float* green_sum, float* blue_sum, int size, int channels )
{
    __shared__ float red_sum_private[1];
    __shared__ float green_sum_private[1];
    __shared__ float blue_sum_private[1];

    if ( threadIdx.x == 0 )
    {
        red_sum_private[0] = 0;    // Only first thread in block initializes to zero.
        green_sum_private[0] = 0;    // Only first thread in block initializes to zero.
        blue_sum_private[0] = 0;    // Only first thread in block initializes to zero.
    }
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while ( i < size )
    { 
        // Only loop for valid pixels
        float red_value   = input_a[i * channels]     * input_b[i];    // Perform element-wise multiplication on red channel values.
        float green_value = input_a[i * channels + 1] * input_b[i];    // Perform element-wise multiplication on green channel values.
        float blue_value  = input_a[i * channels + 2] * input_b[i];    // Perform element-wise multiplication on blue channel values.

        atomicAdd( &red_sum_private[0], red_value);     // Atomically add red channel multiplication result to private red sum variable.
        atomicAdd( &green_sum_private[0], green_value); // Atomically add green channel multiplication result to private green sum variable.
        atomicAdd( &blue_sum_private[0], blue_value);   // Atomically add blue channel multiplication result to private blue sum variable.
        i += stride; // Increment by stride amount
    }
    __syncthreads();

    if ( threadIdx.x == 0 )
    { 
        // Transfer sums to global memory
        atomicAdd( &red_sum[0], red_sum_private[0]);
        atomicAdd( &green_sum[0], green_sum_private[0]);
        atomicAdd( &blue_sum[0], blue_sum_private[0]);
    }
}