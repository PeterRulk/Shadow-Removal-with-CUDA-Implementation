//**************************************************************************************
// Filename    : ResultImageKernel.cu
// Authors     : Angel Silva
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Used to generate the final output image with shadows removed.
//**************************************************************************************
#include "ProjectDefs.h"
#include "IntegrationKernelDefs.cuh"

/**
 *   @brief Kernel that generates the final output image with shadows removed
 *
 *   @param[in]  input       is original input image
 *   @param[in]  smoothmask  is the convoluted image from the YUV mask image
 *   @param[in]  ratio_red   is the ratio of light to shadow for red channel
 *   @param[in]  ratio_green is the ratio of light to shadow for green channel
 *   @param[in]  ratio_blue  is the ratio of light to shadow for blue channel
 *   @param[in]  imageWidth  is the width of the original input image
 *   @param[in]  imageHeight is the height of the original input image
 *   @param[out] result      is the output image with shadows removed
 */
__global__
void ResultImageKernel(float* input, float* smoothmask, float* result, float* ratio_red, float* ratio_green, float* ratio_blue, int imageWidth, int imageHeight)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const int channels = 3;

	if (x < imageWidth && y < imageHeight) 
	{
		int idx = y * imageWidth + x;

		// idx = ( ( blockIdx.y * blockDim.y ) + threadIdx.y ) * imgWidth + (blockIdx.x * blockDim.x) + threadIdx.x; 



		result[channels * idx]		= (ratio_red[0]   + 1) / ((1 - smoothmask[idx]) * ratio_red[0]   + 1) * input[channels * idx];
		result[channels * idx + 1] = (ratio_green[0] + 1) / ((1 - smoothmask[idx]) * ratio_green[0] + 1) * input[channels * idx + 1];
		result[channels * idx + 2] = (ratio_blue[0]  + 1) / ((1 - smoothmask[idx]) * ratio_blue[0]  + 1) * input[channels * idx + 2];
	}

}
