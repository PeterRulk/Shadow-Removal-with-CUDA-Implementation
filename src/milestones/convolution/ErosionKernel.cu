//*********************************************************************************
// Filename    : ErosionKernel.cu
// Course      : ECE 569 
// Semester    : Spring 2024
// Group       : Group 7
// Authors     : Mason Edgar
// Description : Erosion Kernel Implementation  
//*********************************************************************************
#include "ProjectDefs.h"
#include "ConvolutionKernelDefs.cuh"


/**
 *  @brief Performs an efficient erosion operation on the source image. 
 * 
 *  @param[in]  srcImg  Pointer to source image buffer 
 *  @param[out] dstImg  Pointer to destination image buffer 
 *  @param[in]  mask    Structural element 
 *  @param[in]  width   Source image width in pixels 
 *  @param[in]  height  Source image height in pixels 
 */
__global__ 
void ErosionKernel(const float* __restrict__ srcImg, float* __restrict__ dstImg, const float* __restrict__ mask, int width, int height)
{
	int col = threadIdx.x + ( blockIdx.x * blockDim.x );
	int row = threadIdx.y + ( blockIdx.y * blockDim.y );

	if (col < width && row < height)
	{
		uint32_t pixVal = 1U;

		int startCol = ( col - MASK_OFFSET );
		int startRow = ( row - MASK_OFFSET );

		int offset = (MASK_WIDTH - 1U);

		int currRow = ( startRow + offset );
		int currCol = ( startCol + offset );

		if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width)
		{
			pixVal = static_cast<uint32_t>(srcImg[currRow * width + currCol]) & static_cast<uint32_t>(mask[(offset * MASK_WIDTH) + offset]); 
		}

		dstImg[row * width + col] = static_cast<float>(pixVal);
	}
}
