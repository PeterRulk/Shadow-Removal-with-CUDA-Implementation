//******************************************************************************************
// Filename    : ConvolutionKernel.cu
// Authors     : Angel Silva, Mason Edgar
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Convolution Kernel 
// Credit      : Source code based on convolution code from ECE569 Lecture
//******************************************************************************************
#include "ProjectDefs.h"

/**
 *   @brief Function that performs convolution on an input image
 *
 *   @param[in]  in          input image to perform the convolution on
 *   @param[in]  mask        input kernel mask used to perform convolution
 *   @param[in]  maskwidth   width of the kernel matrix mask
 *   @param[in]  width       width of the input image
 *   @param[in]  height      height of the input image
 *   @param[out] out         convolution output image
 */
__global__ 
void ConvolutionKernel(const float* __restrict__ in, const float* __restrict__ mask, float* __restrict__ out, int width, int height)
{
    int col = threadIdx.x + ( blockIdx.x * blockDim.x );
    int row = threadIdx.y + ( blockIdx.y * blockDim.y );

    if (col < width && row < height)
    {
        float pixVal  = 0;
        int start_col = (col - MASK_OFFSET);
        int start_row = (row - MASK_OFFSET);
     
     #pragma unroll 
        for (int j = 0; j < MASK_WIDTH; ++j) 
        {

        #pragma unroll 
            for (int k = 0; k < MASK_WIDTH; ++k) 
            {
                int curRow = start_row + j;
                int curCol = start_col + k;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    pixVal += in[curRow * width + curCol] * mask[j * MASK_WIDTH + k];
                }
            }
        }

        out[row * width + col] = pixVal;
    }
}
