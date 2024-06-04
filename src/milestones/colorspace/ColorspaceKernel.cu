//*********************************************************************************
// Filename    : ColorspaceKernel.cu
// Authors     : Angel Silva
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Colorspace(s) Conversion Kernel 
//*********************************************************************************
#include "ProjectDefs.h"
#include "ColorspaceKernelDefs.cuh"


/**
*   @brief Kernel that performs colorspace transformation to output a YUV, color invariant and greyscale image
*
*   @param[in]  in1 is the input image to perform the colorspace transformation on
*   @param[in]  width is the width of the input image
*   @param[in]  height is the hight of the input image
*   @param[in]  channels is the number of channels for the input image
*   @param[out] uout is the output YUV image
*   @param[out] ciout is the output color invariant image
*   @param[out] gout is the output greyscale image
*
*/

__global__ 
void colorspaceKernel( float* in1, float* uout, float* ciout, float* gout, int width, int height, int channels )
{
   int idx;
   float r;
   float g;
   float b;
   float uc;

   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if ( x < width && y < height )
   {
      idx = y * width + x;

      r = in1[channels * idx];
      g = in1[channels * idx + 1];
      b = in1[channels * idx + 2];

      uc = 0.5f + ( r * -0.14713f - g * 0.28886f + b * 0.436f );
      uout[idx] = uc;

      ciout[channels * idx    ] = atanf(r / fmaxf(g, b));
      ciout[channels * idx + 1] = atanf(g / fmaxf(r, b));
      ciout[channels * idx + 2] = atanf(b / fmaxf(r, g));

      gout[idx] = ( 0.299f * ciout[channels * idx] ) + ( 0.587f * ciout[channels * idx + 1] ) + ( 0.114f * ciout[channels * idx + 2] );
   }
}
