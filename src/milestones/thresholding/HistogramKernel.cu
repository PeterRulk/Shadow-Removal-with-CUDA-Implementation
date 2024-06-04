//*********************************************************************************
// Filename    : HistogramKernel.cu
// Authors     : From ECE569 Lecture
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Histogram Kernel 
//*********************************************************************************
#include "ProjectDefs.h"
#include "ColorspaceKernelDefs.cuh"

/**
 *   @brief Kernel that calculates the histogram for an image using pixel values 0-255
 *
 *   @param[in]  in1   is a single channel input image
 *   @param[in]  size  is the number of elements for the single channel input image
 *   @param[out] histo is the histogram for the single channel input image with 256 bins
 */
__global__ 
void HistogramKernel( float* in1, float* histo, long size )
{
   __shared__ unsigned int histo_private[256];

   for ( int bin = threadIdx.x; bin < size; bin += blockDim.x )
   {
      if ( bin < 256 )
      {
         histo_private[bin] = 0; // Initialize all bins to zero.
      }
   }
   __syncthreads();

   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = blockDim.x * gridDim.x;
   int pixel;

   while ( i < size )
   {
      pixel = in1[i]; //Get pixel intensity
      atomicAdd( &histo_private[pixel], 1 ); //Atomically increment bin value based on pixel itensity
      i += stride; //Increment using stride amount
   }

   __syncthreads();

   for ( int bin = threadIdx.x; bin < size; bin += blockDim.x )
   {
      if ( bin < 256 )
      {
         atomicAdd( &histo[bin], histo_private[bin] ); //Transfer private histogram to global memory histogram
      }
   }
}