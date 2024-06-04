//*********************************************************************************
// Filename    : VarianceKernel.cu
// Authors     : Peter Rulkov
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Histogram Kernel 
//*********************************************************************************

#include "ProjectDefs.h"
#include "ColorspaceKernelDefs.cuh"

__global__ void ImageBinarizationLight(float* image, float* image_binarized, int img_width, int img_height, int threshold) {
    // Calculate global thread index
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


    // Ensure the thread operates within the image bounds
    if (xIndex < img_width && yIndex < img_height) {

        // Calculate the linear index of the current pixel
        int index = yIndex * img_width + xIndex;

        // Retrieve pixel intensity from the input image
        int pixelValue = image[index];

        // Binarize the pixel based on the threshold
        if (pixelValue <= threshold)
        {
            image_binarized[index] = 0; // Background (black)
        }
        else
        {
            image_binarized[index] = 1; // Foreground (white)
        }
    }
}

__global__ void ImageBinarizationShadow(float* image, float* image_binarized, int img_width, int img_height, int threshold) {
    // Calculate global thread index
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


    // Ensure the thread operates within the image bounds
    if (xIndex < img_width && yIndex < img_height) {

        // Calculate the linear index of the current pixel
        int index = yIndex * img_width + xIndex;

        // Retrieve pixel intensity from the input image
        int pixelValue = image[index];

        // Binarize the pixel based on the threshold
        if (pixelValue <= threshold)
        {
            image_binarized[index] = 1; // Background (black)
        }
        else
        {
            image_binarized[index] = 0; // Foreground (white)
        }
    }
}