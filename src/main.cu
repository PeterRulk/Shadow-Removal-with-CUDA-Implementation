//*********************************************************************************
// Filename    : main.cu
// Course      : ECE 569 
// Semester    : Spring 2024
// Group       : Group 7
// Authors     : Mason Edgar, Angel Silva, Peter Rulkov
// Description : CUDA Implementation of Shadow Removal Algorithm 
//*********************************************************************************
#include "ProjectDefs.h"

/**
 *  Main entry point.
 */
int main()
{
   char*     inputImageFile = "plt.ppm";
   int       imageChannels;
   int       imageWidth;
   int       imageHeight;
   wbImage_t inputImage;
   wbImage_t outputImage1;
   wbImage_t outputImage2;
   wbImage_t outputImage3;
   wbImage_t ciImage;
   wbImage_t gImage;
   float*    hostInputImageData;
   float*    hostOutputImageData1;
   float*    hostOutputImageData2;
   float*    hostOutputImageData3;
   float*    deviceInputImageData;
   float*    deviceOutputImageData1;
   float*    deviceOutputImageData2;
   float*    deviceOutputImageData3;
   float*    deviceGrayHisto;
   float*    deviceYUVHisto;
   float*    hostGrayHisto;
   float*    hostYUVHisto;
   float*    deviceGrayHistImage;
   float*    deviceYUVHistImage;
   int       num_bins = 256;

   //=================================================================================
   //! @defgroup Colorspace
   //=================================================================================
   inputImage    = wbImport(inputImageFile);
   imageWidth    = wbImage_getWidth(inputImage);
   imageHeight   = wbImage_getHeight(inputImage);
   imageChannels = wbImage_getChannels(inputImage);

   // Since the image is YUV, it contains 3 channels
   outputImage1 = wbImage_new(imageWidth, imageHeight, 1);
   outputImage2 = wbImage_new(imageWidth, imageHeight, imageChannels);
   outputImage3 = wbImage_new(imageWidth, imageHeight, 1);

   hostInputImageData   = wbImage_getData(inputImage);
   hostOutputImageData1 = wbImage_getData(outputImage1);
   hostOutputImageData2 = wbImage_getData(outputImage2);
   hostOutputImageData3 = wbImage_getData(outputImage3);

   hostGrayHisto = reinterpret_cast<float*>(malloc(256 * sizeof(float)));
   hostYUVHisto  = reinterpret_cast<float*>(malloc(256 * sizeof(float)));

   for (int i = 0; i < 255; i++)
   {
      hostYUVHisto[i] = 0;
   }

   float* hostMask = (float*)malloc(25 * sizeof(float));
   for (int j = 0; j < 25; j++) {
      hostMask[j] = 1;
   }

   float* hostKernelMask = (float*)malloc(25 * sizeof(float));
   for (int j = 0; j < 25; j++) {
      hostKernelMask[j] = 0.04;
   }

   auto imgSize = imageWidth * imageHeight * sizeof(float);

   cudaMalloc(reinterpret_cast<void**>(&deviceInputImageData),   imgSize * imageChannels);
   cudaMalloc(reinterpret_cast<void**>(&deviceOutputImageData1), imgSize);
   cudaMalloc(reinterpret_cast<void**>(&deviceOutputImageData2), imgSize * imageChannels);
   cudaMalloc(reinterpret_cast<void**>(&deviceOutputImageData3), imgSize);

   cudaMemcpy(deviceInputImageData, hostInputImageData, (imgSize * imageChannels), cudaMemcpyHostToDevice);

   dim3 DimGrid((int)ceil((imageWidth - 1) / 16 + 1), (int)ceil((imageHeight - 1) / 16 + 1), 1);
   dim3 DimBlock(16, 16, 1);

   /**
   *  @fn Colorspace Kernel Invocation
   */
   auto start = std::chrono::high_resolution_clock::now();

   colorspaceKernel<<<DimGrid, DimBlock>>>(deviceInputImageData,
                                           deviceOutputImageData1,
                                           deviceOutputImageData2,
                                           deviceOutputImageData3,
                                           imageWidth,
                                           imageHeight,
                                           imageChannels);

   auto stop = std::chrono::high_resolution_clock::now();

   cudaMemcpy(hostOutputImageData1, deviceOutputImageData1, imgSize,                 cudaMemcpyDeviceToHost);
   cudaMemcpy(hostOutputImageData2, deviceOutputImageData2, imgSize * imageChannels, cudaMemcpyDeviceToHost);
   cudaMemcpy(hostOutputImageData3, deviceOutputImageData3, imgSize,                 cudaMemcpyDeviceToHost);

   auto duration = std::chrono::duration<double, std::micro>(stop - start).count();
   std::cout << "Colorspace Execution Time: " << std::fixed << std::setprecision(3) << duration << " [us] " <<  std::endl;

   //=================================================================================
   //! @defgroup Histogram  
   //=================================================================================
   //   yuvImage = wbImage_new( imageWidth, imageHeight, imageChannels, hostOutputImageData1 );
   ciImage = wbImage_new(imageWidth, imageHeight, imageChannels, hostOutputImageData2);
   gImage  = wbImage_new(imageWidth, imageHeight, 1, hostOutputImageData3);

   //const char* outfilename1 = "YUVoutput.ppm";
   const char* outfilename2 = "CIoutput.ppm";
   const char* outfilename3 = "Greyoutput.ppm";

   //wbExport( outfilename1, yuvImage );
   wbExport(outfilename2, ciImage);
   wbExport(outfilename3, gImage);

   size_t imgDim = (static_cast<size_t>(imageWidth) * imageHeight);

   for (auto j = 0; j < imgDim; j++)
   {
      hostOutputImageData1[j] = round(255 * hostOutputImageData1[j]);
      hostOutputImageData3[j] = round(255 * hostOutputImageData3[j]);
   }

   dim3 blockDim(512), gridDim(30);

   // Compute Gray Histogram
   start = std::chrono::high_resolution_clock::now();

   deviceGrayHistImage = reinterpret_cast<float*>(malloc(imgDim * sizeof(float)));
   cudaMalloc(reinterpret_cast<void**>(&deviceGrayHistImage), imgDim * sizeof(float));

   // Copy Grey image to GPU
   cudaMemcpy(deviceGrayHistImage, hostOutputImageData3, (imgDim * sizeof(float)), cudaMemcpyHostToDevice);

   // Allocate memory to GPU for histogram
   cudaMalloc(reinterpret_cast<void**>(&deviceGrayHisto), 256 * sizeof(float));

   /**
   *  @fn Histogram Kernel Invocation (1/2)
   */
   HistogramKernel<<<gridDim, blockDim>>>(deviceGrayHistImage, deviceGrayHisto, imgDim);

   // Copy Histo from GPU to Host
   cudaMemcpy(hostGrayHisto, deviceGrayHisto, (256 * sizeof(float)), cudaMemcpyDeviceToHost);

   deviceYUVHistImage = reinterpret_cast<float*>(malloc(imgDim * sizeof(float)));
   cudaMalloc(reinterpret_cast<void**>(&deviceYUVHistImage), imgDim * sizeof(float));

   // Copy YUV image to GPU
   cudaMemcpy(deviceYUVHistImage, hostOutputImageData1, (imgDim * sizeof(float)), cudaMemcpyHostToDevice);

   // Allocate memory to GPU for histogram
   cudaMalloc(reinterpret_cast<void**>(&deviceYUVHisto), 256 * sizeof(float));

   /**
   *  @fn Histogram Kernel Invocation (1/2)
   */
   HistogramKernel<<<gridDim, blockDim>>>(deviceYUVHistImage, deviceYUVHisto, imgDim);

   // Copy Histo from GPU to Host
   cudaMemcpy(hostYUVHisto, deviceYUVHisto, (256 * sizeof(float)), cudaMemcpyDeviceToHost);

   int gray_threshold = calculate_otsu_threshold(hostGrayHisto, num_bins);
   int yuv_threshold  = calculate_otsu_threshold(hostYUVHisto,  num_bins);

   //printf("Otsu's Threshold Grey: %d\n", gray_threshold);
   //printf("Otsu's Threshold YUV: %d\n", yuv_threshold);

   //=================================================================================
   //! @defgroup Image_Binarization 
   //=================================================================================
   float* light_binarized_image_grey;
   cudaMalloc(reinterpret_cast<void**>(&light_binarized_image_grey), imgSize);

   /**
   *  @fn Binarization Kernel Invocation for Grey Light Mask
   */
   ImageBinarizationLight<<<DimGrid, DimBlock>>>(deviceGrayHistImage, light_binarized_image_grey, imageWidth, imageHeight, gray_threshold);

   // Allocate memory for output image on host
   float* light_binarized_image_out_grey = new float[imageWidth * imageHeight];

   // Copy the binarized image from device to host
   cudaMemcpy(light_binarized_image_out_grey, light_binarized_image_grey, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

   float* shadow_binarized_image_grey;

   cudaMalloc(reinterpret_cast<void**>(&shadow_binarized_image_grey), imgSize);

   /**
   *  @fn Binarization Kernel Invocation for Grey Shadow Mask
   */
   ImageBinarizationShadow<<<DimGrid, DimBlock>>>(deviceGrayHistImage, shadow_binarized_image_grey, imageWidth, imageHeight, gray_threshold);

   // Allocate memory for output image on host
   float* shadow_binarized_image_out_grey = new float[imageWidth * imageHeight];

   // Copy the binarized image from device to host
   cudaMemcpy(shadow_binarized_image_out_grey, shadow_binarized_image_grey, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

   float* binarized_image_YUV;
   cudaMalloc(reinterpret_cast<void**>(&binarized_image_YUV), imgSize);

   /**
   *  @fn Binarization Kernel Invocation for YUV
   */
   ImageBinarizationLight<<<DimGrid, DimBlock>>>(deviceYUVHistImage, binarized_image_YUV, imageWidth, imageHeight, yuv_threshold);

   stop = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration<double, std::micro>(stop - start).count();
   std::cout << "Thresholding Execution Time: " << std::fixed << std::setprecision(3) << duration << " [us] " <<  std::endl;

   // Allocate memory for output image on host
   float* binarized_image_out_YUV = new float[imageWidth * imageHeight];

   // Copy the binarized image from device to host
   cudaMemcpy(binarized_image_out_YUV, binarized_image_YUV, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

   //=================================================================================
   //! @defgroup Convolution 
   //=================================================================================
   int    maskwidth = 5;
   float* hostSmoothMask = reinterpret_cast<float*>(malloc(imgSize));
   float* deviceKernelMask;
   float* deviceConvMask;

   cudaMalloc(reinterpret_cast<void**>(&deviceKernelMask), maskwidth * maskwidth * sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceConvMask), imgSize);
   cudaMemcpy(deviceKernelMask, hostKernelMask, maskwidth * maskwidth * sizeof(float), cudaMemcpyHostToDevice);

   start = std::chrono::high_resolution_clock::now();
   /**
   *  @fn Convolution Kernel Invocation 
   */
   ConvolutionKernel<<<DimGrid, DimBlock>>>(binarized_image_YUV, deviceKernelMask, deviceConvMask, imageWidth, imageHeight);

   stop = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration<double, std::micro>(stop - start).count();
   std::cout << "Convolution Execution Time: " << std::fixed << std::setprecision(3) << duration << " [us] " <<  std::endl;

   cudaMemcpy(hostSmoothMask, deviceConvMask, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

   float* deviceShadowMask;
   float* deviceLightMask;
   float* hostShadowMask = reinterpret_cast<float*>(malloc(imgSize));
   float* hostLightMask  = reinterpret_cast<float*>(malloc(imgSize));
   float* deviceMask;

   cudaMalloc(reinterpret_cast<void**>(&deviceShadowMask), imgSize);
   cudaMalloc(reinterpret_cast<void**>(&deviceLightMask),  imgSize);
   cudaMalloc(reinterpret_cast<void**>(&deviceMask), maskwidth * maskwidth * sizeof(float));
   cudaMemcpy(deviceMask, hostMask, maskwidth * maskwidth * sizeof(float), cudaMemcpyHostToDevice);

   start = std::chrono::high_resolution_clock::now();
   /**
   *  @fn Erosion Kernel Invocation for Light Mask
   */
   ErosionKernel<<<DimGrid, DimBlock>>>(light_binarized_image_grey, deviceLightMask, deviceMask, imageWidth, imageHeight);

   stop = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration<double, std::micro>(stop - start).count();
   std::cout << "Erosion Execution Time: " << std::fixed << std::setprecision(3) << duration << " [us] " <<  std::endl;

   cudaMemcpy(hostLightMask, deviceLightMask, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

   /**
   *  @fn Erosion Kernel Invocation for Shadow Mask
   */
   ErosionKernel<<<DimGrid, DimBlock>>>(shadow_binarized_image_grey, deviceShadowMask, deviceMask, imageWidth, imageHeight);

   cudaMemcpy(hostShadowMask, deviceShadowMask, imageWidth* imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

   //=================================================================================
   //! @defgroup Image_Mapping 
   //=================================================================================
   float* deviceShadowRedSum;
   float* deviceShadowGreenSum;
   float* deviceShadowBlueSum;
   float* deviceLightRedSum;
   float* deviceLightGreenSum;
   float* deviceLightBlueSum;
   float* deviceShadowSum;
   float* deviceLightSum;
   float* hostShadowRedSum   = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostShadowGreenSum = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostShadowBlueSum  = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostLightRedSum    = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostLightGreenSum  = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostLightBlueSum   = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostShadowSum      = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* hostLightSum       = reinterpret_cast<float*>(malloc(sizeof(float)));

   cudaMalloc(reinterpret_cast<void**>(&deviceShadowRedSum),   sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceShadowGreenSum), sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceShadowBlueSum),  sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceLightRedSum),    sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceLightGreenSum),  sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceLightBlueSum),   sizeof(float));

   dim3 IntBlockDim(512), IntGridDim(30);

   start = std::chrono::high_resolution_clock::now();
   /**
   *  @fn Image Mapping Kernel Invocation for Shadow Mask
   */
   ImageMappingKernel<<<IntGridDim, IntBlockDim>>>(deviceInputImageData, 
                                                   deviceShadowMask, 
                                                   deviceShadowRedSum, 
                                                   deviceShadowGreenSum, 
                                                   deviceShadowBlueSum, 
                                                   imgDim, 3);

   cudaMemcpy(hostShadowRedSum,   deviceShadowRedSum,   sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(hostShadowGreenSum, deviceShadowGreenSum, sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(hostShadowBlueSum,  deviceShadowBlueSum,  sizeof(float), cudaMemcpyDeviceToHost);

   /**
   *  @fn Image Mapping Kernel Invocation for Light Mask
   */
   ImageMappingKernel<<<IntGridDim, IntBlockDim>>>(deviceInputImageData, 
                                                   deviceLightMask, 
                                                   deviceLightRedSum, 
                                                   deviceLightGreenSum, 
                                                   deviceLightBlueSum, 
                                                   imgDim, 3);

   cudaMemcpy(hostLightRedSum,   deviceLightRedSum,   sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(hostLightGreenSum, deviceLightGreenSum, sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(hostLightBlueSum,  deviceLightBlueSum,  sizeof(float), cudaMemcpyDeviceToHost);

   cudaMalloc(reinterpret_cast<void**>(&deviceShadowSum), sizeof(float));

   /**
   *  @fn Shadow Mask Summation Kernel Invocation
   */
   ImageMaskCombinationKernel<<<IntGridDim, IntBlockDim>>>(deviceShadowMask, deviceShadowSum, imgDim);

   cudaMemcpy(hostShadowSum, deviceShadowSum, sizeof(float), cudaMemcpyDeviceToHost);
   cudaMalloc(reinterpret_cast<void**>(&deviceLightSum), sizeof(float));

   /**
   *  @fn Light Mask Summation Kernel Invocation
   */
   ImageMaskCombinationKernel<<<IntGridDim, IntBlockDim>>>(deviceLightMask, deviceLightSum, imgDim);

   cudaMemcpy(hostLightSum, deviceLightSum, sizeof(float), cudaMemcpyDeviceToHost);

   //=================================================================================
   //! @defgroup Result_Integration 
   //=================================================================================
   // Calculate Ratios
   float shadowavg_red   = (   hostShadowRedSum[0] / hostShadowSum[0] );
   float shadowavg_green = ( hostShadowGreenSum[0] / hostShadowSum[0] );
   float shadowavg_blue  = (  hostShadowBlueSum[0] / hostShadowSum[0] );
   float lightavg_red    = (    hostLightRedSum[0] /  hostLightSum[0] );
   float lightavg_green  = (  hostLightGreenSum[0] /  hostLightSum[0] );
   float lightavg_blue   = (   hostLightBlueSum[0] /  hostLightSum[0] );

   float* ratio_red      = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* ratio_green    = reinterpret_cast<float*>(malloc(sizeof(float)));
   float* ratio_blue     = reinterpret_cast<float*>(malloc(sizeof(float)));

   ratio_red[0]   = (  lightavg_red / shadowavg_red  ) - 1;
   ratio_green[0] = (lightavg_green / shadowavg_green) - 1;
   ratio_blue[0]  = ( lightavg_blue / shadowavg_blue ) - 1;

   float* device_ratio_red;
   float* device_ratio_green;
   float* device_ratio_blue;

   cudaMalloc(reinterpret_cast<void**>(&device_ratio_red),   sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&device_ratio_green), sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&device_ratio_blue),  sizeof(float));

   cudaMemcpy(device_ratio_red,   ratio_red,   sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(device_ratio_green, ratio_green, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(device_ratio_blue,  ratio_blue,  sizeof(float), cudaMemcpyHostToDevice);

   float* deviceResult;
   float* deviceSmoothMask;

   cudaMalloc(reinterpret_cast<void**>(&deviceResult),     imageWidth * imageHeight * imageChannels * sizeof(float));
   cudaMalloc(reinterpret_cast<void**>(&deviceSmoothMask), imageWidth * imageHeight * sizeof(float));

   cudaMemcpy(deviceSmoothMask, hostSmoothMask, (imgDim * sizeof(float)), cudaMemcpyHostToDevice);

   /**
   *  @fn Result Integration Kernel Invocation
   */
   ResultImageKernel<<<DimGrid, DimBlock>>>(deviceInputImageData, 
                                             deviceSmoothMask, 
                                             deviceResult, 
                                             device_ratio_red, 
                                             device_ratio_green, 
                                             device_ratio_blue, 
                                             imageWidth, 
                                             imageHeight);

   stop = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration<double, std::micro>(stop - start).count();
   std::cout << "Integration Execution Time: " << std::fixed << std::setprecision(3) << duration << " [us] " <<  std::endl;

   float* hostResult = (float*)malloc(imageWidth * imageHeight * imageChannels * sizeof(float));
   cudaMemcpy(hostResult, deviceResult, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

   //=================================================================================
   //! @defgroup Final_Image_Generation 
   //=================================================================================
   wbImage_t binImage1, binImage2, ErodeImage1, ErodeImage2, ResultImage;
   const char* outfilename4 = "BinaryOutput_Grey.ppm";
   const char* outfilename5 = "BinaryOutput_YUV.ppm";
   const char* outfilename6 = "ErodedOutput_Light.ppm";
   const char* outfilename7 = "ErodedOutput_Shadow.ppm";
   const char* outfilename8 = "Result.ppm";

   // Generate Output Artifacts 
   binImage1 = wbImage_new(imageWidth, imageHeight, 1, light_binarized_image_out_grey);
   wbExport(outfilename4, binImage1);

   binImage2 = wbImage_new(imageWidth, imageHeight, 1, binarized_image_out_YUV);
   wbExport(outfilename5, binImage2);

   ErodeImage1 = wbImage_new(imageWidth, imageHeight, 1, hostLightMask);
   wbExport(outfilename6, ErodeImage1);

   ErodeImage2 = wbImage_new(imageWidth, imageHeight, 1, hostShadowMask);
   wbExport(outfilename7, ErodeImage2);

   ResultImage = wbImage_new(imageWidth, imageHeight, 3, hostResult);
   wbExport(outfilename8, ResultImage);

   //=================================================================================
   //! @defgroup Cleanup 
   //=================================================================================
   cudaFree(deviceInputImageData);
   cudaFree(deviceOutputImageData1);
   cudaFree(deviceOutputImageData2);
   cudaFree(deviceOutputImageData3);
   cudaFree(deviceGrayHistImage);
   cudaFree(deviceGrayHisto);
   cudaFree(deviceYUVHistImage);
   cudaFree(deviceYUVHisto);
   cudaFree(light_binarized_image_grey);
   cudaFree(binarized_image_YUV);
   cudaFree(deviceShadowRedSum);
   cudaFree(deviceShadowGreenSum);
   cudaFree(deviceShadowBlueSum);
   cudaFree(deviceLightRedSum);
   cudaFree(deviceLightGreenSum);
   cudaFree(deviceLightBlueSum);
   cudaFree(deviceShadowSum);
   cudaFree(deviceLightSum);
   cudaFree(deviceResult);
   cudaFree(deviceSmoothMask);
   cudaFree(deviceShadowMask);
   cudaFree(deviceLightMask);
   cudaFree(deviceMask);
   cudaFree(deviceKernelMask);
   cudaFree(deviceConvMask);
   wbImage_delete(outputImage1);
   wbImage_delete(outputImage2);
   wbImage_delete(outputImage3);
   wbImage_delete(inputImage);
   wbImage_delete(binImage1);
   wbImage_delete(binImage2);
   wbImage_delete(ErodeImage1);
   wbImage_delete(ErodeImage2);
   wbImage_delete(ResultImage);

   return (EXIT_SUCCESS);
}

