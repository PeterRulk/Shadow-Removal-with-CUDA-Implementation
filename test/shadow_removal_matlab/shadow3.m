%-------------------------------------------------------------------------
% Serial implementation of third shadow removal algorithm
% taken from "Contrast Limited Adaptive Histogram Equalization (CLAHE) 
% and Shadow Removal for Controlled Environment Plant Production Systems"
% By Burak Unal
%-------------------------------------------------------------------------
% Modified By: Mason Edgar 
%-------------------------------------------------------------------------
clc
clear 
close  all
format compact

VIEW_FIGURES   =  0;    % Toggle figures 
NUM_ITERATIONS = 10;    % Average over 10 iterations 
kernel_avgs    = [];

%% Reading and plotting original image
image = imread('plt4.jpg');

% Test Data
colorspace_items      = zeros(NUM_ITERATIONS, 1, 'double');
gray_histo_items      = zeros(NUM_ITERATIONS, 1, 'double');
yuv_histo_items       = zeros(NUM_ITERATIONS, 1, 'double');
gray_ImBin_items      = zeros(NUM_ITERATIONS, 1, 'double');
yuv_ImBin_items       = zeros(NUM_ITERATIONS, 1, 'double');
erosion_shadow_items  = zeros(NUM_ITERATIONS, 1, 'double');
erosion_light_items   = zeros(NUM_ITERATIONS, 1, 'double');
convolution_items     = zeros(NUM_ITERATIONS, 1, 'double');
shadow_ImMap_items    = zeros(NUM_ITERATIONS, 1, 'double');
light_ImMap_items     = zeros(NUM_ITERATIONS, 1, 'double');
shadow_MaskComb_items = zeros(NUM_ITERATIONS, 1, 'double');
light_MaskComb_items  = zeros(NUM_ITERATIONS, 1, 'double');
result_image_items    = zeros(NUM_ITERATIONS, 1, 'double');


if (VIEW_FIGURES)
    figure('NumberTitle','off','Name','Original Lettuce Image');
    imshow(image)
    title('Original Lettuce Lmage');
end 

% Calculating Color Invariant Image
image_double          = im2double(image);
redPart               = image_double(:,:,1);
greenPart             = image_double(:,:,2);
bluePart              = image_double(:,:,3);
[row, col, ~]         = size(image);
color_invariant_image = zeros(row,col,3);


[yuv, gray, color_invariant_image] = ColorspaceKernel(row, col,              ...
                                                      color_invariant_image, ...
                                                      redPart,               ...
                                                      greenPart,             ...
                                                      bluePart,              ...
                                                      image);
%-------------------------------------------------------------
func = @() ColorspaceKernel(row, col, color_invariant_image, redPart, greenPart, bluePart, image);
for i=1:NUM_ITERATIONS
    %---
    colorspace_items(i) = ( timeit(func, 2) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(colorspace_items)];
disp("Completed ColorspaceKernel")
%-------------------------------------------------------------

if (VIEW_FIGURES)
    figure('NumberTitle','off','Name','Color Invariant Image');
    imshow(color_invariant_image);
    title("Color Invariant Image");
    figure('NumberTitle','off','Name','YUV Image');
    imshow(yuv);
    title("YUV Image");
end 

%% Histograms 
num_bins  = 256;
IGray     = im2uint8(gray(:));
yuv_Uchan = im2uint8(yuv(:,:,2));
%-------------------------------------------------------------
% HistogramKernel(IGray, num_bins);
func = @() HistogramKernel(IGray, num_bins);
for i=1:NUM_ITERATIONS
    %---
    gray_histo_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(gray_histo_items)];
disp("Completed HistogramKernel 1")
%-------------------------------------------------------------
% HistogramKernel(IYUV, num_bins);
func = @() HistogramKernel(yuv_Uchan, num_bins);
%-------------------------------------------------------------
for i=1:NUM_ITERATIONS
    %---
    yuv_histo_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(yuv_histo_items)];
disp("Completed HistogramKernel 2")
%-------------------------------------------------------------

grayThreshold = eddie_graythresh(gray);
yuvThreshold  = eddie_graythresh(yuv_Uchan);

%% Image Binarization 
gray_mask = ImageBinarizationKernelGray(gray, grayThreshold);
%-------------------------------------------------------------
func = @() ImageBinarizationKernelGray(gray, grayThreshold);
for i=1:NUM_ITERATIONS
   %---
   gray_ImBin_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(gray_ImBin_items)];
disp("Completed ImageBinarizationKernel 1")
%-------------------------------------------------------------

yuv_mask = ImageBinarizationKernelYUV(yuv_Uchan, yuvThreshold);
%-------------------------------------------------------------
func = @() ImageBinarizationKernelYUV(yuv_Uchan, yuvThreshold);
for i=1:NUM_ITERATIONS
   %---
   yuv_ImBin_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(yuv_ImBin_items)];
disp("Completed ImageBinarizationKernel 2")
%-------------------------------------------------------------

if (VIEW_FIGURES)
    figure('NumberTitle','off','Name','YUV Mask Image');
    imshow(yuv_mask);
    title("YUV Mask Image");
    figure('NumberTitle','off','Name','Gray Image');
    imshow(gray);
    title("Gray Image");
    figure('NumberTitle','off','Name','Gray Mask Image');
    imshow(gray_mask);
    title("Gray Mask Image");
end 

%% Defining structuring element and performing erosion
strel = [1 1 1 1 1; 
         1 1 1 1 1; 
         1 1 1 1 1; 
         1 1 1 1 1; 
         1 1 1 1 1];

light_mask = 1 - gray_mask;


[eroded_gray_shadow_mask] = ErosionKernel(gray_mask, strel);
%-------------------------------------------------------------
func = @() ErosionKernel(gray_mask, strel);
for i=1:NUM_ITERATIONS
   %---
   erosion_shadow_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(erosion_shadow_items)];
disp("Completed ErosionKernel 1")
%-------------------------------------------------------------

[eroded_gray_light_mask] = ErosionKernel(light_mask, strel);
%-------------------------------------------------------------
func = @() ErosionKernel(light_mask, strel);
for i=1:NUM_ITERATIONS
   %---
   erosion_light_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(erosion_light_items)];
disp("Completed ErosionKernel 2")
%-------------------------------------------------------------

if (VIEW_FIGURES)
    figure('NumberTitle','off','Name','Eroded Gray Mask Image');
    imshow(eroded_gray_shadow_mask);
    title("Eroded Gray Shadow Mask Image");
    
    figure('NumberTitle','off','Name','Eroded Gray Light Mask Image');
    imshow(eroded_gray_light_mask);
    title("Eroded Gray Light Mask Image");
end 
%% Using structuring element to create smooth mask
strel = strel / 25;

[smoothmask] = ConvolutionKernel(yuv_mask, strel);
%-------------------------------------------------------------
func = @() ConvolutionKernel(yuv_mask, strel);
for i=1:NUM_ITERATIONS
   %---
   convolution_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(convolution_items)];
disp("Completed ConvolutionKernel")
%-------------------------------------------------------------

if (VIEW_FIGURES)
    figure('NumberTitle','off','Name','Smooth Mask Image');
    imshow(smoothmask);
    title("Smooth Mask Image");
end 


%% Finding average channel values in shadow/light areas for every channel
[shadowSum_red, shadowSum_green, shadowSum_blue] = ImageMappingKernelShadow(image_double, eroded_gray_shadow_mask);
%-------------------------------------------------------------
func = @() ImageMappingKernelShadow(image_double, eroded_gray_shadow_mask);
for i=1:NUM_ITERATIONS
   %---
   shadow_ImMap_items(i) = ( timeit(func, 3) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(shadow_ImMap_items)];
disp("Completed ImageMappingKernelShadow")
%-------------------------------------------------------------

[lightSum_red, lightSum_green, lightSum_blue] = ImageMappingKernelLight(image_double, eroded_gray_light_mask);
%-------------------------------------------------------------
func = @() ImageMappingKernelLight(image_double, eroded_gray_light_mask);
for i=1:NUM_ITERATIONS
   %---
   light_ImMap_items(i) = ( timeit(func, 3) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(light_ImMap_items)];
disp("Completed ImageMappingKernelLight")
%-------------------------------------------------------------

%% Mask Combination Kernels 
[shadowMask_sum, ~, ~] = ImageMaskCombinationKernelShadow(eroded_gray_shadow_mask);
%-------------------------------------------------------------
func = @() ImageMaskCombinationKernelShadow(eroded_gray_shadow_mask);
for i=1:NUM_ITERATIONS
   %---
   shadow_MaskComb_items(i) = ( timeit(func, 3) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(shadow_MaskComb_items)];
disp("Completed ImageMaskCombinationKernelShadow")
%-------------------------------------------------------------

[lightMask_sum, ~, ~] = ImageMaskCombinationKernelLight(eroded_gray_light_mask);
%-------------------------------------------------------------
func = @() ImageMaskCombinationKernelLight(eroded_gray_light_mask);
for i=1:NUM_ITERATIONS
   %---
   light_MaskComb_items(i) = ( timeit(func, 3) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(light_MaskComb_items)];
disp("Completed ImageMaskCombinationKernelLight")
%-------------------------------------------------------------

%% Averaging 
shadowavg_red   = shadowSum_red   / shadowMask_sum;
shadowavg_green = shadowSum_green / shadowMask_sum;
shadowavg_blue  = shadowSum_blue  / shadowMask_sum;

lightavg_red    = lightSum_red    / lightMask_sum;
lightavg_green  = lightSum_green  / lightMask_sum;
lightavg_blue   = lightSum_blue   / lightMask_sum;

% Calculating ratio of light-to-shadow in every channel
ratio_red   = lightavg_red   / shadowavg_red   - 1;
ratio_green = lightavg_green / shadowavg_green - 1;
ratio_blue  = lightavg_blue  / shadowavg_blue  - 1;

%% Image Result Kernel Removing shadow
result = zeros(size(image_double));

result = ResultImageKernel(result, ratio_red, smoothmask, image_double, ratio_green, ratio_blue);
%-------------------------------------------------------------
func = @() ResultImageKernel(result, ratio_red, smoothmask, image_double, ratio_green, ratio_blue);
for i=1:NUM_ITERATIONS
   %---
   result_image_items(i) = ( timeit(func, 1) * 1E6 );
end
kernel_avgs = [kernel_avgs; mean(result_image_items)];
disp("Completed ResultImageKernel")
%-------------------------------------------------------------

if (1)
    % Outputing final result
    figure('NumberTitle','off','Name','Shadowless Lettuce Image');
    imshow(result);
    title("Shadowless Lettuce Image");    
end 

return 
%--------------------------------------------------------------------------------------------
%                                       END OF SCRIPT 
%--------------------------------------------------------------------------------------------


%% MATLAB Equivalent Colorspace Kernel 
function [yuv, gray, color_invariant_image] = ColorspaceKernel(row, col, color_invariant_image, redPart, greenPart, bluePart, image)
    %---
    for i=1:row
        for j=1:col
            color_invariant_image(i,j,1) = atan(   redPart(i,j) / max(greenPart(i,j), bluePart(i,j))  );
            color_invariant_image(i,j,2) = atan( greenPart(i,j) / max(redPart(i,j),   bluePart(i,j))  );
            color_invariant_image(i,j,3) = atan(  bluePart(i,j) / max(redPart(i,j),   greenPart(i,j)) );
        end
    end
    %---
    yuv  = rgb2ycbcr(image);                  % Calculating RGB -> YUV 
    gray = rgb2gray(color_invariant_image);   % Calculating Color Invariant RGB -> Grayscale 
end


%% MATLAB Equivalent Histogram Kernel 
function [counts] = HistogramKernel(I, num_bins)
    %---
    % Note: This is done inside eddie_graythresh() normally
    %---
    counts = imhist(I, num_bins);
end


%% MATLAB Equivalent Image Binarization Kernel 
function imgMask = ImageBinarizationKernelGray(img, thresh)
    %---
    imgMask = ( 1 - double(imbinarize(img, thresh)) );
end

function imgMask = ImageBinarizationKernelYUV(img, thresh)
    %---
    imgMask = double(imbinarize(img, thresh));
end


%% MATLAB Equivalent Erosion Kernel 
function [eroded_mask] = ErosionKernel(mask, strel)
    %---
    eroded_mask = imerode(mask, strel);
end


%% MATLAB Equivalent Convolution Kernel 
function [smoothmask] = ConvolutionKernel(yuv_mask, strel)
    %---
    smoothmask = conv2(yuv_mask, strel, 'same');
end


%% MATLAB Equivalent Image Mask Summation Kernels 
function [shadowSum_red, shadowSum_green, shadowSum_blue] = ImageMappingKernelShadow(image_double, eroded_gray_shadow_mask)
    %---
    shadowSum_red   = sum(sum(image_double(:,:,1) .* eroded_gray_shadow_mask));
    shadowSum_green = sum(sum(image_double(:,:,2) .* eroded_gray_shadow_mask));
    shadowSum_blue  = sum(sum(image_double(:,:,3) .* eroded_gray_shadow_mask));
end

function [lightSum_red, lightSum_green, lightSum_blue] = ImageMappingKernelLight(image_double, eroded_gray_light_mask)
    %---
    lightSum_red   = sum(sum(image_double(:,:,1) .* eroded_gray_light_mask));
    lightSum_green = sum(sum(image_double(:,:,2) .* eroded_gray_light_mask));
    lightSum_blue  = sum(sum(image_double(:,:,3) .* eroded_gray_light_mask));
end


%% MATLAB Equivalent Image Mask Averaging Kernels 
function [shadowMask_sum, foo, bar] = ImageMaskCombinationKernelShadow(eroded_gray_shadow_mask)
    % Aren't needed but keeping becuase they are in the original code 
    foo = sum(sum(eroded_gray_shadow_mask));
    bar = sum(sum(eroded_gray_shadow_mask));
    % The actual summation 
    shadowMask_sum = sum(sum(eroded_gray_shadow_mask));
end

function [lightMask_sum, foo, bar] = ImageMaskCombinationKernelLight(eroded_gray_light_mask)
    % Aren't needed but keeping becuase they are in the original code 
    foo = sum(sum(eroded_gray_light_mask));
    bar = sum(sum(eroded_gray_light_mask));
    % The actual summation 
    lightMask_sum = sum(sum(eroded_gray_light_mask));
end


%% MATLAB Equivalent Result Image Kernel 
function result = ResultImageKernel(result, ratio_red, smoothmask, image_double, ratio_green, ratio_blue)
    %---
    result(:,:,1) = (ratio_red   + 1) ./ ((1 - smoothmask) * ratio_red   + 1) .* image_double(:,:,1);
    result(:,:,2) = (ratio_green + 1) ./ ((1 - smoothmask) * ratio_green + 1) .* image_double(:,:,2);
    result(:,:,3) = (ratio_blue  + 1) ./ ((1 - smoothmask) * ratio_blue  + 1) .* image_double(:,:,3);
end
