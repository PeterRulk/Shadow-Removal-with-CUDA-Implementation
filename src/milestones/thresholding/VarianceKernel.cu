//*********************************************************************************
// Filename    : VarianceKernel.cu
// Authors     : Peter Rulkov
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Histogram Kernel 
//*********************************************************************************

#include "ProjectDefs.h"
#include "ColorspaceKernelDefs.cuh"

// Function to calculate the Otsu threshold
int calculate_otsu_threshold(float histogram[], int num_bins) {
    int total_pixels = 0;
    double sum = 0;
    double sum_bg = 0, weight_bg = 0;
    double max_variance = 0;
    int threshold = 0;

    // Calculate total number of pixels and the weighted sum of the histogram.
    for (int i = 0; i < num_bins; i++) {
        total_pixels += histogram[i];       // total pixel count.
        sum += i * histogram[i];            // weighted sum of pixel intensities.
    }

    // Second pass: Evaluate the variance for each possible threshold value.
    for (int t = 1; t < num_bins; t++) {
        weight_bg += histogram[t - 1];         //  background weight. (t-1) used for adjustment
        sum_bg += (t - 1) * histogram[t - 1];  //  background sum. (t-1) used for adjustment

        double weight_fg = total_pixels - weight_bg; // Calculate the foreground weight.
        double mean_bg = sum_bg / weight_bg;         // Calculate the mean for the background.
        double mean_fg = (sum - sum_bg) / weight_fg; // Calculate the mean for the foreground.

        // Calculate the variance for the current threshold.
        double variance = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);

        //ARGMAX
        if (variance > max_variance) {
            max_variance = variance; //update maximum variance values
            threshold = t; //store threshold
        }
    }

    // Return the optimal threshold where the maximum variance occurs.
    return threshold;
}
