//*********************************************************************************
// Filename    : ProjectDefs.h
// Authors     : Mason W Edgar
// Course      : ECE 569 
// Semester    : Spring 2024
// Description : Top-Level Definitions for ECE 569 Project 
//*********************************************************************************
#ifndef INCLUDED_PROJECTDEFS_H_
#define INCLUDED_PROJECTDEFS_H_

// System Includes 
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <memory>
#include <array>
#include <time.h>
#include <atomic>
#include <initializer_list>
#include <chrono>

// CUDA Infrastructure 
#include <cuda.h>
#include <cuda_runtime.h>

// ECE 569 Lab Toolkit 
#include <wb.h>

// Kernel Definitions 
#include "ColorspaceKernelDefs.cuh"
#include "ConvolutionKernelDefs.cuh"
#include "IntegrationKernelDefs.cuh"
#include "ThresholdingKernelDefs.cuh"



#endif /* INCLUDED_PROJECTDEFS_H_ */
