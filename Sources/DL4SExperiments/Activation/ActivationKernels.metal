//
//  ActivationKernels.metal
//  DL4SExperiments
//
//  Created by Philip Turner on 11/13/21.
//

#include <metal_stdlib>
#include "Activation.h"
using namespace metal;

kernel void testActivation(constant uint   &data_count  [[buffer(0)]],
                           device   float2 *test_inputs [[buffer(1)]])
{ 
    for (uint i = 0; i < data_count; ++i)
    {
        test_inputs[i] = { Activation::mish_grad(test_inputs[i].x, test_inputs[i].y), 0 };
    }
}

// Combine all into one shader with function constants. That way, the needed functions can be compiled in parallel at runtime. Even thought that may actually harm total time taken to compile shaders, it makes the back-end compiler takes less time and makes the library object smaller. Use a high-level switch statement with a function constant.

// I do not have evidence to determine whether this is an ubershader phenomenon, so I may be wrong and doing this could harm performance. I must test my hypothesis and produce scientifically rigorous results.

// The biggest bottleneck is high-levelness (maintainability). It takes much less lines of code to call an enumeration from Swift that matches perfectly to the Metal one. So, I will go with the ubershader scheme.

// The problem is definitely with being high-level. I don't care if it harms load time right now, especially if using 8 performance cores on the M1 Max to load in parallel.
