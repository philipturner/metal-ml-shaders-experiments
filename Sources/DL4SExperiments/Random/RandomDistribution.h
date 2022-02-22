//
//  RandomDistribution.h
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

#ifndef RandomDistribution_h
#define RandomDistribution_h

#include <metal_stdlib>
using namespace metal;

class RandomDistribution {
public:
    enum DistributionType: ushort {
        uniform = 0,
        normal2 = 1,
        bernoulli = 2
    };
    
    static float get_uniform(float input, float range, float minimum)
    {
        return fma(input, range, minimum);
    }
    
    template <typename sincos_type> // accelerate bfloat16 computation with half precision
    static float2 get_normal2(float2 input, float mean, float standard_deviation)
    {
        float log_val = -log(input[0]);
        float scale = sqrt(log_val + log_val) * standard_deviation;
        float angle = input[1] * 2 * M_PI_F;
        
        sincos_type cosval;
        sincos_type sinval = sincos(sincos_type(angle), cosval);
        
        return {
            scale * float(sinval) + mean,
            scale * float(cosval) + mean
        };
    }
    
    template <typename output_type> // skip compressing a bfloat16 known at compile time
    static output_type get_bernoulli(float input, float p)
    {
        return (input <= p) ? output_type(1) : output_type(0);
    }
};

#endif /* RandomDistribution_h */
