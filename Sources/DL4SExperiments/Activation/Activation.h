//
//  Activation.h
//  DL4SExperiments
//
//  Created by Philip Turner on 11/13/21.
//

#ifndef Activations_h
#define Activations_h

#include <metal_stdlib>
using namespace metal;

namespace Activation {
    /// Source: https://developer.apple.com/documentation/mlcompute/mlcactivationtype/celu
    float celu(float x, float a, float a_recip);
    float celu_grad(float x, float grad_out, float a, float a_recip);
    
    float elu(float x, float a);
    float elu_grad(float x, float grad_out, float a);
    
    float gelu(float x);
    float gelu_grad(float x, float grad_out);
    
    /// Source: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
    float hard_shrink(float x, float a);
    float hard_shrink_grad(float x, float grad_out, float a);
    
    /// Source: https://arxiv.org/abs/1905.02244
    float hard_sigmoid(float x);
    float hard_sigmoid_grad(float x, float grad_out);
    
    /// Source: https://arxiv.org/abs/1905.02244
    float hard_swish(float x);
    float hard_swish_grad(float x, float grad_out);
    
    /// Source: https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh
    float hard_tanh(float x);
    float hard_tanh_grad(float x, float grad_out);
    
    float leaky_relu(float x);
    float leaky_relu_grad(float x, float grad_out);
    
    float linear(float x, float a, float b);
    float linear_grad(float x, float grad_out, float a, float b);
    
    float lisht(float x);
    float lisht_grad(float x, float grad_out);
    
    float log_sigmoid(float x);
    float log_sigmoid_grad(float x, float grad_out);
    
    /**
     Optimized version of `x * tanh(softplus(x))` with a threshold of 20.
     
     When fast math is enabled, Metal will return `NAN` when putting large numbers into `tanh`. However, the optimized `mish` does not have that problem.
     */
    float mish(float x);
    /// Source: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h
    float mish_grad(float x, float grad_out);
    
    float parametric_relu(float x, float a);
    float parametric_relu_grad(float x, float grad_out, float a);
    
    /// Source: https://arxiv.org/abs/2006.14536
    float parametric_soft_plus(float x, float a, float a_recip);
    float parametric_soft_plus_grad(float x, float grad_out, float a, float a_recip);
    
    float relu(float x);
    float relu_grad(float x, float grad_out);
    
    float relu6(float x);
    float relu6_grad(float x, float grad_out);
    
    /// Source: https://developer.apple.com/documentation/mlcompute/mlcactivationtype/relun
    float relun(float x, float a, float b);
    float relun(float x, float grad_out, float a, float b);
    
    /// Source: https://developer.apple.com/documentation/mlcompute/mlcactivationtype/selu
    float selu(float x);
    float selu_grad(float x, float grad_out, float a, float b);
    
    /// Source: https://docs.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_activation_shrink_operator_desc
    float shrink(float x, float a);
    float shrink_grad(float x, float grad_out, float a);
    
    float sigmoid(float x);
    float sigmoid_grad(float x, float grad_out);
    
    /// Source: https://arxiv.org/abs/2006.14536
    float smooth_relu(float x, float a, float a_recip);
    float smooth_relu_grad(float x, float grad_out, float a, float a_recip);
    
    float softplus(float x);
    float softplus_grad(float x, float grad_out);
    
    float softsign(float x);
    float softsign_grad(float x, float grad_out);
    
    float squareplus(float x);
    float squareplus_grad(float x, float grad_out);
    
    float swish(float x);
    float swish_grad(float x, float grad_out);
    
    /// Source: https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html#torch.nn.Tanhshrink
    float tanh_shrink(float x);
    float tanh_shrink_grad(float x, float grad_out);
    
    /// Source: https://developer.apple.com/documentation/mlcompute/mlcactivationtype/threshold
    float threshold(float x, float a, float b);
    float threshold_grad(float x, float grad_out, float a, float b);
    
    /// Slower and/or numerically unstable versions of complex activation functions, from before optimization.
    namespace Naive {
        float mish(float x);
        float mish_grad(float x, float grad_out);
        
        float swish_grad(float x, float grad_out);
    }
}

#endif /* Activations_h */
