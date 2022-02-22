//
//  Activation.metal
//  DL4SExperiments
//
//  Created by Philip Turner on 11/13/21.
//

#include <metal_stdlib>
#include "Activation.h"
using namespace metal;

// CELU

float Activation::celu(float x, float a, float a_recip)
{
    if (x >= 0) { return x; }
    
    return a * (exp(x * a_recip) - 1);
}

float Activation::celu_grad(float x, float grad_out, float a, float a_recip)
{
    float grad = (x >= 0) ? 1 : exp(x * a_recip);
    return grad_out * grad;
}

// ELU

float Activation::elu(float x, float a)
{
    if (x >= 0) { return x; }
    
    return a * (exp(x) - 1);
}

float Activation::elu_grad(float x, float grad_out, float a)
{
    float grad = (x >= 0) ? 1 : (exp(x) * a);
    return grad_out * grad;
}

// GeLU

float Activation::gelu(float x)
{
    return x * sigmoid(1.702 * x);
}

float Activation::gelu_grad(float x, float grad_out)
{
    float sgmd = sigmoid(1.702 * x);
    float grad_sgmd = sgmd * (1.702 - 1.702 * sgmd);
    
    float grad = sgmd + x * grad_sgmd;
    return grad_out * grad;
}

// Hard Shrink

float Activation::hard_shrink(float x, float a)
{
    return (abs(x) > a) ? x : 0;
}

float Activation::hard_shrink_grad(float x, float grad_out, float a)
{
    return shrink_grad(x, grad_out, a);
}

// Hard Sigmoid

float Activation::hard_sigmoid(float x)
{
    return relu6(x + 3) / 6;
}

float Activation::hard_sigmoid_grad(float x, float grad_out)
{
    float is_inside = clamp(x, float(-3), float(3)) == x;
    float grad = select(float(0), float(1) / 6, is_inside);
    
    return grad_out * grad;
}

// Hard Swish

float Activation::hard_swish(float x)
{
    return x * hard_sigmoid(x);
}

float Activation::hard_swish_grad(float x, float grad_out)
{
    float r6 = relu6(x);
    float grad_r6 = select(0, 1, r6 == x);
    
    float grad = r6 + x * grad_r6;
    return grad_out * grad;
}

// Hard Tanh

float Activation::hard_tanh(float x)
{
    return clamp(x, float(-1), float(1));
}

float Activation::hard_tanh_grad(float x, float grad_out)
{
    float grad = select(0, 1, abs(x) > 1);
    return grad_out * grad;
}

// Leaky ReLU

float Activation::leaky_relu(float x)
{
    return parametric_relu(x, 0.01);
}

float Activation::leaky_relu_grad(float x, float grad_out)
{
    return parametric_relu_grad(x, grad_out, 0.01);
}

// Linear

float Activation::linear(float x, float a, float b)
{
    return a * x + b;
}

float Activation::linear_grad(float x, float grad_out, float a, float b)
{
    return grad_out * a;
}

// LiSHT

float Activation::lisht(float x)
{
    return x * tanh(x);
}

float Activation::lisht_grad(float x, float grad_out)
{
    float tx = tanh(x);
    float grad_tx = 1 - tx * tx;
    
    float grad = x * grad_tx + tx;
    return grad_out * grad;
}

// Log-Sigmoid

float Activation::log_sigmoid(float x)
{
    return log(sigmoid(x));
}

float Activation::log_sigmoid_grad(float x, float grad_out)
{
    float grad = 1 / (1 + exp(x));
    return grad_out * grad;
}

// Mish

float Activation::mish(float x)
{
    if (x >= 20) { return x; }
    
    float t_sqrt = 1 + exp(x);
    
    float numerator = fma(t_sqrt, t_sqrt, -1);
    float denominator = fma(t_sqrt, t_sqrt, 1);
    
    return x * numerator / denominator;
}

float Activation::mish_grad(float x, float grad_out)
{
    if (x >= 20) { return grad_out; }
    
    float t_sqrt = 1 + exp(x);
    float grad_sp = 1 - 1 / t_sqrt; // gradient of softplus
    
    float numerator = fma(t_sqrt, t_sqrt, -1);
    float denominator = fma(t_sqrt, t_sqrt, 1);
    float tsp = numerator / denominator; // tanh(softplus)
    
    float grad_tsp = (1 - tsp * tsp) * grad_sp;
    float grad = x * grad_tsp + tsp;
    
    return grad_out * grad;
}

// Parametric ReLU

float Activation::parametric_relu(float x, float a)
{
    float coeff = (x > 0) ? 1 : a;
    return coeff * x;
}

float Activation::parametric_relu_grad(float x, float grad_out, float a)
{
    float coeff = (x > 0) ? 1 : a;
    return grad_out * coeff;
}

// Parametric SoftPlus

float Activation::parametric_soft_plus(float x, float a, float a_recip)
{
    return a_recip * softplus(a * x);
}

float Activation::parametric_soft_plus_grad(float x, float grad_out, float a, float a_recip)
{
    return grad_out / (1 + exp(a * -x));
}

// ReLU

float Activation::relu(float x)
{
    return fmax(x, 0);
}

float Activation::relu_grad(float x, float grad_out)
{
    return grad_out * sign(x);
}

// ReLU6

float Activation::relu6(float x)
{
    return clamp(x, float(0), float(6));
}

float Activation::relu6_grad(float x, float grad_out)
{
    float grad = (relu6(x) == x) ? grad_out : 0;
    return grad_out * grad;
}

// ReLUN

float Activation::relun(float x, float a, float b)
{
    if (x > b) { return b; }
    
    return (x >= 0) ? x : a * x;
}

float Activation::relun(float x, float grad_out, float a, float b)
{
    if (x > b) { return 0; }
    
    float grad = select(a, float(1), x >= 0);
    return grad_out * grad;
}

// SELU

constant float SELU_A = 1.6732632423543772848170429916717;
constant float SELU_SCALE = 1.0507009873554804934193349852946;

float Activation::selu(float x)
{
    if (x >= 0) { return SELU_SCALE * x; }
    
    return SELU_SCALE * SELU_A * (exp(x) - 1);
}

float Activation::selu_grad(float x, float grad_out, float a, float b)
{
    float grad = (x >= 0) ? SELU_SCALE : (exp(x) * SELU_SCALE * SELU_A);
    return grad_out * grad;
}

// Shrink

float Activation::shrink(float x, float a)
{
    if (abs(x) < a) { return 0; }
    
    return x - copysign(a, x);
}

float Activation::shrink_grad(float x, float grad_out, float a)
{
    float grad = select(0, 1, abs(x) > a);
    return grad_out * grad;
}

// Sigmoid

float Activation::sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float Activation::sigmoid_grad(float x, float grad_out)
{
    float sgmd = sigmoid(x);
    float grad = sgmd * (1 - sgmd);
    
    return grad_out * grad;
}

// Smooth ReLU

float Activation::smooth_relu(float x, float a, float a_recip)
{
    if (x <= 0) { return 0; }
    
    float lgx = log(a * x + 1);
    return x - a_recip * lgx;
}

float Activation::smooth_relu_grad(float x, float grad_out, float a, float a_recip)
{
    if (x <= 0) { return 0; }
    
    float grad = x / (a_recip + x);
    return grad_out * grad;
}

// Softplus

float Activation::softplus(float x)
{
    return log(1 + exp(x));
}

float Activation::softplus_grad(float x, float grad_out)
{
    return grad_out / (1 + exp(-x));
}

// Softsign

float Activation::softsign(float x)
{
    return x / (abs(x) + 1);
}

float Activation::softsign_grad(float x, float grad_out)
{
    float denom_sqrt = 1 + abs(x);
    return grad_out / (denom_sqrt * denom_sqrt);
}

// Squareplus

float Activation::squareplus(float x)
{
    float temp = sqrt(x * x + 4);
    return 0.5 * (x + temp);
}

float Activation::squareplus_grad(float x, float grad_out)
{
    float sqrt_grad = x * rsqrt(x * x + 4);
    float grad = 0.5 + 0.5 * sqrt_grad;
    
    return grad_out * grad;
}

// Swish

float Activation::swish(float x)
{
    return x * sigmoid(x);
}

float Activation::swish_grad(float x, float grad_out)
{
    float sgmd = sigmoid(x);
    float temp = x + fma(-x, sgmd, 1);
    
    return grad_out * sgmd * temp;
}

// Tanhshrink

float Activation::tanh_shrink(float x)
{
    return x - tanh(x);
}

float Activation::tanh_shrink_grad(float x, float grad_out)
{
    float tx = tanh(x);
    return grad_out * (tx * tx);
}

// Threshold

float Activation::threshold(float x, float a, float b)
{
    return select(b, x, x > a);
}

float Activation::threshold_grad(float x, float grad_out, float a, float b)
{
    return (x > a) ? grad_out : 0;
}



namespace Naive = Activation::Naive;

// Mish

float Naive::mish(float x)
{
    return x * tanh(x < 20 ? softplus(x) : x);
}

float Naive::mish_grad(float x, float grad_out)
{
    float sp = x < 20 ? softplus(x) : x;
    float grad_sp = 1 - exp(-sp);
    
    float tsp = tanh(sp);
    float grad_tsp = (1 - tsp * tsp) * grad_sp;
    
    float grad = x * grad_tsp + tsp;
    return grad_out * grad;
}

// Swish

float Naive::swish_grad(float x, float grad_out)
{
    float sgmd = sigmoid(x);
    float fx = x * sgmd;
    
    float grad = fx + sgmd * (1 - fx);
    return grad_out * grad;
}
