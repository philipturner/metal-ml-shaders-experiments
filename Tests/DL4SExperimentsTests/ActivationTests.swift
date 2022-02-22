import XCTest
@testable import DL4SExperiments

import ARHeadsetUtil

extension DL4SExperimentsTests {
    
    func testOptimizingActivation() throws {
        func sigmoid(_ x: Float) -> Float {
            1 / (1 + exp(-x))
        }
        
        func sigmoid_derivative(_ x: Float) -> Float {
            let sgmd = sigmoid(x)
            return sgmd * (1 - sgmd)
        }
        
        func relu6(_ x: Float) -> Float {
            simd_clamp(x, 0, 6)
        }
        
        func relu6_derivative(_ x: Float) -> Float {
            (x > 0 && x < 6) ? 1 : 0
        }
        
        func activation_grad(_ input: simd_float2) -> Float {
            let x = input[0]
            
            let grad = 1 / sigmoid(x) * sigmoid_derivative(x)
            
            let grad_out = input[1]
            return grad_out * grad;
        }
        
        func activation_grad2(_ input: simd_float2) -> Float {
            let x = input[0]
            
            let grad = 1 / (exp(x) + 1)
            
            let grad_out = input[1]
            return grad_out * grad;
        }
        
//        let vals1: [Float] = [
//            -2, -0.8, 0, 0.5, 1.9, 7, 10, 25, 200
//        ]
        
        let vals2: [simd_float2] = [
            [ -2,    0.9  ],
            [ -0.8,  8    ],
            [  0,    0    ],
            [  0.5, -0.05 ],
            [  1.9,  0.001],
            [  7,   -4    ],
            [ 10, 10000   ],
            [ 25,   30    ],
            [200, -900    ]
        ]
        
        print(vals2.map(activation_grad(_:)))
        print(vals2.map(activation_grad2(_:)))
        
        // get array data into a buffer
        
        let resources = ActivationResources()
        
        let bufferSize = vals2.count * MemoryLayout<simd_float2>.stride
        let buffer = resources.device.makeBuffer(length: bufferSize)!
        buffer.label = "test buffer"
        
        let pointer = buffer.contents().assumingMemoryBound(to: simd_float2.self)
        var vals2_copy = vals2
        memcpy(pointer, &vals2_copy, vals2.count * 8)
        
        // test for threshold for the GPU kernel
        
        let commandBuffer = resources.commandQueue.makeDebugCommandBuffer()
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.setComputePipelineState(resources.activationPipelineState)
        
        var numVals = vals2.count
        computeEncoder.setBytes(&numVals, length: 4, index: 0)
        computeEncoder.setBuffer(buffer,  offset: 0, index: 1)
        computeEncoder.dispatchThreadgroups(1, threadsPerThreadgroup: 1)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        memcpy(&vals2_copy, pointer, vals2.count * 8)
        print(vals2_copy.map { $0.x })
    }
    
}
