import XCTest
@testable import DL4SExperiments

import ARHeadsetUtil

extension DL4SExperimentsTests {
    
    func testAllRandomPermutations() throws {
        var combinations: [[Bool]] = []
        
        let a = 0
        let b = 0
        
//        for a in 0..<2 {
//            for b in 0..<2 {
                combinations.append([ a != 0, b != 0 ])
//            }
//        }
        
        for _ in 0..<1 {
            for combination in combinations {
                let resources = RandomResources(randomNumberSize: 50000,
                                                supportsLong: combination[0],
                                                generatingBF16: combination[1],
                                                distributionType: .bernoulli,
                                                paramA: 0.5,
                                                paramB: 1)
                
                print()
                print("supportsLong:\(resources.supportsLong) generatingBF16:\(resources.generatingBF16)")
                
//                try resources.testRandomExample(times: 10)
                
                try resources.testDistributionExample(min: 0, max: 1, intervals: 10)
            }
        }
    }

    func testRandomHistogram() throws {
        let resources = RandomResources(randomNumberSize: 64,
                                        supportsLong: true,
                                        generatingBF16: false,
                                        distributionType: .uniform)
        
        
        
        var s4tf_rng = S4TF_PhiloxRandomNumberGenerator(uint64Seed: .init(resources.key))
        print("\nS4TF")
        
        // from the 128-bit counter, goes to: [2] -> [3] -> [0] -> [1]
        let vec1 = unsafeBitCast(s4tf_rng.next(), to: simd_uint2.self)
        let vec2 = unsafeBitCast(s4tf_rng.next(), to: simd_uint2.self)
        print(vec1[0], vec1[1], vec2[0], vec2[1])
        
        var rng = PhiloxRandomNumberGenerator(uint64Seed: .init(resources.key))
        print("\nDL4S")
        print(simd_float4(rng.next()) / simd_float4(repeating: Float(UInt32.max) + 1))
        
        _ = resources.apple_randomMethod()
        
        print("\nMPS")
        let appleRandomArray = (0..<resources.randomNumberSize).map {
            Float(bitPattern: resources.readValue32(index: $0))
        }
        
        let appleHistCodes = appleRandomArray.map { min(Int($0 * 10), 9) }
        
        _ = resources.custom_randomMethod()
        
        print("\nCustom")
        let customRandomArray = (0..<resources.randomNumberSize).map {
            resources.readValue32(index: $0)
        }
        
        for i in 0..<4 {
            print(customRandomArray[i])
        }
        
        let customHistCodes = customRandomArray.map { (Int($0) * 10) >> 32 }
        
        // need to profile my execution time against MPS's
        
        func showHistogram(codes: [Int]) {
            var histogram = [Int: Int]()
            
            for i in 0..<10 {
                histogram[i] = 0
            }
            
            for code in codes {
                histogram[code]! += 1
            }
            
            print("histogram")
            
            for i in 0..<10 {
                let val = histogram[i]!
                
                let proportion = Float(val) / Float(codes.count)
                let percent = Int(proportion * 100)
                
                print(percent, "%")
            }
        }
        
        showHistogram(codes: appleHistCodes)
        showHistogram(codes: customHistCodes)
    }
    
    
    /**
     
     
     0.3954389
     0.7087767
     0.84839225
     0.24764442
     0.35422432
     0.6420766
     0.86323714
     0.49132705
     0.39693177
     0.3910414
     0.6787739
     0.4789164
     0.3813876
     0.048594594
     0.8042885
     0.018913269
     */
    
    
}
