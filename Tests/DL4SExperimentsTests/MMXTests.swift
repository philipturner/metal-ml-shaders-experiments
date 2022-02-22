import XCTest
@testable import DL4SExperiments

import ARHeadsetUtil

extension DL4SExperimentsTests {
    
    func testAllMMXPermutations() throws {
        var combinations: [[Bool]] = []
        
//        let a = 0
        let b = 0
        let c = 1
        let d = 1
        
        // haven't even tested having the input vector be bfloat16
        // haven't tried a very specific matrix multiplication with values that aren't the same for every component
        // haven't tested having different input vector stride than size
        
        for a in 0..<2 {
//            for b in 0..<2 {
//                for c in 0..<2 {
//                    for d in 0..<2 {
                        combinations.append([a != 0, b != 0, c != 0, d != 0])
//                    }
//                }
//            }
        }
        
        // TODO: - test every parameter combination possible, guarantee no bugs
        
        for combination in combinations {
            let resources = MMXResources(isTranspose: combination[0],
                                         matrixIsBF16: combination[1],
                                         usingNonuniformDispatch: combination[2],
                                         supportsSIMDPermute: combination[3])

            print()
            print("isTranspose:\(resources.isTranspose)")
            print("matrixIsBF16:\(resources.matrixIsBF16)")
            print("usingNonuniformDispatch:\(resources.usingNonuniformDispatch)")
            print("supportsSIMDPermute:\(resources.supportsSIMDPermute)")

            try resources.testSGEMVExample()
        }
        
        for combination in combinations {
            let resources = MMXResources(isTranspose: combination[0],
                                         matrixIsBF16: combination[1],
                                         usingNonuniformDispatch: combination[2],
                                         supportsSIMDPermute: combination[3])
            
            print()
            print("isTranspose:\(resources.isTranspose)")
            print("matrixIsBF16:\(resources.matrixIsBF16)")
            print("usingNonuniformDispatch:\(resources.usingNonuniformDispatch)")
            print("supportsSIMDPermute:\(resources.supportsSIMDPermute)")
            
            try resources.testSGEMMExample()
        }
    }
    
}
