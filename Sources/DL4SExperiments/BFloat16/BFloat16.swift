//
//  BFloat16.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/16/21.
//

import ARHeadsetUtil

struct BFloat16 {
    fileprivate var storage: UInt16
    
    init(_ floatVal: Float) {
        self.storage = unsafeBitCast(floatVal, to: simd_ushort2.self)[1]
    }
}

struct simd_bfloat16_2 {
    fileprivate var storage: simd_ushort2
    
    init(_ floatVal: simd_float2) {
        storage = .init(
            BFloat16(floatVal[0]).storage,
            BFloat16(floatVal[1]).storage
        )
    }
}

struct simd_bfloat16_3 {
    fileprivate var storage: simd_ushort3
    
    init(_ floatVal: simd_float3) {
        storage = .init(
            BFloat16(floatVal[0]).storage,
            BFloat16(floatVal[1]).storage,
            BFloat16(floatVal[2]).storage
        )
    }
}

struct simd_bfloat16_4 {
    fileprivate var storage: simd_ushort4
    
    init(_ floatVal: simd_float4) {
        storage = .init(
            BFloat16(floatVal[0]).storage,
            BFloat16(floatVal[1]).storage,
            BFloat16(floatVal[2]).storage,
            BFloat16(floatVal[3]).storage
        )
    }
}

//struct simd_bfloat16_8 {
//    fileprivate var storage: simd_ushort8
//
//    init(_ floatVal: simd_float4) {
//        storage = .init(
//            BFloat16(floatVal[0]).storage,
//            BFloat16(floatVal[1]).storage,
//            BFloat16(floatVal[2]).storage,
//            BFloat16(floatVal[3]).storage,
//
//            BFloat16(floatVal[4]).storage,
//            BFloat16(floatVal[5]).storage,
//            BFloat16(floatVal[6]).storage,
//            BFloat16(floatVal[7]).storage
//        )
//    }
//}



extension Float {
    
    init(_ bfloat16Val: BFloat16) {
        let vector = simd_ushort2(0, bfloat16Val.storage)
        self = unsafeBitCast(vector, to: Float.self)
    }
    
}

extension simd_float2 {
    
    init(_ bfloat16Val: simd_bfloat16_2) {
        self = .init(
            Float(unsafeBitCast(bfloat16Val.storage[0], to: BFloat16.self)),
            Float(unsafeBitCast(bfloat16Val.storage[1], to: BFloat16.self))
        )
    }
    
}

extension simd_float3 {
    
    init(_ bfloat16Val: simd_bfloat16_3) {
        self = .init(
            Float(unsafeBitCast(bfloat16Val.storage[0], to: BFloat16.self)),
            Float(unsafeBitCast(bfloat16Val.storage[1], to: BFloat16.self)),
            Float(unsafeBitCast(bfloat16Val.storage[2], to: BFloat16.self))
        )
    }
    
}

extension simd_float4 {
    
    init(_ bfloat16Val: simd_bfloat16_4) {
        self = .init(
            Float(unsafeBitCast(bfloat16Val.storage[0], to: BFloat16.self)),
            Float(unsafeBitCast(bfloat16Val.storage[1], to: BFloat16.self)),
            Float(unsafeBitCast(bfloat16Val.storage[2], to: BFloat16.self)),
            Float(unsafeBitCast(bfloat16Val.storage[3], to: BFloat16.self))
        )
    }
    
}

//extension simd_float8 {
//
//    init(_ bfloat16Val: simd_bfloat16_8) {
//        self = .init(
//            Float(unsafeBitCast(bfloat16Val.storage[0], to: BFloat16.self)),
//            Float(unsafeBitCast(bfloat16Val.storage[1], to: BFloat16.self)),
//            Float(unsafeBitCast(bfloat16Val.storage[2], to: BFloat16.self)),
//            Float(unsafeBitCast(bfloat16Val.storage[3], to: BFloat16.self)),
//
//            Float(unsafeBitCast(bfloat16Val.storage[4], to: BFloat16.self)),
//            Float(unsafeBitCast(bfloat16Val.storage[5], to: BFloat16.self)),
//            Float(unsafeBitCast(bfloat16Val.storage[6], to: BFloat16.self)),
//            Float(unsafeBitCast(bfloat16Val.storage[7], to: BFloat16.self))
//        )
//    }
//
//}
