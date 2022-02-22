//
//  SafeModule.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/12/21.
//

import Foundation

private class BundleFinder { }

extension Foundation.Bundle {
    /// Returns the resource bundle associated with the current Swift module.
    static var safeModule: Bundle = {
        #if SWIFT_PACKAGE
        let bundleName = "DL4SExperiments_DL4SExperiments"
        
        let candidates = [
            // Bundle should be present here when the package is linked into an App.
            Bundle.main.resourceURL,
            
            // Bundle should be present here when the package is linked into a framework.
            Bundle(for: BundleFinder.self).resourceURL,
            
            // For command-line tools.
            Bundle.main.bundleURL,
        ]
        
        for candidate in candidates {
            let bundlePath = candidate?.appendingPathComponent(bundleName + ".bundle")
            if let bundle = bundlePath.flatMap(Bundle.init(url:)) {
                return bundle
            }
        }
        fatalError("unable to find bundle named DL4SExperiments_DL4SExperiments")
        #else
        Bundle(for: BundleFinder.self)
        #endif
    }()
}
