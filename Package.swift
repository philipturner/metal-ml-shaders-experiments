// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "DL4SExperiments",
    platforms: [ // change back to macOS 10.15, iOS 13, tvOS 13, watchOS 6 after Xcode 13.2 is released
        .macOS(.v10_15),
        .iOS(.v13),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "DL4SExperiments",
            targets: ["DL4SExperiments"]),
    ],
    dependencies: [
        .package(name: "ARHeadsetUtil", url: "https://github.com/philipturner/ARHeadsetUtil", branch: "main")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "DL4SExperiments",
            dependencies: ["ARHeadsetUtil"],
            resources: [
                .process("Activation/Activation.metal"),
                .process("Activation/ActivationKernels.metal"),
                .process("Conv/Conv.metal"),
                .process("Random/Random.metal"),
                .process("MMX/SGEMMSmall.metal"),
                .process("MMX/SGEMV.metal"),
            ],
            swiftSettings: [.unsafeFlags(["-enable-testing"])]),
        .testTarget(
            name: "DL4SExperimentsTests",
            dependencies: ["DL4SExperiments"]),
    ]
)
