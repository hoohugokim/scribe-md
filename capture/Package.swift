// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "appaudio-capture",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "appaudio-capture",
            path: "Sources"
        )
    ]
)
