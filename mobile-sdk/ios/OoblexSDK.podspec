Pod::Spec.new do |spec|
  spec.name         = "OoblexSDK"
  spec.version      = "1.0.0"
  spec.summary      = "Real-time AI video processing SDK for iOS"
  spec.description  = <<-DESC
    Ooblex SDK provides real-time AI-powered video processing capabilities for iOS applications.
    Features include face detection, emotion recognition, object detection, background effects,
    and more. Integrates seamlessly with WebRTC for low-latency video streaming.
  DESC

  spec.homepage     = "https://github.com/ooblex/ooblex-ios-sdk"
  spec.license      = { :type => "MIT", :file => "LICENSE" }
  spec.author       = { "Ooblex" => "support@ooblex.com" }
  spec.source       = { :git => "https://github.com/ooblex/ooblex-ios-sdk.git", :tag => "v#{spec.version}" }
  
  spec.platform     = :ios, "13.0"
  spec.swift_version = "5.0"
  
  spec.source_files = "OoblexSDK/**/*.{swift,h,m}"
  spec.exclude_files = "OoblexSDK/**/*Tests.swift"
  
  # Dependencies
  spec.dependency "GoogleWebRTC", "~> 1.1"
  
  # Frameworks
  spec.frameworks = [
    "Foundation",
    "UIKit",
    "AVFoundation",
    "CoreMedia",
    "CoreVideo",
    "CoreML",
    "Vision",
    "Metal",
    "MetalKit"
  ]
  
  # Build settings
  spec.pod_target_xcconfig = {
    'ENABLE_BITCODE' => 'NO',
    'SWIFT_VERSION' => '5.0',
    'OTHER_SWIFT_FLAGS' => '-DCocoaPods'
  }
  
  # Resources
  spec.resource_bundles = {
    'OoblexSDK' => ['OoblexSDK/Resources/**/*']
  }
  
  # Preserve paths
  spec.preserve_paths = "README.md", "CHANGELOG.md"
  
  # Documentation
  spec.documentation_url = "https://docs.ooblex.com/ios-sdk"
  
  # Requirements
  spec.requires_arc = true
end