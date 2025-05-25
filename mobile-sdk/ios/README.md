# Ooblex iOS SDK

Real-time AI video processing SDK for iOS applications.

## Features

- 🎥 **Real-time Video Processing** - Process video frames with minimal latency
- 👤 **Face Detection** - Detect faces and facial landmarks
- 😊 **Emotion Recognition** - Analyze facial expressions and emotions
- 🎯 **Object Detection** - Identify objects in video frames
- 🌟 **Visual Effects** - Apply background blur, virtual backgrounds, and beautification
- 🤝 **WebRTC Integration** - Built-in WebRTC support for video streaming
- 📱 **iOS Optimized** - Leverages CoreML and Metal for optimal performance

## Requirements

- iOS 13.0+
- Xcode 12.0+
- Swift 5.0+

## Installation

### CocoaPods

Add to your `Podfile`:

```ruby
pod 'OoblexSDK', '~> 1.0.0'
```

Then run:

```bash
pod install
```

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ooblex/ooblex-ios-sdk.git", from: "1.0.0")
]
```

### Manual Installation

1. Download the framework from [releases](https://github.com/ooblex/ooblex-ios-sdk/releases)
2. Drag `OoblexSDK.framework` into your Xcode project
3. Add to "Embedded Binaries" in your target settings

## Quick Start

### 1. Import the SDK

```swift
import OoblexSDK
```

### 2. Configure the SDK

```swift
// In your AppDelegate or initial view controller
OoblexSDK.shared.configure(
    serverURL: "https://your-ooblex-server.com",
    apiKey: "your-api-key" // Optional
)
```

### 3. Set up delegates

```swift
class ViewController: UIViewController, OoblexSDKDelegate {
    override func viewDidLoad() {
        super.viewDidLoad()
        OoblexSDK.shared.delegate = self
    }
    
    // MARK: - OoblexSDKDelegate
    
    func ooblexSDKDidConnect() {
        print("Connected to Ooblex server")
    }
    
    func ooblexSDKDidDisconnect() {
        print("Disconnected from Ooblex server")
    }
    
    func ooblexSDK(didProcessFrame frame: ProcessedFrame) {
        // Handle processed frame
        updateUI(with: frame.metadata)
    }
    
    func ooblexSDK(didEncounterError error: OoblexError) {
        print("Error: \(error.localizedDescription)")
    }
}
```

### 4. Start video capture

```swift
do {
    try OoblexSDK.shared.startVideoCapture(
        cameraPosition: .front,
        resolution: .hd720p,
        fps: 30
    )
} catch {
    print("Failed to start video capture: \(error)")
}
```

### 5. Enable AI features

```swift
// Enable specific features
OoblexSDK.shared.setAIFeatures([.faceDetection, .emotionRecognition])

// Or enable all features
OoblexSDK.shared.setAIFeatures(.all)
```

### 6. Connect to server

```swift
OoblexSDK.shared.connect { result in
    switch result {
    case .success:
        print("Connected successfully")
    case .failure(let error):
        print("Connection failed: \(error)")
    }
}
```

## Advanced Usage

### Display Processed Video

```swift
// Get the processed video track
if let videoTrack = OoblexSDK.shared.getProcessedVideoTrack() {
    // Create a video view
    let videoView = RTCEAGLVideoView(frame: view.bounds)
    videoView.delegate = self
    view.addSubview(videoView)
    
    // Attach video track
    videoTrack.add(videoView)
}
```

### Take Snapshots

```swift
OoblexSDK.shared.takeSnapshot { result in
    switch result {
    case .success(let image):
        // Use the snapshot image
        imageView.image = image
    case .failure(let error):
        print("Snapshot failed: \(error)")
    }
}
```

### Handle Face Detection Results

```swift
func ooblexSDK(didProcessFrame frame: ProcessedFrame) {
    for face in frame.metadata.faces {
        print("Face detected at: \(face.boundingBox)")
        print("Confidence: \(face.confidence)")
        
        // Draw face rectangle on UI
        drawFaceRectangle(face.boundingBox)
    }
}
```

### Process Emotions

```swift
func ooblexSDK(didProcessFrame frame: ProcessedFrame) {
    for emotionResult in frame.metadata.emotions {
        let dominantEmotion = emotionResult.emotions.max { $0.value < $1.value }
        print("Face \(emotionResult.faceId): \(dominantEmotion?.key ?? .neutral)")
    }
}
```

## Permissions

Add to your `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for video processing</string>
```

## Best Practices

1. **Performance**: For best performance, use `.hd720p` resolution on older devices
2. **Battery**: Disable unused AI features to conserve battery
3. **Privacy**: Always inform users when video processing is active
4. **Network**: Handle connection failures gracefully with retry logic

## Error Handling

```swift
func handleOoblexError(_ error: OoblexError) {
    switch error {
    case .notConfigured:
        // SDK not configured
        configureSDK()
    case .connectionFailed(let reason):
        // Handle connection failure
        showRetryDialog(reason: reason)
    case .processingFailed(let reason):
        // Handle processing failure
        logError(reason)
    case .noFrameAvailable:
        // No video frame available
        break
    case .invalidAPIKey:
        // Invalid API key
        requestNewAPIKey()
    }
}
```

## Sample Apps

Check out our sample apps:
- [Basic Integration](https://github.com/ooblex/ooblex-ios-sample-basic)
- [Advanced Features](https://github.com/ooblex/ooblex-ios-sample-advanced)
- [SwiftUI Example](https://github.com/ooblex/ooblex-ios-sample-swiftui)

## Support

- 📧 Email: support@ooblex.com
- 📚 Documentation: https://docs.ooblex.com/ios-sdk
- 💬 Discord: https://discord.gg/ooblex
- 🐛 Issues: https://github.com/ooblex/ooblex-ios-sdk/issues

## License

Ooblex iOS SDK is available under the MIT license. See the LICENSE file for more info.