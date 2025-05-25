# Ooblex Android SDK

Real-time AI video processing SDK for Android applications.

## Features

- 🎥 **Real-time Video Processing** - Process video frames with minimal latency
- 👤 **Face Detection** - Detect faces using ML Kit
- 😊 **Emotion Recognition** - Analyze facial expressions
- 🎯 **Object Detection** - Identify objects in video frames
- 🌟 **Visual Effects** - Apply background blur, virtual backgrounds, and beautification
- 🤝 **WebRTC Integration** - Built-in WebRTC support for video streaming
- 📱 **Android Optimized** - Uses ML Kit and hardware acceleration

## Requirements

- Android API 21+ (Android 5.0 Lollipop)
- Kotlin 1.5+

## Installation

### Gradle

Add to your module's `build.gradle`:

```gradle
dependencies {
    implementation 'com.ooblex:ooblex-sdk:1.0.0'
}
```

Add to your project's `build.gradle`:

```gradle
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://maven.pkg.github.com/ooblex/ooblex-android-sdk' }
    }
}
```

### Maven

```xml
<dependency>
    <groupId>com.ooblex</groupId>
    <artifactId>ooblex-sdk</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Quick Start

### 1. Add permissions

Add to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

### 2. Initialize the SDK

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var ooblexSDK: OoblexSDK
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Get SDK instance
        ooblexSDK = OoblexSDK.getInstance(this)
        
        // Configure the SDK
        ooblexSDK.configure(
            serverURL = "https://your-ooblex-server.com",
            apiKey = "your-api-key" // Optional
        )
    }
}
```

### 3. Set up listeners

```kotlin
// Connection listener
ooblexSDK.connectionListener = object : OoblexSDK.ConnectionListener {
    override fun onConnected() {
        Log.d("OoblexSDK", "Connected to server")
    }
    
    override fun onDisconnected() {
        Log.d("OoblexSDK", "Disconnected from server")
    }
}

// Frame processing listener
ooblexSDK.frameListener = object : OoblexSDK.FrameListener {
    override fun onFrameProcessed(frame: ProcessedFrame) {
        // Handle processed frame
        runOnUiThread {
            updateUI(frame.metadata)
        }
    }
}

// Error listener
ooblexSDK.errorListener = object : OoblexSDK.ErrorListener {
    override fun onError(error: OoblexError) {
        Log.e("OoblexSDK", "Error: ${error.message}")
    }
}
```

### 4. Start video capture

```kotlin
// Request camera permission first
if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
    == PackageManager.PERMISSION_GRANTED) {
    
    ooblexSDK.startVideoCapture(
        cameraFacing = CameraFacing.FRONT,
        resolution = VideoResolution.HD_720P,
        fps = 30
    )
} else {
    ActivityCompat.requestPermissions(
        this,
        arrayOf(Manifest.permission.CAMERA),
        CAMERA_PERMISSION_REQUEST_CODE
    )
}
```

### 5. Enable AI features

```kotlin
// Enable specific features
ooblexSDK.setAIFeatures(setOf(
    AIFeature.FACE_DETECTION,
    AIFeature.EMOTION_RECOGNITION,
    AIFeature.BACKGROUND_BLUR
))

// Or enable all features
ooblexSDK.setAIFeatures(AIFeature.values().toSet())
```

### 6. Connect to server

```kotlin
ooblexSDK.connect()
```

## Advanced Usage

### Display Video

```kotlin
// Add SurfaceViewRenderer to your layout
val videoRenderer = findViewById<SurfaceViewRenderer>(R.id.videoRenderer)

// Initialize the renderer
videoRenderer.init(EglBase.create().eglBaseContext, null)
videoRenderer.setScalingType(RendererCommon.ScalingType.SCALE_ASPECT_FIT)
videoRenderer.setMirror(true)

// Attach to SDK
ooblexSDK.attachVideoRenderer(videoRenderer)
```

### Take Snapshots

```kotlin
ooblexSDK.takeSnapshot { result ->
    result.fold(
        onSuccess = { bitmap ->
            // Use the snapshot
            runOnUiThread {
                imageView.setImageBitmap(bitmap)
            }
        },
        onFailure = { error ->
            Log.e("Snapshot", "Failed: ${error.message}")
        }
    )
}
```

### Handle Face Detection

```kotlin
override fun onFrameProcessed(frame: ProcessedFrame) {
    frame.metadata.faces.forEach { face ->
        Log.d("Face", "Detected at: ${face.boundingBox}")
        Log.d("Face", "Confidence: ${face.confidence}")
        
        // Draw face rectangle on overlay
        drawFaceRectangle(face.boundingBox)
    }
}
```

### Process Emotions

```kotlin
override fun onFrameProcessed(frame: ProcessedFrame) {
    frame.metadata.emotions.forEach { emotionResult ->
        val dominantEmotion = emotionResult.emotions.maxByOrNull { it.value }
        Log.d("Emotion", "Face ${emotionResult.faceId}: ${dominantEmotion?.key}")
        
        // Update UI with emotion
        updateEmotionDisplay(emotionResult.faceId, dominantEmotion?.key)
    }
}
```

## Lifecycle Management

```kotlin
override fun onPause() {
    super.onPause()
    ooblexSDK.stopVideoCapture()
}

override fun onResume() {
    super.onResume()
    if (hasPermissions()) {
        ooblexSDK.startVideoCapture()
    }
}

override fun onDestroy() {
    super.onDestroy()
    ooblexSDK.disconnect()
    ooblexSDK.release()
}
```

## ProGuard Rules

If using ProGuard/R8, add these rules:

```proguard
-keep class com.ooblex.sdk.** { *; }
-keep class org.webrtc.** { *; }
-keep class com.google.mlkit.** { *; }
```

## Best Practices

1. **Performance**: Use HD_720P resolution for optimal performance/quality balance
2. **Battery**: Disable unused AI features to conserve battery
3. **Permissions**: Handle runtime permissions properly
4. **Lifecycle**: Release resources in onDestroy()
5. **Threading**: UI updates should be on main thread

## Error Handling

```kotlin
when (error) {
    is OoblexError.NotConfigured -> {
        // SDK not configured
        showConfigurationDialog()
    }
    is OoblexError.ConnectionFailed -> {
        // Connection failed
        showRetryDialog(error.message)
    }
    is OoblexError.ProcessingFailed -> {
        // Processing failed
        Log.e("Processing", error.message)
    }
    is OoblexError.NoFrameAvailable -> {
        // No frame available
        // Usually temporary, can be ignored
    }
    is OoblexError.InvalidAPIKey -> {
        // Invalid API key
        requestNewAPIKey()
    }
}
```

## Sample Apps

- [Basic Sample](https://github.com/ooblex/ooblex-android-sample-basic)
- [Advanced Sample](https://github.com/ooblex/ooblex-android-sample-advanced)
- [Compose Sample](https://github.com/ooblex/ooblex-android-sample-compose)

## Support

- 📧 Email: support@ooblex.com
- 📚 Documentation: https://docs.ooblex.com/android-sdk
- 💬 Discord: https://discord.gg/ooblex
- 🐛 Issues: https://github.com/ooblex/ooblex-android-sdk/issues

## License

Ooblex Android SDK is available under the MIT license. See the LICENSE file for more info.