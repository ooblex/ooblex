# Ooblex SDK for Flutter

Real-time AI video processing SDK for Flutter applications.

## Features

- 🎥 **Real-time Video Processing** - Process video with minimal latency
- 👤 **Face Detection** - Detect faces and facial landmarks
- 😊 **Emotion Recognition** - Analyze facial expressions
- 🎯 **Object Detection** - Identify objects in video
- 🌟 **Visual Effects** - Background blur, virtual backgrounds, beautification
- 🤝 **WebRTC Support** - Built on WebRTC for low-latency streaming
- 📱 **Cross-Platform** - Works on iOS, Android, Web (coming soon)

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  ooblex_sdk: ^1.0.0
```

### iOS Setup

Add to your `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for video processing</string>
```

Minimum iOS version: 11.0

### Android Setup

Add to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Minimum Android SDK: 21

## Quick Start

```dart
import 'package:flutter/material.dart';
import 'package:ooblex_sdk/ooblex_sdk.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _ooblexSDK = OoblexSDK();
  bool _isConnected = false;
  List<DetectedFace> _faces = [];
  
  @override
  void initState() {
    super.initState();
    _initializeSDK();
  }
  
  Future<void> _initializeSDK() async {
    // Configure SDK
    await _ooblexSDK.configure(
      serverURL: 'https://your-ooblex-server.com',
      apiKey: 'your-api-key', // optional
    );
    
    // Listen to events
    _ooblexSDK.onConnection.listen((event) {
      setState(() {
        _isConnected = event == ConnectionEvent.connected;
      });
    });
    
    _ooblexSDK.onFaceDetected.listen((faces) {
      setState(() {
        _faces = faces;
      });
    });
    
    _ooblexSDK.onError.listen((error) {
      print('Ooblex error: ${error.message}');
    });
  }
  
  Future<void> _startProcessing() async {
    try {
      // Enable AI features
      await _ooblexSDK.setAIFeatures({
        AIFeature.faceDetection,
        AIFeature.emotionRecognition,
        AIFeature.backgroundBlur,
      });
      
      // Start video capture
      await _ooblexSDK.startVideoCapture(
        cameraFacing: CameraFacing.front,
        resolution: VideoResolution.hd720p,
        fps: 30,
      );
      
      // Connect to server
      await _ooblexSDK.connect();
    } catch (e) {
      print('Failed to start: $e');
    }
  }
  
  Future<void> _stopProcessing() async {
    await _ooblexSDK.disconnect();
    await _ooblexSDK.stopVideoCapture();
  }
  
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Ooblex SDK Demo'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Connected: $_isConnected'),
              Text('Faces detected: ${_faces.length}'),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _isConnected ? null : _startProcessing,
                child: Text('Start Processing'),
              ),
              ElevatedButton(
                onPressed: _isConnected ? _stopProcessing : null,
                child: Text('Stop Processing'),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _ooblexSDK.dispose();
    super.dispose();
  }
}
```

## Video Display Widget

```dart
import 'package:flutter/material.dart';
import 'package:ooblex_sdk/ooblex_sdk.dart';

class OoblexVideoView extends StatefulWidget {
  final bool mirror;
  final Function(ProcessedFrame)? onFrameProcessed;
  
  const OoblexVideoView({
    Key? key,
    this.mirror = true,
    this.onFrameProcessed,
  }) : super(key: key);
  
  @override
  _OoblexVideoViewState createState() => _OoblexVideoViewState();
}

class _OoblexVideoViewState extends State<OoblexVideoView> {
  final _ooblexSDK = OoblexSDK();
  
  @override
  void initState() {
    super.initState();
    
    if (widget.onFrameProcessed != null) {
      _ooblexSDK.onFrameProcessed.listen(widget.onFrameProcessed!);
    }
    
    _ooblexSDK.setMirrorMode(widget.mirror);
  }
  
  @override
  Widget build(BuildContext context) {
    // Platform-specific video view implementation
    // This would use platform channels to display native video
    return Container(
      color: Colors.black,
      child: Center(
        child: Text(
          'Video View',
          style: TextStyle(color: Colors.white),
        ),
      ),
    );
  }
}
```

## API Reference

### Configuration

```dart
await ooblexSDK.configure(
  serverURL: 'https://your-server.com',
  apiKey: 'optional-api-key',
);
```

### Video Capture

```dart
// Start capture
await ooblexSDK.startVideoCapture(
  cameraFacing: CameraFacing.front, // or .back
  resolution: VideoResolution.hd720p, // .vga480p, .fullHd1080p, .uhd4k
  fps: 30,
);

// Stop capture
await ooblexSDK.stopVideoCapture();
```

### AI Features

```dart
await ooblexSDK.setAIFeatures({
  AIFeature.faceDetection,
  AIFeature.emotionRecognition,
  AIFeature.objectDetection,
  AIFeature.backgroundBlur,
  AIFeature.virtualBackground,
  AIFeature.beautification,
  AIFeature.gestureRecognition,
});
```

### Connection

```dart
// Connect
await ooblexSDK.connect();

// Disconnect
await ooblexSDK.disconnect();
```

### Snapshots

```dart
// Take snapshot (returns Uint8List)
final imageBytes = await ooblexSDK.takeSnapshot();

// Display snapshot
Image.memory(imageBytes);
```

### Visual Effects

```dart
// Virtual background
await ooblexSDK.setVirtualBackground('assets/background.jpg');
await ooblexSDK.removeVirtualBackground();

// Beauty filter
await ooblexSDK.setBeautyLevel(0.5); // 0.0 to 1.0

// Mirror mode
await ooblexSDK.setMirrorMode(true);
```

## Event Streams

### Listen to Events

```dart
// Connection events
ooblexSDK.onConnection.listen((event) {
  if (event == ConnectionEvent.connected) {
    print('Connected');
  } else {
    print('Disconnected');
  }
});

// Face detection
ooblexSDK.onFaceDetected.listen((faces) {
  for (var face in faces) {
    print('Face at: ${face.boundingBox}');
    print('Confidence: ${face.confidence}');
  }
});

// Emotion detection
ooblexSDK.onEmotionDetected.listen((emotions) {
  for (var emotion in emotions) {
    print('Face ${emotion.faceId}: ${emotion.dominantEmotion}');
  }
});

// Error handling
ooblexSDK.onError.listen((error) {
  print('Error ${error.code}: ${error.message}');
});
```

## Advanced Examples

### Face Detection Overlay

```dart
class FaceOverlay extends StatelessWidget {
  final List<DetectedFace> faces;
  final Size videoSize;
  
  const FaceOverlay({
    required this.faces,
    required this.videoSize,
  });
  
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: FacePainter(faces, videoSize),
      child: Container(),
    );
  }
}

class FacePainter extends CustomPainter {
  final List<DetectedFace> faces;
  final Size videoSize;
  
  FacePainter(this.faces, this.videoSize);
  
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.green;
    
    for (var face in faces) {
      final rect = Rect.fromLTWH(
        face.boundingBox.x * size.width,
        face.boundingBox.y * size.height,
        face.boundingBox.width * size.width,
        face.boundingBox.height * size.height,
      );
      
      canvas.drawRect(rect, paint);
    }
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
```

### Emotion Display

```dart
class EmotionDisplay extends StatelessWidget {
  final List<EmotionResult> emotions;
  
  const EmotionDisplay({required this.emotions});
  
  @override
  Widget build(BuildContext context) {
    return Column(
      children: emotions.map((emotion) {
        final dominant = emotion.dominantEmotion;
        final confidence = emotion.emotions[dominant] ?? 0;
        
        return ListTile(
          leading: _getEmotionIcon(dominant),
          title: Text('Face ${emotion.faceId}'),
          subtitle: Text('${dominant.name}: ${(confidence * 100).toStringAsFixed(1)}%'),
        );
      }).toList(),
    );
  }
  
  Widget _getEmotionIcon(Emotion emotion) {
    IconData icon;
    Color color;
    
    switch (emotion) {
      case Emotion.happy:
        icon = Icons.sentiment_very_satisfied;
        color = Colors.green;
        break;
      case Emotion.sad:
        icon = Icons.sentiment_very_dissatisfied;
        color = Colors.blue;
        break;
      case Emotion.angry:
        icon = Icons.sentiment_very_dissatisfied;
        color = Colors.red;
        break;
      case Emotion.surprised:
        icon = Icons.sentiment_satisfied;
        color = Colors.orange;
        break;
      default:
        icon = Icons.sentiment_neutral;
        color = Colors.grey;
    }
    
    return Icon(icon, color: color, size: 40);
  }
}
```

### Gesture Recognition

```dart
ooblexSDK.onGestureDetected.listen((gestures) {
  for (var gesture in gestures) {
    switch (gesture.gesture) {
      case Gesture.thumbsUp:
        _showSnackBar('👍 Thumbs up!');
        break;
      case Gesture.peace:
        _showSnackBar('✌️ Peace!');
        break;
      case Gesture.wave:
        _showSnackBar('👋 Wave detected!');
        break;
      default:
        break;
    }
  }
});
```

## Performance Tips

1. **Resolution**: Use HD (720p) for best performance/quality balance
2. **Features**: Enable only needed AI features to save resources
3. **Lifecycle**: Properly manage SDK lifecycle with widget lifecycle
4. **Disposal**: Always call `dispose()` when done

## Error Handling

```dart
try {
  await ooblexSDK.startVideoCapture();
} on OoblexException catch (e) {
  // Handle Ooblex-specific errors
  showDialog(
    context: context,
    builder: (_) => AlertDialog(
      title: Text('Error'),
      content: Text(e.message),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text('OK'),
        ),
      ],
    ),
  );
} catch (e) {
  // Handle other errors
  print('Unexpected error: $e');
}
```

## Platform-Specific Notes

### iOS
- Minimum iOS 11.0
- Camera permission required
- Add to Info.plist

### Android
- Minimum SDK 21
- Camera permission required
- ProGuard rules included automatically

### Web (Coming Soon)
- Will use WebRTC APIs
- Requires HTTPS

## Sample Apps

- [Basic Example](https://github.com/ooblex/flutter-ooblex-sdk/tree/main/example)
- [Advanced Demo](https://github.com/ooblex/flutter-ooblex-demo)
- [Production App](https://github.com/ooblex/flutter-ooblex-showcase)

## Support

- 📧 Email: support@ooblex.com
- 📚 Documentation: https://docs.ooblex.com/flutter-sdk
- 💬 Discord: https://discord.gg/ooblex
- 🐛 Issues: https://github.com/ooblex/flutter-ooblex-sdk/issues

## License

MIT License - see LICENSE file for details