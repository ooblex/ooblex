# React Native Ooblex SDK

Real-time AI video processing SDK for React Native applications.

## Features

- 🎥 **Real-time Video Processing** - Process video with minimal latency
- 👤 **Face Detection** - Detect faces and facial landmarks
- 😊 **Emotion Recognition** - Analyze facial expressions
- 🎯 **Object Detection** - Identify objects in video
- 🌟 **Visual Effects** - Background blur, virtual backgrounds, beautification
- 🤝 **WebRTC Support** - Built on WebRTC for low-latency streaming
- 📱 **Cross-Platform** - Works on both iOS and Android

## Installation

```bash
npm install react-native-ooblex-sdk
# or
yarn add react-native-ooblex-sdk
```

### iOS Setup

```bash
cd ios && pod install
```

Add to your `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for video processing</string>
```

### Android Setup

Add to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

## Quick Start

```javascript
import React, { useEffect, useState } from 'react';
import { View, Button, Text } from 'react-native';
import OoblexSDK, { AIFeature, CameraFacing } from 'react-native-ooblex-sdk';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState([]);

  useEffect(() => {
    // Configure SDK
    OoblexSDK.configure({
      serverURL: 'https://your-ooblex-server.com',
      apiKey: 'your-api-key' // optional
    });

    // Set up event listeners
    OoblexSDK.addEventListener('connected', () => {
      setIsConnected(true);
      console.log('Connected to Ooblex server');
    });

    OoblexSDK.addEventListener('disconnected', () => {
      setIsConnected(false);
      console.log('Disconnected from Ooblex server');
    });

    OoblexSDK.addEventListener('faceDetected', (faces) => {
      setDetectedFaces(faces);
    });

    OoblexSDK.addEventListener('error', (error) => {
      console.error('Ooblex error:', error);
    });

    return () => {
      OoblexSDK.removeAllListeners();
    };
  }, []);

  const startProcessing = async () => {
    try {
      // Enable AI features
      await OoblexSDK.setAIFeatures([
        AIFeature.FACE_DETECTION,
        AIFeature.EMOTION_RECOGNITION,
        AIFeature.BACKGROUND_BLUR
      ]);

      // Start video capture
      await OoblexSDK.startVideoCapture({
        cameraFacing: CameraFacing.FRONT,
        resolution: '720p',
        fps: 30
      });

      // Connect to server
      await OoblexSDK.connect();
    } catch (error) {
      console.error('Failed to start:', error);
    }
  };

  const stopProcessing = async () => {
    await OoblexSDK.disconnect();
    await OoblexSDK.stopVideoCapture();
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', padding: 20 }}>
      <Text>Connected: {isConnected ? 'Yes' : 'No'}</Text>
      <Text>Faces detected: {detectedFaces.length}</Text>
      
      <Button 
        title="Start Processing" 
        onPress={startProcessing}
        disabled={isConnected}
      />
      
      <Button 
        title="Stop Processing" 
        onPress={stopProcessing}
        disabled={!isConnected}
      />
    </View>
  );
}

export default App;
```

## API Reference

### Configuration

```javascript
await OoblexSDK.configure({
  serverURL: 'https://your-server.com',
  apiKey: 'optional-api-key'
});
```

### Video Capture

```javascript
// Start capture
await OoblexSDK.startVideoCapture({
  cameraFacing: 'front', // or 'back'
  resolution: '720p',    // '480p', '720p', '1080p', '4k'
  fps: 30               // frames per second
});

// Stop capture
await OoblexSDK.stopVideoCapture();
```

### AI Features

```javascript
// Enable features
await OoblexSDK.setAIFeatures([
  'faceDetection',
  'emotionRecognition',
  'objectDetection',
  'backgroundBlur',
  'virtualBackground',
  'beautification',
  'gestureRecognition'
]);
```

### Connection

```javascript
// Connect to server
await OoblexSDK.connect();

// Disconnect
await OoblexSDK.disconnect();
```

### Snapshots

```javascript
// Take a snapshot (returns base64 image)
const imageBase64 = await OoblexSDK.takeSnapshot();
```

### Visual Effects

```javascript
// Set virtual background
await OoblexSDK.setVirtualBackground('path/to/image.jpg');

// Remove virtual background
await OoblexSDK.removeVirtualBackground();

// Set beauty level (0.0 to 1.0)
await OoblexSDK.setBeautyLevel(0.5);

// Enable/disable mirror mode
await OoblexSDK.setMirrorMode(true);
```

## Event Handling

### Available Events

- `connected` - Connected to server
- `disconnected` - Disconnected from server
- `frameProcessed` - Frame has been processed
- `error` - An error occurred
- `faceDetected` - Faces detected in frame
- `emotionDetected` - Emotions detected
- `objectDetected` - Objects detected
- `gestureDetected` - Gestures detected

### Event Listener Example

```javascript
// Add listener
const handleFaceDetected = (faces) => {
  faces.forEach(face => {
    console.log('Face bounds:', face.boundingBox);
    console.log('Confidence:', face.confidence);
  });
};

OoblexSDK.addEventListener('faceDetected', handleFaceDetected);

// Remove listener
OoblexSDK.removeEventListener('faceDetected', handleFaceDetected);

// Remove all listeners for an event
OoblexSDK.removeAllListeners('faceDetected');

// Remove all listeners
OoblexSDK.removeAllListeners();
```

## With React Native Camera

```javascript
import { RNCamera } from 'react-native-camera';
import OoblexVideoView from 'react-native-ooblex-sdk/OoblexVideoView';

function CameraScreen() {
  return (
    <View style={{ flex: 1 }}>
      <OoblexVideoView 
        style={{ flex: 1 }}
        mirror={true}
        onFrameProcessed={(metadata) => {
          console.log('Frame metadata:', metadata);
        }}
      />
    </View>
  );
}
```

## TypeScript Support

The SDK includes TypeScript definitions:

```typescript
import OoblexSDK, { 
  AIFeature, 
  CameraFacing,
  VideoResolution,
  Emotion,
  Gesture
} from 'react-native-ooblex-sdk';

interface Face {
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  landmarks?: FaceLandmark[];
}

interface EmotionResult {
  faceId: number;
  emotions: Record<Emotion, number>;
}
```

## Error Handling

```javascript
try {
  await OoblexSDK.startVideoCapture();
} catch (error) {
  switch (error.code) {
    case 'NOT_CONFIGURED':
      console.error('SDK not configured');
      break;
    case 'CAMERA_PERMISSION_DENIED':
      console.error('Camera permission denied');
      break;
    case 'CONNECTION_FAILED':
      console.error('Connection failed:', error.message);
      break;
    default:
      console.error('Unknown error:', error);
  }
}
```

## Best Practices

1. **Permissions**: Always check and request permissions before starting
2. **Lifecycle**: Stop capture in `componentWillUnmount` or cleanup
3. **Performance**: Use 720p for optimal performance/quality balance
4. **Battery**: Disable unused AI features to save battery
5. **Error Handling**: Always wrap SDK calls in try-catch

## Examples

### Face Detection with Overlay

```javascript
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Rect } from 'react-native-svg';

function FaceOverlay({ faces, screenDimensions }) {
  return (
    <Svg style={StyleSheet.absoluteFillObject}>
      {faces.map((face, index) => (
        <Rect
          key={index}
          x={face.boundingBox.x * screenDimensions.width}
          y={face.boundingBox.y * screenDimensions.height}
          width={face.boundingBox.width * screenDimensions.width}
          height={face.boundingBox.height * screenDimensions.height}
          stroke="green"
          strokeWidth="2"
          fill="transparent"
        />
      ))}
    </Svg>
  );
}
```

### Emotion Display

```javascript
function EmotionDisplay({ emotions }) {
  const getDominantEmotion = (emotionData) => {
    return Object.entries(emotionData.emotions)
      .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  };

  return (
    <View>
      {emotions.map((emotion, index) => (
        <Text key={index}>
          Face {emotion.faceId}: {getDominantEmotion(emotion)}
        </Text>
      ))}
    </View>
  );
}
```

## Troubleshooting

### iOS Issues

1. **Build fails**: Make sure to run `pod install`
2. **Camera not working**: Check Info.plist permissions
3. **Crash on startup**: Ensure minimum iOS 13.0

### Android Issues

1. **Build fails**: Check minSdkVersion is at least 21
2. **Camera black screen**: Check camera permissions
3. **WebRTC issues**: Ensure ProGuard rules are added

## Support

- 📧 Email: support@ooblex.com
- 📚 Documentation: https://docs.ooblex.com/react-native-sdk
- 💬 Discord: https://discord.gg/ooblex
- 🐛 Issues: https://github.com/ooblex/react-native-ooblex-sdk/issues

## License

MIT License - see LICENSE file for details