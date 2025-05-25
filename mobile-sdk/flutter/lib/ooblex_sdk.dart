import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Ooblex SDK for Flutter
/// Real-time AI video processing
class OoblexSDK {
  static const MethodChannel _channel = MethodChannel('ooblex_sdk');
  static const EventChannel _eventChannel = EventChannel('ooblex_sdk/events');
  
  static final OoblexSDK _instance = OoblexSDK._internal();
  
  factory OoblexSDK() => _instance;
  
  OoblexSDK._internal() {
    _setupEventStream();
  }
  
  // Event streams
  final _connectionController = StreamController<ConnectionEvent>.broadcast();
  final _frameController = StreamController<ProcessedFrame>.broadcast();
  final _errorController = StreamController<OoblexError>.broadcast();
  final _faceController = StreamController<List<DetectedFace>>.broadcast();
  final _emotionController = StreamController<List<EmotionResult>>.broadcast();
  final _objectController = StreamController<List<DetectedObject>>.broadcast();
  final _gestureController = StreamController<List<DetectedGesture>>.broadcast();
  
  // Public streams
  Stream<ConnectionEvent> get onConnection => _connectionController.stream;
  Stream<ProcessedFrame> get onFrameProcessed => _frameController.stream;
  Stream<OoblexError> get onError => _errorController.stream;
  Stream<List<DetectedFace>> get onFaceDetected => _faceController.stream;
  Stream<List<EmotionResult>> get onEmotionDetected => _emotionController.stream;
  Stream<List<DetectedObject>> get onObjectDetected => _objectController.stream;
  Stream<List<DetectedGesture>> get onGestureDetected => _gestureController.stream;
  
  bool _isConfigured = false;
  
  /// Configure the SDK
  Future<void> configure({
    required String serverURL,
    String? apiKey,
  }) async {
    try {
      await _channel.invokeMethod('configure', {
        'serverURL': serverURL,
        'apiKey': apiKey,
      });
      _isConfigured = true;
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Configuration failed');
    }
  }
  
  /// Start video capture
  Future<void> startVideoCapture({
    CameraFacing cameraFacing = CameraFacing.front,
    VideoResolution resolution = VideoResolution.hd720p,
    int fps = 30,
  }) async {
    _checkConfigured();
    
    try {
      await _channel.invokeMethod('startVideoCapture', {
        'cameraFacing': cameraFacing.toString().split('.').last,
        'resolution': resolution.toString().split('.').last,
        'fps': fps,
      });
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to start video capture');
    }
  }
  
  /// Stop video capture
  Future<void> stopVideoCapture() async {
    try {
      await _channel.invokeMethod('stopVideoCapture');
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to stop video capture');
    }
  }
  
  /// Set AI features
  Future<void> setAIFeatures(Set<AIFeature> features) async {
    _checkConfigured();
    
    final featureStrings = features.map((f) => f.toString().split('.').last).toList();
    
    try {
      await _channel.invokeMethod('setAIFeatures', {
        'features': featureStrings,
      });
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to set AI features');
    }
  }
  
  /// Connect to Ooblex server
  Future<void> connect() async {
    _checkConfigured();
    
    try {
      await _channel.invokeMethod('connect');
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Connection failed');
    }
  }
  
  /// Disconnect from server
  Future<void> disconnect() async {
    try {
      await _channel.invokeMethod('disconnect');
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Disconnection failed');
    }
  }
  
  /// Take a snapshot
  Future<Uint8List> takeSnapshot() async {
    _checkConfigured();
    
    try {
      final result = await _channel.invokeMethod<Uint8List>('takeSnapshot');
      if (result == null) {
        throw OoblexException('No frame available for snapshot');
      }
      return result;
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to take snapshot');
    }
  }
  
  /// Set virtual background
  Future<void> setVirtualBackground(String imagePath) async {
    _checkConfigured();
    
    try {
      await _channel.invokeMethod('setVirtualBackground', {
        'imagePath': imagePath,
      });
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to set virtual background');
    }
  }
  
  /// Remove virtual background
  Future<void> removeVirtualBackground() async {
    try {
      await _channel.invokeMethod('removeVirtualBackground');
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to remove virtual background');
    }
  }
  
  /// Set beauty level
  Future<void> setBeautyLevel(double level) async {
    if (level < 0.0 || level > 1.0) {
      throw ArgumentError('Beauty level must be between 0.0 and 1.0');
    }
    
    try {
      await _channel.invokeMethod('setBeautyLevel', {
        'level': level,
      });
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to set beauty level');
    }
  }
  
  /// Set mirror mode
  Future<void> setMirrorMode(bool enabled) async {
    try {
      await _channel.invokeMethod('setMirrorMode', {
        'enabled': enabled,
      });
    } on PlatformException catch (e) {
      throw OoblexException(e.message ?? 'Failed to set mirror mode');
    }
  }
  
  /// Clean up resources
  void dispose() {
    _connectionController.close();
    _frameController.close();
    _errorController.close();
    _faceController.close();
    _emotionController.close();
    _objectController.close();
    _gestureController.close();
  }
  
  // Private methods
  
  void _checkConfigured() {
    if (!_isConfigured) {
      throw OoblexException('SDK not configured. Call configure() first.');
    }
  }
  
  void _setupEventStream() {
    _eventChannel.receiveBroadcastStream().listen((event) {
      if (event is Map) {
        final eventType = event['type'] as String?;
        final eventData = event['data'];
        
        switch (eventType) {
          case 'connected':
            _connectionController.add(ConnectionEvent.connected);
            break;
          case 'disconnected':
            _connectionController.add(ConnectionEvent.disconnected);
            break;
          case 'frameProcessed':
            if (eventData is Map) {
              _frameController.add(ProcessedFrame.fromMap(eventData));
            }
            break;
          case 'error':
            if (eventData is Map) {
              _errorController.add(OoblexError.fromMap(eventData));
            }
            break;
          case 'faceDetected':
            if (eventData is List) {
              final faces = eventData
                  .map((f) => DetectedFace.fromMap(f as Map))
                  .toList();
              _faceController.add(faces);
            }
            break;
          case 'emotionDetected':
            if (eventData is List) {
              final emotions = eventData
                  .map((e) => EmotionResult.fromMap(e as Map))
                  .toList();
              _emotionController.add(emotions);
            }
            break;
          case 'objectDetected':
            if (eventData is List) {
              final objects = eventData
                  .map((o) => DetectedObject.fromMap(o as Map))
                  .toList();
              _objectController.add(objects);
            }
            break;
          case 'gestureDetected':
            if (eventData is List) {
              final gestures = eventData
                  .map((g) => DetectedGesture.fromMap(g as Map))
                  .toList();
              _gestureController.add(gestures);
            }
            break;
        }
      }
    });
  }
}

// Enums

enum CameraFacing { front, back }

enum VideoResolution {
  vga480p,
  hd720p,
  fullHd1080p,
  uhd4k,
}

enum AIFeature {
  faceDetection,
  emotionRecognition,
  objectDetection,
  backgroundBlur,
  virtualBackground,
  beautification,
  gestureRecognition,
}

enum ConnectionEvent { connected, disconnected }

enum Emotion {
  happy,
  sad,
  angry,
  surprised,
  neutral,
  disgusted,
  fearful,
}

enum Gesture {
  thumbsUp,
  thumbsDown,
  peace,
  ok,
  wave,
  point,
  fist,
}

// Data classes

class ProcessedFrame {
  final int timestamp;
  final FrameMetadata metadata;
  
  ProcessedFrame({
    required this.timestamp,
    required this.metadata,
  });
  
  factory ProcessedFrame.fromMap(Map<dynamic, dynamic> map) {
    return ProcessedFrame(
      timestamp: map['timestamp'] as int,
      metadata: FrameMetadata.fromMap(map['metadata'] as Map),
    );
  }
}

class FrameMetadata {
  final List<DetectedFace> faces;
  final List<DetectedObject> objects;
  final List<EmotionResult> emotions;
  
  FrameMetadata({
    required this.faces,
    required this.objects,
    required this.emotions,
  });
  
  factory FrameMetadata.fromMap(Map<dynamic, dynamic> map) {
    return FrameMetadata(
      faces: (map['faces'] as List?)
          ?.map((f) => DetectedFace.fromMap(f as Map))
          .toList() ?? [],
      objects: (map['objects'] as List?)
          ?.map((o) => DetectedObject.fromMap(o as Map))
          .toList() ?? [],
      emotions: (map['emotions'] as List?)
          ?.map((e) => EmotionResult.fromMap(e as Map))
          .toList() ?? [],
    );
  }
}

class DetectedFace {
  final BoundingBox boundingBox;
  final double confidence;
  final List<FaceLandmark> landmarks;
  
  DetectedFace({
    required this.boundingBox,
    required this.confidence,
    required this.landmarks,
  });
  
  factory DetectedFace.fromMap(Map<dynamic, dynamic> map) {
    return DetectedFace(
      boundingBox: BoundingBox.fromMap(map['boundingBox'] as Map),
      confidence: (map['confidence'] as num).toDouble(),
      landmarks: (map['landmarks'] as List?)
          ?.map((l) => FaceLandmark.fromMap(l as Map))
          .toList() ?? [],
    );
  }
}

class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;
  
  BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
  
  factory BoundingBox.fromMap(Map<dynamic, dynamic> map) {
    return BoundingBox(
      x: (map['x'] as num).toDouble(),
      y: (map['y'] as num).toDouble(),
      width: (map['width'] as num).toDouble(),
      height: (map['height'] as num).toDouble(),
    );
  }
}

class FaceLandmark {
  final String type;
  final Point position;
  
  FaceLandmark({
    required this.type,
    required this.position,
  });
  
  factory FaceLandmark.fromMap(Map<dynamic, dynamic> map) {
    return FaceLandmark(
      type: map['type'] as String,
      position: Point.fromMap(map['position'] as Map),
    );
  }
}

class Point {
  final double x;
  final double y;
  
  Point({required this.x, required this.y});
  
  factory Point.fromMap(Map<dynamic, dynamic> map) {
    return Point(
      x: (map['x'] as num).toDouble(),
      y: (map['y'] as num).toDouble(),
    );
  }
}

class DetectedObject {
  final String label;
  final BoundingBox boundingBox;
  final double confidence;
  
  DetectedObject({
    required this.label,
    required this.boundingBox,
    required this.confidence,
  });
  
  factory DetectedObject.fromMap(Map<dynamic, dynamic> map) {
    return DetectedObject(
      label: map['label'] as String,
      boundingBox: BoundingBox.fromMap(map['boundingBox'] as Map),
      confidence: (map['confidence'] as num).toDouble(),
    );
  }
}

class EmotionResult {
  final int faceId;
  final Map<Emotion, double> emotions;
  
  EmotionResult({
    required this.faceId,
    required this.emotions,
  });
  
  factory EmotionResult.fromMap(Map<dynamic, dynamic> map) {
    final emotionsMap = <Emotion, double>{};
    final emotionsData = map['emotions'] as Map;
    
    emotionsData.forEach((key, value) {
      final emotion = Emotion.values.firstWhere(
        (e) => e.toString().split('.').last == key,
        orElse: () => Emotion.neutral,
      );
      emotionsMap[emotion] = (value as num).toDouble();
    });
    
    return EmotionResult(
      faceId: map['faceId'] as int,
      emotions: emotionsMap,
    );
  }
  
  Emotion get dominantEmotion {
    return emotions.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }
}

class DetectedGesture {
  final Gesture gesture;
  final double confidence;
  final BoundingBox? boundingBox;
  
  DetectedGesture({
    required this.gesture,
    required this.confidence,
    this.boundingBox,
  });
  
  factory DetectedGesture.fromMap(Map<dynamic, dynamic> map) {
    final gestureString = map['gesture'] as String;
    final gesture = Gesture.values.firstWhere(
      (g) => g.toString().split('.').last == gestureString,
      orElse: () => Gesture.fist,
    );
    
    return DetectedGesture(
      gesture: gesture,
      confidence: (map['confidence'] as num).toDouble(),
      boundingBox: map['boundingBox'] != null
          ? BoundingBox.fromMap(map['boundingBox'] as Map)
          : null,
    );
  }
}

class OoblexError {
  final String code;
  final String message;
  
  OoblexError({
    required this.code,
    required this.message,
  });
  
  factory OoblexError.fromMap(Map<dynamic, dynamic> map) {
    return OoblexError(
      code: map['code'] as String,
      message: map['message'] as String,
    );
  }
}

class OoblexException implements Exception {
  final String message;
  
  OoblexException(this.message);
  
  @override
  String toString() => 'OoblexException: $message';
}