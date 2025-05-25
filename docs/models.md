# ML Model Guide

## Overview

Ooblex provides a comprehensive suite of machine learning models optimized for real-time video processing. Our models support edge computing, GPU acceleration, and blockchain verification for secure and efficient AI inference.

## Getting Started

### Available Models

| Model | Version | Purpose | Edge Support | Min FPS | Accuracy |
|-------|---------|---------|--------------|---------|----------|
| Face Detection | 3.0.1 | Detect and track faces | ✓ | 60 | 98.5% |
| Emotion Recognition | 2.1.0 | Classify facial emotions | ✓ | 30 | 94.2% |
| Object Detection | 4.2.0 | Detect 80+ object classes | ✓ | 25 | 96.1% |
| Pose Estimation | 1.5.0 | 17-point body pose | ✓ | 20 | 92.8% |
| Action Recognition | 2.0.0 | Recognize 400+ actions | ✓ | 15 | 89.7% |
| Face Recognition | 3.1.0 | Identity verification | ✓ | 25 | 99.2% |
| Scene Understanding | 1.2.0 | Scene classification | ✓ | 30 | 91.5% |
| Text Detection | 1.0.0 | OCR in video streams | ✓ | 20 | 95.3% |
| Speech Recognition | 2.5.0 | Real-time transcription | ✓ | RT | 93.6% |
| Anomaly Detection | 1.1.0 | Detect unusual behavior | ✓ | 30 | 88.9% |

### Quick Start

```python
from ooblex import ModelPipeline, EdgeDevice

# Initialize model pipeline
pipeline = ModelPipeline(
    models=['face_detection', 'emotion', 'pose_estimation'],
    device=EdgeDevice.AUTO,  # Auto-select best device
    optimization='balanced'  # balanced, speed, or accuracy
)

# Process video stream
async def process_stream(video_source):
    async for frame in video_source:
        results = await pipeline.process(frame)
        
        # Access individual model results
        faces = results.get('face_detection', [])
        emotions = results.get('emotion', {})
        poses = results.get('pose_estimation', [])
        
        # Draw results
        annotated_frame = pipeline.visualize(frame, results)
        yield annotated_frame
```

## Detailed Usage Examples

### Face Detection Model

Our face detection model uses advanced neural architecture with multi-scale detection:

```python
import numpy as np
from ooblex.models import FaceDetector

# Initialize face detector
detector = FaceDetector(
    model_path='models/face_detection_v3.tflite',
    confidence_threshold=0.7,
    nms_threshold=0.3,
    max_faces=10,
    min_face_size=(20, 20),
    device='edge_tpu'  # or 'gpu', 'cpu'
)

# Configure for different scenarios
detector.configure({
    'scenario': 'crowd',  # 'single', 'small_group', 'crowd'
    'quality': 'high',    # 'low', 'medium', 'high'
    'tracking': True,     # Enable face tracking
    'landmarks': True     # Enable facial landmarks (68 points)
})

# Process frame
def detect_faces(frame):
    # Preprocess
    input_tensor = detector.preprocess(frame)
    
    # Detect faces
    detections = detector.detect(input_tensor)
    
    # Post-process results
    faces = []
    for detection in detections:
        face = {
            'bbox': detection.bbox,  # [x, y, width, height]
            'confidence': detection.confidence,
            'landmarks': detection.landmarks,  # 68 facial points
            'tracking_id': detection.track_id,
            'quality': detection.quality_score,
            'pose': {
                'yaw': detection.yaw,
                'pitch': detection.pitch,
                'roll': detection.roll
            }
        }
        faces.append(face)
    
    return faces

# Advanced features
# Face quality assessment
quality_metrics = detector.assess_quality(face_roi)
print(f"Blur score: {quality_metrics['blur']}")
print(f"Brightness: {quality_metrics['brightness']}")
print(f"Occlusion: {quality_metrics['occlusion']}")

# Face alignment for recognition
aligned_face = detector.align_face(frame, landmarks)

# Anti-spoofing check
is_real = detector.check_liveness(face_sequence)
```

### Emotion Recognition Model

Advanced emotion recognition with temporal modeling:

```python
from ooblex.models import EmotionRecognizer

# Initialize emotion recognizer
emotion_model = EmotionRecognizer(
    model_path='models/emotion_v2.pb',
    use_temporal=True,  # Use temporal information
    window_size=5,      # Frames to consider
    device='gpu'
)

# Emotion categories
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

# Process faces for emotions
def recognize_emotions(face_roi, face_history=None):
    # Single frame inference
    emotions = emotion_model.predict(face_roi)
    
    # Temporal smoothing (if history available)
    if face_history and emotion_model.use_temporal:
        emotions = emotion_model.predict_temporal(face_history)
    
    # Get detailed results
    results = {
        'primary_emotion': emotions.argmax(),
        'emotion_scores': {
            EMOTIONS[i]: float(score) 
            for i, score in enumerate(emotions)
        },
        'valence': emotion_model.get_valence(emotions),  # Positive/negative
        'arousal': emotion_model.get_arousal(emotions),  # Calm/excited
        'confidence': float(emotions.max())
    }
    
    return results

# Micro-expression detection
micro_expressions = emotion_model.detect_micro_expressions(
    face_sequence,
    fps=30,
    min_duration=0.04,  # 40ms minimum
    max_duration=0.5    # 500ms maximum
)

# Group emotion analysis
def analyze_group_emotions(faces):
    individual_emotions = [recognize_emotions(face) for face in faces]
    
    # Aggregate metrics
    group_metrics = {
        'dominant_emotion': emotion_model.get_dominant_emotion(individual_emotions),
        'emotional_diversity': emotion_model.calculate_diversity(individual_emotions),
        'group_valence': np.mean([e['valence'] for e in individual_emotions]),
        'group_arousal': np.mean([e['arousal'] for e in individual_emotions]),
        'synchrony': emotion_model.calculate_synchrony(individual_emotions)
    }
    
    return group_metrics
```

### Object Detection Model

Multi-class object detection with custom training support:

```python
from ooblex.models import ObjectDetector

# Initialize object detector
detector = ObjectDetector(
    model_type='yolov8',  # 'yolov8', 'efficientdet', 'centernet'
    model_size='medium',  # 'nano', 'small', 'medium', 'large'
    classes='coco',       # 'coco', 'custom', or list of classes
    device='edge_tpu'
)

# Custom class configuration
custom_classes = {
    'person': {'id': 0, 'color': (255, 0, 0)},
    'vehicle': {'id': 1, 'color': (0, 255, 0)},
    'package': {'id': 2, 'color': (0, 0, 255)},
    'weapon': {'id': 3, 'color': (255, 255, 0), 'alert': True}
}

detector.set_classes(custom_classes)

# Detection with tracking
def detect_and_track(frame, previous_detections=None):
    # Run detection
    detections = detector.detect(frame)
    
    # Apply tracking
    if previous_detections:
        tracked = detector.track(detections, previous_detections)
    else:
        tracked = detections
    
    # Process each detection
    results = []
    for obj in tracked:
        result = {
            'class': obj.class_name,
            'confidence': obj.confidence,
            'bbox': obj.bbox,
            'track_id': obj.track_id,
            'velocity': obj.velocity,  # pixels/frame
            'direction': obj.direction,  # angle in degrees
            'time_visible': obj.time_visible,  # frames
            'attributes': detector.get_attributes(obj)  # color, size, etc.
        }
        
        # Special handling for alert classes
        if custom_classes[obj.class_name].get('alert'):
            result['alert_level'] = 'high'
            result['recommended_action'] = 'notify_security'
        
        results.append(result)
    
    return results

# Advanced features
# Object counting in regions
regions = [
    {'name': 'entrance', 'polygon': [(100, 100), (200, 100), (200, 200), (100, 200)]},
    {'name': 'exit', 'polygon': [(300, 100), (400, 100), (400, 200), (300, 200)]}
]

counts = detector.count_in_regions(detections, regions)

# Object relationship analysis
relationships = detector.analyze_relationships(detections)
# e.g., "person holding package", "vehicle near person"

# Custom model training
detector.train_custom_model(
    dataset_path='data/custom_objects',
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    augmentation={
        'flip': True,
        'rotate': [-10, 10],
        'brightness': [0.8, 1.2],
        'noise': 0.01
    }
)
```

### Pose Estimation Model

Real-time human pose estimation with action recognition:

```python
from ooblex.models import PoseEstimator

# Initialize pose estimator
pose_model = PoseEstimator(
    model='movenet_thunder',  # 'movenet_lightning', 'movenet_thunder', 'posenet'
    num_poses=5,             # Max number of poses to detect
    min_confidence=0.3,
    device='gpu'
)

# Keypoint indices
KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

def estimate_poses(frame):
    # Detect poses
    poses = pose_model.detect(frame)
    
    # Process each pose
    results = []
    for pose in poses:
        # Extract keypoints
        keypoints = {
            name: {
                'position': pose.keypoints[idx][:2],
                'confidence': pose.keypoints[idx][2]
            }
            for name, idx in KEYPOINTS.items()
        }
        
        # Calculate pose metrics
        metrics = {
            'pose_confidence': pose.score,
            'visibility': pose_model.calculate_visibility(keypoints),
            'symmetry': pose_model.calculate_symmetry(keypoints),
            'stability': pose_model.calculate_stability(pose.keypoints, pose.prev_keypoints)
        }
        
        # Recognize actions
        action = pose_model.recognize_action(pose.keypoints_sequence)
        
        results.append({
            'person_id': pose.track_id,
            'keypoints': keypoints,
            'metrics': metrics,
            'action': action,
            'skeleton': pose_model.get_skeleton_connections()
        })
    
    return results

# Advanced pose analysis
# Gesture recognition
gestures = pose_model.recognize_gestures(keypoints_sequence)
# e.g., "waving", "pointing", "thumbs_up"

# Fall detection
fall_detection = pose_model.detect_falls(
    poses_sequence,
    sensitivity='high',
    min_confidence=0.7
)

# Exercise form analysis
exercise_analysis = pose_model.analyze_exercise(
    exercise_type='squat',  # 'squat', 'pushup', 'plank', etc.
    poses_sequence=poses,
    provide_feedback=True
)

print(f"Rep count: {exercise_analysis['rep_count']}")
print(f"Form score: {exercise_analysis['form_score']}")
print(f"Feedback: {exercise_analysis['feedback']}")

# 3D pose estimation
if pose_model.supports_3d:
    pose_3d = pose_model.estimate_3d(frame, camera_params)
    # Returns 3D coordinates for each keypoint
```

### Action Recognition Model

Temporal action recognition with context understanding:

```python
from ooblex.models import ActionRecognizer

# Initialize action recognizer
action_model = ActionRecognizer(
    model='x3d_m',  # 'x3d_s', 'x3d_m', 'i3d', 'slowfast'
    num_frames=32,   # Temporal window
    sampling_rate=2, # Frame sampling rate
    device='gpu'
)

# Action categories (Kinetics-400 classes)
ACTION_CATEGORIES = [
    'walking', 'running', 'jumping', 'sitting', 'standing',
    'waving', 'clapping', 'dancing', 'fighting', 'falling',
    # ... 400+ action classes
]

def recognize_actions(video_clip):
    # Prepare video clip
    clip_tensor = action_model.preprocess_clip(video_clip)
    
    # Recognize action
    predictions = action_model.predict(clip_tensor)
    
    # Get top-k actions
    top_actions = action_model.get_top_k(predictions, k=5)
    
    results = {
        'primary_action': top_actions[0]['action'],
        'confidence': top_actions[0]['score'],
        'all_actions': top_actions,
        'temporal_segments': action_model.get_temporal_segments(clip_tensor),
        'context': action_model.analyze_context(video_clip)
    }
    
    return results

# Real-time action recognition
class RealtimeActionRecognizer:
    def __init__(self, model):
        self.model = model
        self.buffer = []
        self.predictions_history = []
    
    def process_frame(self, frame):
        self.buffer.append(frame)
        
        # Process when buffer is full
        if len(self.buffer) >= self.model.num_frames:
            clip = self.buffer[-self.model.num_frames:]
            action = recognize_actions(clip)
            
            # Temporal smoothing
            self.predictions_history.append(action)
            smoothed_action = self.model.smooth_predictions(
                self.predictions_history[-10:]
            )
            
            return smoothed_action
        
        return None

# Complex action understanding
# Multi-person action recognition
multi_person_actions = action_model.recognize_multi_person(
    video_clip,
    pose_keypoints,
    interaction_threshold=0.5
)
# Returns: ["person1: walking", "person2: sitting", "interaction: conversation"]

# Anomalous action detection
anomaly_score = action_model.detect_anomaly(
    video_clip,
    context='retail_store',  # Context-specific anomaly detection
    baseline_actions=['walking', 'shopping', 'browsing']
)

# Fine-grained action recognition
fine_actions = action_model.recognize_fine_grained(
    video_clip,
    domain='sports',  # 'sports', 'cooking', 'medical', etc.
    level='detailed'  # 'coarse', 'medium', 'detailed'
)
# e.g., "basketball: crossover dribble to left"
```

### Scene Understanding Model

Comprehensive scene analysis and understanding:

```python
from ooblex.models import SceneAnalyzer

# Initialize scene analyzer
scene_model = SceneAnalyzer(
    models=['scene_classification', 'semantic_segmentation', 'depth_estimation'],
    device='gpu'
)

def analyze_scene(frame):
    # Scene classification
    scene_class = scene_model.classify_scene(frame)
    
    # Semantic segmentation
    segmentation = scene_model.segment(frame)
    
    # Depth estimation
    depth_map = scene_model.estimate_depth(frame)
    
    # Comprehensive analysis
    analysis = {
        'scene_type': scene_class['primary_scene'],
        'scene_attributes': scene_class['attributes'],
        'objects_present': segmentation['object_counts'],
        'spatial_layout': scene_model.analyze_layout(segmentation, depth_map),
        'lighting_conditions': scene_model.analyze_lighting(frame),
        'camera_parameters': scene_model.estimate_camera_params(frame)
    }
    
    # Safety analysis
    safety_analysis = scene_model.analyze_safety(analysis)
    if safety_analysis['hazards']:
        analysis['safety_alerts'] = safety_analysis['hazards']
    
    return analysis

# Advanced scene understanding
# Activity zones detection
zones = scene_model.detect_activity_zones(
    video_sequence,
    min_activity_duration=30,  # frames
    clustering_method='dbscan'
)

# Crowd analysis
crowd_metrics = scene_model.analyze_crowd(frame)
print(f"Crowd density: {crowd_metrics['density']}")
print(f"Flow direction: {crowd_metrics['primary_flow_direction']}")
print(f"Congestion points: {crowd_metrics['congestion_points']}")

# Scene change detection
scene_changes = scene_model.detect_scene_changes(
    video_sequence,
    threshold=0.8,
    min_duration=10
)
```

## Configuration Options

### Model Configuration

```python
# Global model configuration
MODEL_CONFIG = {
    # Hardware acceleration
    'device': {
        'primary': 'edge_tpu',  # 'cpu', 'gpu', 'edge_tpu', 'npu'
        'fallback': 'cpu',
        'multi_device': True,
        'device_memory_limit': '4GB'
    },
    
    # Optimization settings
    'optimization': {
        'quantization': 'int8',  # 'none', 'int8', 'int16', 'dynamic'
        'pruning': 0.1,          # Sparsity level
        'batch_size': 4,
        'num_threads': 4,
        'use_xnnpack': True,     # Mobile/edge optimization
        'use_gpu_delegate': True
    },
    
    # Inference settings
    'inference': {
        'precision': 'mixed',    # 'fp32', 'fp16', 'mixed'
        'dynamic_batching': True,
        'max_batch_size': 8,
        'timeout_ms': 100,
        'enable_profiling': False
    },
    
    # Model management
    'model_cache': {
        'enabled': True,
        'max_size': '10GB',
        'eviction_policy': 'lru',
        'preload_models': ['face_detection', 'emotion']
    },
    
    # Edge computing
    'edge': {
        'enabled': True,
        'sync_interval': 300,    # seconds
        'model_updates': 'auto', # 'auto', 'manual', 'scheduled'
        'telemetry': True,
        'compression': 'zstd'
    },
    
    # Blockchain verification
    'blockchain': {
        'enabled': True,
        'network': 'polygon',
        'verify_models': True,
        'verify_results': False,  # Performance impact
        'encryption': 'aes-256'
    }
}
```

### Pipeline Configuration

```python
# Multi-model pipeline configuration
PIPELINE_CONFIG = {
    'face_pipeline': {
        'models': ['face_detection', 'face_recognition', 'emotion', 'age_gender'],
        'execution': 'sequential',  # 'sequential', 'parallel', 'conditional'
        'conditions': {
            'face_recognition': 'face_detection.confidence > 0.8',
            'emotion': 'face_detection.success == True'
        },
        'optimization': {
            'share_preprocessing': True,
            'cache_intermediate': True,
            'early_exit': True
        }
    },
    
    'security_pipeline': {
        'models': ['object_detection', 'action_recognition', 'anomaly_detection'],
        'execution': 'parallel',
        'merge_strategy': 'weighted_average',
        'weights': {
            'object_detection': 0.3,
            'action_recognition': 0.5,
            'anomaly_detection': 0.2
        },
        'alert_threshold': 0.8
    },
    
    'analytics_pipeline': {
        'models': ['face_detection', 'pose_estimation', 'scene_understanding'],
        'execution': 'conditional',
        'scheduling': 'adaptive',  # Adjust based on scene complexity
        'output_format': 'structured_json'
    }
}
```

## Best Practices

### Performance Optimization

1. **Model Selection**
```python
def select_optimal_model(requirements):
    """Select best model based on requirements"""
    models = {
        'face_detection': {
            'high_accuracy': 'retinaface',
            'balanced': 'mtcnn',
            'high_speed': 'ultraface'
        },
        'object_detection': {
            'high_accuracy': 'yolov8l',
            'balanced': 'yolov8m',
            'high_speed': 'yolov8n'
        }
    }
    
    priority = requirements.get('priority', 'balanced')
    task = requirements.get('task')
    
    return models[task][priority]
```

2. **Batch Processing**
```python
class BatchProcessor:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
    
    async def process(self, frame):
        self.queue.append(frame)
        
        if len(self.queue) >= self.batch_size:
            batch = np.array(self.queue[:self.batch_size])
            results = await self.model.predict_batch(batch)
            self.queue = self.queue[self.batch_size:]
            return results
        
        return None
```

3. **Resource Management**
```python
class ModelResourceManager:
    def __init__(self, memory_limit='4GB'):
        self.memory_limit = self._parse_memory(memory_limit)
        self.loaded_models = {}
        self.memory_usage = 0
    
    def load_model(self, model_name, model_path):
        model_size = self._get_model_size(model_path)
        
        # Check if we need to unload models
        while self.memory_usage + model_size > self.memory_limit:
            self._unload_least_used()
        
        # Load model
        model = self._load_model_impl(model_path)
        self.loaded_models[model_name] = {
            'model': model,
            'size': model_size,
            'last_used': time.time(),
            'usage_count': 0
        }
        self.memory_usage += model_size
        
        return model
```

### Accuracy Improvements

1. **Ensemble Methods**
```python
class EnsembleModel:
    def __init__(self, models, voting='soft'):
        self.models = models
        self.voting = voting
    
    def predict(self, input_data):
        predictions = []
        
        for model in self.models:
            pred = model.predict(input_data)
            predictions.append(pred)
        
        if self.voting == 'soft':
            # Average probabilities
            return np.mean(predictions, axis=0)
        else:
            # Majority voting
            return stats.mode(predictions, axis=0)[0]
```

2. **Adaptive Thresholds**
```python
class AdaptiveThreshold:
    def __init__(self, initial_threshold=0.5, window_size=100):
        self.threshold = initial_threshold
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def update(self, predictions, ground_truth):
        self.history.extend(zip(predictions, ground_truth))
        
        # Calculate optimal threshold
        if len(self.history) >= self.window_size:
            thresholds = np.linspace(0.1, 0.9, 20)
            best_f1 = 0
            best_threshold = self.threshold
            
            for t in thresholds:
                preds = [p > t for p, _ in self.history]
                truths = [gt for _, gt in self.history]
                f1 = f1_score(truths, preds)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            
            self.threshold = best_threshold
```

### Edge Deployment

1. **Model Optimization for Edge**
```python
def optimize_for_edge(model_path, target_device='edge_tpu'):
    """Optimize model for edge deployment"""
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # Edge TPU compilation
    if target_device == 'edge_tpu':
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    # Compile for Edge TPU
    if target_device == 'edge_tpu':
        import subprocess
        subprocess.run([
            'edgetpu_compiler',
            '-o', 'optimized_models',
            'model.tflite'
        ])
    
    return tflite_model
```

2. **Distributed Inference**
```python
class DistributedInference:
    def __init__(self, edge_nodes):
        self.edge_nodes = edge_nodes
        self.load_balancer = LoadBalancer(edge_nodes)
    
    async def infer(self, data, model_name):
        # Select best node
        node = self.load_balancer.select_node(model_name)
        
        try:
            # Send inference request
            result = await node.infer(data, model_name)
            return result
        except Exception as e:
            # Fallback to another node
            fallback_node = self.load_balancer.get_fallback(node)
            return await fallback_node.infer(data, model_name)
```

## Troubleshooting

### Common Issues

1. **Low FPS Performance**
```python
def diagnose_performance(model, test_frames):
    """Diagnose model performance issues"""
    
    diagnostics = {
        'preprocessing_time': [],
        'inference_time': [],
        'postprocessing_time': [],
        'total_time': []
    }
    
    for frame in test_frames:
        # Measure preprocessing
        t0 = time.time()
        preprocessed = model.preprocess(frame)
        t1 = time.time()
        diagnostics['preprocessing_time'].append(t1 - t0)
        
        # Measure inference
        result = model.infer(preprocessed)
        t2 = time.time()
        diagnostics['inference_time'].append(t2 - t1)
        
        # Measure postprocessing
        final = model.postprocess(result)
        t3 = time.time()
        diagnostics['postprocessing_time'].append(t3 - t2)
        
        diagnostics['total_time'].append(t3 - t0)
    
    # Analyze results
    print("Performance Analysis:")
    for stage, times in diagnostics.items():
        avg_time = np.mean(times) * 1000  # Convert to ms
        print(f"{stage}: {avg_time:.2f}ms (avg)")
    
    # Recommendations
    if np.mean(diagnostics['preprocessing_time']) > 0.01:
        print("⚠️ Consider optimizing preprocessing (resize, normalization)")
    if np.mean(diagnostics['inference_time']) > 0.02:
        print("⚠️ Consider using a smaller model or hardware acceleration")
```

2. **Accuracy Issues**
```python
def analyze_errors(model, test_dataset):
    """Analyze model errors and provide insights"""
    
    error_analysis = {
        'false_positives': [],
        'false_negatives': [],
        'confidence_distribution': [],
        'error_by_class': defaultdict(list)
    }
    
    for image, ground_truth in test_dataset:
        prediction = model.predict(image)
        
        # Analyze errors
        for gt, pred in zip(ground_truth, prediction):
            if gt['class'] != pred['class']:
                if pred['confidence'] > 0.5:
                    error_analysis['false_positives'].append({
                        'predicted': pred,
                        'actual': gt,
                        'image_id': image.id
                    })
                else:
                    error_analysis['false_negatives'].append({
                        'predicted': pred,
                        'actual': gt,
                        'image_id': image.id
                    })
            
            error_analysis['confidence_distribution'].append(pred['confidence'])
            error_analysis['error_by_class'][gt['class']].append(
                pred['class'] != gt['class']
            )
    
    # Generate report
    print("Error Analysis Report:")
    print(f"False Positives: {len(error_analysis['false_positives'])}")
    print(f"False Negatives: {len(error_analysis['false_negatives'])}")
    print(f"Avg Confidence: {np.mean(error_analysis['confidence_distribution']):.3f}")
    
    print("\nError Rate by Class:")
    for class_name, errors in error_analysis['error_by_class'].items():
        error_rate = sum(errors) / len(errors) * 100
        print(f"{class_name}: {error_rate:.1f}%")
```

### Model Validation

```python
class ModelValidator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def validate(self):
        """Comprehensive model validation"""
        
        results = {
            'accuracy': self.calculate_accuracy(),
            'precision': self.calculate_precision(),
            'recall': self.calculate_recall(),
            'f1_score': self.calculate_f1(),
            'inference_speed': self.measure_speed(),
            'memory_usage': self.measure_memory(),
            'robustness': self.test_robustness()
        }
        
        # Generate validation report
        self.generate_report(results)
        
        return results
    
    def test_robustness(self):
        """Test model robustness to various conditions"""
        
        robustness_tests = {
            'blur': self.test_blur_robustness(),
            'noise': self.test_noise_robustness(),
            'lighting': self.test_lighting_robustness(),
            'occlusion': self.test_occlusion_robustness()
        }
        
        return robustness_tests
```

### Support Resources

- Model Zoo: https://models.ooblex.com
- Training Guides: https://docs.ooblex.com/training
- Performance Benchmarks: https://benchmarks.ooblex.com
- Community Models: https://hub.ooblex.com