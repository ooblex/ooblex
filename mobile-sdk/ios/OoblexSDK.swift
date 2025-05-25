import Foundation
import WebRTC

/// Ooblex SDK - Real-time AI video processing for iOS
public class OoblexSDK {
    
    // MARK: - Properties
    
    private var webRTCManager: WebRTCManager
    private var aiProcessor: AIProcessor
    private var serverURL: String
    private var apiKey: String?
    
    // MARK: - Singleton
    
    public static let shared = OoblexSDK()
    
    // MARK: - Delegates
    
    public weak var delegate: OoblexSDKDelegate?
    
    // MARK: - Initialization
    
    private init() {
        self.webRTCManager = WebRTCManager()
        self.aiProcessor = AIProcessor()
        self.serverURL = ""
    }
    
    /// Configure the SDK with server URL and optional API key
    public func configure(serverURL: String, apiKey: String? = nil) {
        self.serverURL = serverURL
        self.apiKey = apiKey
        
        webRTCManager.configure(serverURL: serverURL)
        aiProcessor.configure(serverURL: serverURL, apiKey: apiKey)
    }
    
    // MARK: - Public Methods
    
    /// Start video capture and processing
    public func startVideoCapture(cameraPosition: AVCaptureDevice.Position = .front,
                                 resolution: VideoResolution = .hd720p,
                                 fps: Int = 30) throws {
        try webRTCManager.startCapture(
            cameraPosition: cameraPosition,
            resolution: resolution,
            fps: fps
        )
        
        webRTCManager.onFrameCaptured = { [weak self] videoFrame in
            self?.processVideoFrame(videoFrame)
        }
    }
    
    /// Stop video capture
    public func stopVideoCapture() {
        webRTCManager.stopCapture()
    }
    
    /// Enable/disable specific AI features
    public func setAIFeatures(_ features: AIFeatures) {
        aiProcessor.setFeatures(features)
    }
    
    /// Process a single video frame
    private func processVideoFrame(_ frame: RTCVideoFrame) {
        aiProcessor.processFrame(frame) { [weak self] result in
            switch result {
            case .success(let processedFrame):
                self?.delegate?.ooblexSDK(didProcessFrame: processedFrame)
            case .failure(let error):
                self?.delegate?.ooblexSDK(didEncounterError: error)
            }
        }
    }
    
    /// Get processed video stream
    public func getProcessedVideoTrack() -> RTCVideoTrack? {
        return webRTCManager.localVideoTrack
    }
    
    /// Connect to Ooblex server for real-time processing
    public func connect(completion: @escaping (Result<Void, OoblexError>) -> Void) {
        guard !serverURL.isEmpty else {
            completion(.failure(.notConfigured))
            return
        }
        
        webRTCManager.connect { [weak self] result in
            switch result {
            case .success:
                self?.delegate?.ooblexSDKDidConnect()
                completion(.success(()))
            case .failure(let error):
                completion(.failure(.connectionFailed(error.localizedDescription)))
            }
        }
    }
    
    /// Disconnect from server
    public func disconnect() {
        webRTCManager.disconnect()
        delegate?.ooblexSDKDidDisconnect()
    }
    
    /// Take a snapshot of the current processed frame
    public func takeSnapshot(completion: @escaping (Result<UIImage, OoblexError>) -> Void) {
        guard let currentFrame = webRTCManager.currentFrame else {
            completion(.failure(.noFrameAvailable))
            return
        }
        
        aiProcessor.convertFrameToImage(currentFrame) { result in
            completion(result)
        }
    }
}

// MARK: - Supporting Types

public enum VideoResolution {
    case vga480p
    case hd720p
    case fullHd1080p
    case uhd4K
    
    var width: Int32 {
        switch self {
        case .vga480p: return 640
        case .hd720p: return 1280
        case .fullHd1080p: return 1920
        case .uhd4K: return 3840
        }
    }
    
    var height: Int32 {
        switch self {
        case .vga480p: return 480
        case .hd720p: return 720
        case .fullHd1080p: return 1080
        case .uhd4K: return 2160
        }
    }
}

public struct AIFeatures: OptionSet {
    public let rawValue: Int
    
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    public static let faceDetection = AIFeatures(rawValue: 1 << 0)
    public static let emotionRecognition = AIFeatures(rawValue: 1 << 1)
    public static let objectDetection = AIFeatures(rawValue: 1 << 2)
    public static let backgroundBlur = AIFeatures(rawValue: 1 << 3)
    public static let virtualBackground = AIFeatures(rawValue: 1 << 4)
    public static let beautification = AIFeatures(rawValue: 1 << 5)
    public static let gestureRecognition = AIFeatures(rawValue: 1 << 6)
    
    public static let all: AIFeatures = [
        .faceDetection, .emotionRecognition, .objectDetection,
        .backgroundBlur, .virtualBackground, .beautification,
        .gestureRecognition
    ]
}

public enum OoblexError: Error, LocalizedError {
    case notConfigured
    case connectionFailed(String)
    case processingFailed(String)
    case noFrameAvailable
    case invalidAPIKey
    
    public var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "SDK not configured. Call configure() first."
        case .connectionFailed(let reason):
            return "Connection failed: \(reason)"
        case .processingFailed(let reason):
            return "Processing failed: \(reason)"
        case .noFrameAvailable:
            return "No video frame available"
        case .invalidAPIKey:
            return "Invalid API key"
        }
    }
}

// MARK: - Delegate Protocol

public protocol OoblexSDKDelegate: AnyObject {
    func ooblexSDKDidConnect()
    func ooblexSDKDidDisconnect()
    func ooblexSDK(didProcessFrame frame: ProcessedFrame)
    func ooblexSDK(didEncounterError error: OoblexError)
}

// MARK: - Processed Frame

public struct ProcessedFrame {
    public let videoFrame: RTCVideoFrame
    public let metadata: FrameMetadata
}

public struct FrameMetadata {
    public let timestamp: TimeInterval
    public let faces: [DetectedFace]
    public let objects: [DetectedObject]
    public let emotions: [EmotionResult]
}

public struct DetectedFace {
    public let boundingBox: CGRect
    public let confidence: Float
    public let landmarks: [FaceLandmark]
}

public struct FaceLandmark {
    public let type: LandmarkType
    public let position: CGPoint
}

public enum LandmarkType {
    case leftEye, rightEye, nose, leftMouth, rightMouth
}

public struct DetectedObject {
    public let label: String
    public let boundingBox: CGRect
    public let confidence: Float
}

public struct EmotionResult {
    public let faceId: Int
    public let emotions: [Emotion: Float]
}

public enum Emotion {
    case happy, sad, angry, surprised, neutral, disgusted, fearful
}