import Foundation
import WebRTC
import CoreML
import Vision
import UIKit

/// Handles AI processing of video frames
class AIProcessor {
    
    // MARK: - Properties
    
    private var serverURL: String = ""
    private var apiKey: String?
    private var enabledFeatures: AIFeatures = .all
    private let processQueue = DispatchQueue(label: "com.ooblex.ai.processing", qos: .userInitiated)
    
    // Local ML models for edge processing
    private var faceDetector: VNDetectFaceRectanglesRequest?
    private var emotionClassifier: VNCoreMLRequest?
    
    // MARK: - Configuration
    
    func configure(serverURL: String, apiKey: String?) {
        self.serverURL = serverURL
        self.apiKey = apiKey
        setupLocalModels()
    }
    
    func setFeatures(_ features: AIFeatures) {
        self.enabledFeatures = features
    }
    
    // MARK: - Frame Processing
    
    func processFrame(_ frame: RTCVideoFrame, completion: @escaping (Result<ProcessedFrame, OoblexError>) -> Void) {
        processQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Convert RTCVideoFrame to CVPixelBuffer
            guard let pixelBuffer = self.convertFrameToPixelBuffer(frame) else {
                completion(.failure(.processingFailed("Failed to convert frame")))
                return
            }
            
            var metadata = FrameMetadata(
                timestamp: Date().timeIntervalSince1970,
                faces: [],
                objects: [],
                emotions: []
            )
            
            // Process based on enabled features
            let group = DispatchGroup()
            
            if self.enabledFeatures.contains(.faceDetection) {
                group.enter()
                self.detectFaces(in: pixelBuffer) { faces in
                    metadata.faces = faces
                    group.leave()
                }
            }
            
            if self.enabledFeatures.contains(.emotionRecognition) && !metadata.faces.isEmpty {
                group.enter()
                self.recognizeEmotions(in: pixelBuffer, faces: metadata.faces) { emotions in
                    metadata.emotions = emotions
                    group.leave()
                }
            }
            
            if self.enabledFeatures.contains(.objectDetection) {
                group.enter()
                self.detectObjects(in: pixelBuffer) { objects in
                    metadata.objects = objects
                    group.leave()
                }
            }
            
            group.notify(queue: self.processQueue) {
                // Apply visual effects if needed
                if self.enabledFeatures.contains(.backgroundBlur) ||
                   self.enabledFeatures.contains(.virtualBackground) ||
                   self.enabledFeatures.contains(.beautification) {
                    self.applyVisualEffects(to: frame, metadata: metadata) { processedFrame in
                        completion(.success(ProcessedFrame(videoFrame: processedFrame, metadata: metadata)))
                    }
                } else {
                    completion(.success(ProcessedFrame(videoFrame: frame, metadata: metadata)))
                }
            }
        }
    }
    
    // MARK: - Local ML Processing
    
    private func setupLocalModels() {
        // Set up face detection
        faceDetector = VNDetectFaceRectanglesRequest { request, error in
            // Handle results in detection methods
        }
        faceDetector?.preferBackgroundProcessing = true
        
        // Set up emotion classifier (would need actual CoreML model)
        // emotionClassifier = VNCoreMLRequest(model: emotionModel)
    }
    
    private func detectFaces(in pixelBuffer: CVPixelBuffer, completion: @escaping ([DetectedFace]) -> Void) {
        guard let faceDetector = faceDetector else {
            completion([])
            return
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([faceDetector])
            
            let faces = faceDetector.results?.compactMap { observation -> DetectedFace? in
                guard let faceObservation = observation as? VNFaceObservation else { return nil }
                
                // Convert normalized coordinates to points
                let boundingBox = CGRect(
                    x: faceObservation.boundingBox.origin.x,
                    y: 1 - faceObservation.boundingBox.origin.y - faceObservation.boundingBox.height,
                    width: faceObservation.boundingBox.width,
                    height: faceObservation.boundingBox.height
                )
                
                // Extract landmarks if available
                var landmarks: [FaceLandmark] = []
                if let faceLandmarks = faceObservation.landmarks {
                    if let leftEye = faceLandmarks.leftEye {
                        landmarks.append(FaceLandmark(
                            type: .leftEye,
                            position: leftEye.normalizedPoints.first ?? .zero
                        ))
                    }
                    if let rightEye = faceLandmarks.rightEye {
                        landmarks.append(FaceLandmark(
                            type: .rightEye,
                            position: rightEye.normalizedPoints.first ?? .zero
                        ))
                    }
                }
                
                return DetectedFace(
                    boundingBox: boundingBox,
                    confidence: faceObservation.confidence,
                    landmarks: landmarks
                )
            } ?? []
            
            completion(faces)
        } catch {
            print("Face detection error: \(error)")
            completion([])
        }
    }
    
    private func recognizeEmotions(in pixelBuffer: CVPixelBuffer, faces: [DetectedFace], completion: @escaping ([EmotionResult]) -> Void) {
        // For production, this would use a CoreML model or server API
        // For now, we'll use server-side processing
        
        guard !faces.isEmpty else {
            completion([])
            return
        }
        
        // Convert pixel buffer to base64 for server processing
        guard let imageData = pixelBufferToJPEGData(pixelBuffer) else {
            completion([])
            return
        }
        
        let url = URL(string: "\(serverURL)/api/ai/emotions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey = apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        let body: [String: Any] = [
            "image": imageData.base64EncodedString(),
            "faces": faces.map { face in
                [
                    "x": face.boundingBox.origin.x,
                    "y": face.boundingBox.origin.y,
                    "width": face.boundingBox.width,
                    "height": face.boundingBox.height
                ]
            }
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let results = json["emotions"] as? [[String: Any]] else {
                completion([])
                return
            }
            
            let emotionResults = results.enumerated().compactMap { index, result -> EmotionResult? in
                guard let emotionScores = result["scores"] as? [String: Double] else { return nil }
                
                var emotions: [Emotion: Float] = [:]
                for (emotionName, score) in emotionScores {
                    if let emotion = Emotion(rawValue: emotionName) {
                        emotions[emotion] = Float(score)
                    }
                }
                
                return EmotionResult(faceId: index, emotions: emotions)
            }
            
            completion(emotionResults)
        }.resume()
    }
    
    private func detectObjects(in pixelBuffer: CVPixelBuffer, completion: @escaping ([DetectedObject]) -> Void) {
        // Use Vision framework for object detection
        let request = VNDetectRectanglesRequest { request, error in
            guard let results = request.results as? [VNRectangleObservation] else {
                completion([])
                return
            }
            
            let objects = results.map { observation in
                DetectedObject(
                    label: "Rectangle", // Would use actual object classifier
                    boundingBox: CGRect(
                        x: observation.boundingBox.origin.x,
                        y: 1 - observation.boundingBox.origin.y - observation.boundingBox.height,
                        width: observation.boundingBox.width,
                        height: observation.boundingBox.height
                    ),
                    confidence: observation.confidence
                )
            }
            
            completion(objects)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    // MARK: - Visual Effects
    
    private func applyVisualEffects(to frame: RTCVideoFrame, metadata: FrameMetadata, completion: @escaping (RTCVideoFrame) -> Void) {
        // For production, these effects would be applied using Metal or server-side processing
        // For now, we'll pass through to server for processing
        
        guard let pixelBuffer = convertFrameToPixelBuffer(frame),
              let imageData = pixelBufferToJPEGData(pixelBuffer) else {
            completion(frame)
            return
        }
        
        let url = URL(string: "\(serverURL)/api/ai/effects")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey = apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        var effects: [String] = []
        if enabledFeatures.contains(.backgroundBlur) { effects.append("blur") }
        if enabledFeatures.contains(.virtualBackground) { effects.append("virtual_bg") }
        if enabledFeatures.contains(.beautification) { effects.append("beautify") }
        
        let body: [String: Any] = [
            "image": imageData.base64EncodedString(),
            "effects": effects,
            "faces": metadata.faces.map { face in
                [
                    "x": face.boundingBox.origin.x,
                    "y": face.boundingBox.origin.y,
                    "width": face.boundingBox.width,
                    "height": face.boundingBox.height
                ]
            }
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let processedImageBase64 = json["processed_image"] as? String,
                  let processedImageData = Data(base64Encoded: processedImageBase64),
                  let processedFrame = self?.createFrame(from: processedImageData, originalFrame: frame) else {
                completion(frame)
                return
            }
            
            completion(processedFrame)
        }.resume()
    }
    
    // MARK: - Utility Methods
    
    func convertFrameToImage(_ frame: RTCVideoFrame, completion: @escaping (Result<UIImage, OoblexError>) -> Void) {
        processQueue.async { [weak self] in
            guard let pixelBuffer = self?.convertFrameToPixelBuffer(frame) else {
                completion(.failure(.processingFailed("Failed to convert frame")))
                return
            }
            
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                completion(.failure(.processingFailed("Failed to create image")))
                return
            }
            
            let image = UIImage(cgImage: cgImage)
            completion(.success(image))
        }
    }
    
    private func convertFrameToPixelBuffer(_ frame: RTCVideoFrame) -> CVPixelBuffer? {
        // Implementation would convert RTCVideoFrame to CVPixelBuffer
        // This is a simplified version - actual implementation would handle different frame types
        return frame.buffer.pixelBuffer
    }
    
    private func pixelBufferToJPEGData(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let jpegData = context.jpegRepresentation(of: ciImage, colorSpace: colorSpace) else {
            return nil
        }
        
        return jpegData
    }
    
    private func createFrame(from imageData: Data, originalFrame: RTCVideoFrame) -> RTCVideoFrame? {
        // Convert processed image data back to RTCVideoFrame
        // This is a simplified version - actual implementation would be more complex
        guard let image = UIImage(data: imageData),
              let cgImage = image.cgImage else {
            return nil
        }
        
        // Create pixel buffer from CGImage and wrap in RTCVideoFrame
        // Implementation details would depend on the specific requirements
        return originalFrame // Placeholder
    }
}

// MARK: - Emotion Extension

extension Emotion {
    init?(rawValue: String) {
        switch rawValue.lowercased() {
        case "happy": self = .happy
        case "sad": self = .sad
        case "angry": self = .angry
        case "surprised": self = .surprised
        case "neutral": self = .neutral
        case "disgusted": self = .disgusted
        case "fearful": self = .fearful
        default: return nil
        }
    }
}