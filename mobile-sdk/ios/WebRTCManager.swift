import Foundation
import WebRTC

/// Manages WebRTC connections and video streaming
class WebRTCManager: NSObject {
    
    // MARK: - Properties
    
    private var peerConnectionFactory: RTCPeerConnectionFactory!
    private var peerConnection: RTCPeerConnection?
    private var localVideoSource: RTCVideoSource?
    private var videoCapturer: RTCCameraVideoCapturer?
    private var localVideoTrack: RTCVideoTrack?
    private var serverURL: String = ""
    
    var currentFrame: RTCVideoFrame?
    var onFrameCaptured: ((RTCVideoFrame) -> Void)?
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        initializeWebRTC()
    }
    
    private func initializeWebRTC() {
        RTCInitializeSSL()
        
        let videoEncoderFactory = RTCDefaultVideoEncoderFactory()
        let videoDecoderFactory = RTCDefaultVideoDecoderFactory()
        
        peerConnectionFactory = RTCPeerConnectionFactory(
            encoderFactory: videoEncoderFactory,
            decoderFactory: videoDecoderFactory
        )
    }
    
    // MARK: - Configuration
    
    func configure(serverURL: String) {
        self.serverURL = serverURL
    }
    
    // MARK: - Video Capture
    
    func startCapture(cameraPosition: AVCaptureDevice.Position,
                     resolution: VideoResolution,
                     fps: Int) throws {
        localVideoSource = peerConnectionFactory.videoSource()
        videoCapturer = RTCCameraVideoCapturer(delegate: localVideoSource!)
        
        let devices = RTCCameraVideoCapturer.captureDevices()
        guard let device = devices.first(where: { $0.position == cameraPosition }) else {
            throw WebRTCError.noCameraAvailable
        }
        
        let format = selectFormat(for: device, targetWidth: resolution.width, targetHeight: resolution.height)
        let fps = selectFPS(for: format, target: fps)
        
        videoCapturer?.startCapture(with: device, format: format, fps: fps)
        
        localVideoTrack = peerConnectionFactory.videoTrack(with: localVideoSource!, trackId: "video0")
        localVideoTrack?.isEnabled = true
        
        // Set up frame capture
        localVideoSource?.adaptOutputFormat(
            toWidth: resolution.width,
            height: resolution.height,
            fps: Int32(fps)
        )
    }
    
    func stopCapture() {
        videoCapturer?.stopCapture()
        localVideoTrack?.isEnabled = false
        localVideoTrack = nil
        localVideoSource = nil
        videoCapturer = nil
    }
    
    // MARK: - Connection Management
    
    func connect(completion: @escaping (Result<Void, Error>) -> Void) {
        createPeerConnection { [weak self] result in
            switch result {
            case .success:
                self?.createOffer(completion: completion)
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    func disconnect() {
        peerConnection?.close()
        peerConnection = nil
    }
    
    // MARK: - Private Methods
    
    private func createPeerConnection(completion: @escaping (Result<Void, Error>) -> Void) {
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        config.sdpSemantics = .unifiedPlan
        config.continualGatheringPolicy = .gatherContinually
        
        let constraints = RTCMediaConstraints(
            mandatoryConstraints: nil,
            optionalConstraints: ["DtlsSrtpKeyAgreement": kRTCMediaConstraintsValueTrue]
        )
        
        guard let pc = peerConnectionFactory.peerConnection(
            with: config,
            constraints: constraints,
            delegate: self
        ) else {
            completion(.failure(WebRTCError.peerConnectionFailed))
            return
        }
        
        peerConnection = pc
        
        if let localVideoTrack = localVideoTrack {
            pc.add(localVideoTrack, streamIds: ["stream0"])
        }
        
        completion(.success(()))
    }
    
    private func createOffer(completion: @escaping (Result<Void, Error>) -> Void) {
        let constraints = RTCMediaConstraints(
            mandatoryConstraints: [
                kRTCMediaConstraintsOfferToReceiveAudio: kRTCMediaConstraintsValueFalse,
                kRTCMediaConstraintsOfferToReceiveVideo: kRTCMediaConstraintsValueTrue
            ],
            optionalConstraints: nil
        )
        
        peerConnection?.offer(for: constraints) { [weak self] sdp, error in
            guard let self = self, let sdp = sdp else {
                completion(.failure(error ?? WebRTCError.offerCreationFailed))
                return
            }
            
            self.peerConnection?.setLocalDescription(sdp) { error in
                if let error = error {
                    completion(.failure(error))
                } else {
                    self.sendOfferToServer(sdp: sdp, completion: completion)
                }
            }
        }
    }
    
    private func sendOfferToServer(sdp: RTCSessionDescription, completion: @escaping (Result<Void, Error>) -> Void) {
        // Send SDP offer to Ooblex server
        let url = URL(string: "\(serverURL)/api/webrtc/offer")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "sdp": sdp.sdp,
            "type": RTCSessionDescription.string(for: sdp.type)
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let answerSDP = json["sdp"] as? String,
                  let answerTypeString = json["type"] as? String,
                  let answerType = RTCSessionDescription.type(for: answerTypeString) else {
                completion(.failure(WebRTCError.invalidServerResponse))
                return
            }
            
            let answer = RTCSessionDescription(type: answerType, sdp: answerSDP)
            self?.peerConnection?.setRemoteDescription(answer) { error in
                if let error = error {
                    completion(.failure(error))
                } else {
                    completion(.success(()))
                }
            }
        }.resume()
    }
    
    private func selectFormat(for device: AVCaptureDevice, targetWidth: Int32, targetHeight: Int32) -> AVCaptureDevice.Format {
        let formats = RTCCameraVideoCapturer.supportedFormats(for: device)
        var selectedFormat = formats[0]
        var currentDiff = Int32.max
        
        for format in formats {
            let dimension = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let diff = abs(dimension.width - targetWidth) + abs(dimension.height - targetHeight)
            if diff < currentDiff {
                selectedFormat = format
                currentDiff = diff
            }
        }
        
        return selectedFormat
    }
    
    private func selectFPS(for format: AVCaptureDevice.Format, target: Int) -> Int {
        var maxFPS = 0
        for range in format.videoSupportedFrameRateRanges {
            maxFPS = Int(max(maxFPS, Int(range.maxFrameRate)))
        }
        return min(maxFPS, target)
    }
}

// MARK: - RTCPeerConnectionDelegate

extension WebRTCManager: RTCPeerConnectionDelegate {
    func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {
        print("Signaling state changed: \(stateChanged)")
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {
        print("Stream added")
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {
        print("Stream removed")
    }
    
    func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {
        print("Negotiation needed")
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {
        print("ICE connection state changed: \(newState)")
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {
        print("ICE gathering state changed: \(newState)")
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {
        // Send ICE candidate to server
        sendIceCandidateToServer(candidate: candidate)
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {
        print("ICE candidates removed")
    }
    
    func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {
        print("Data channel opened")
    }
    
    private func sendIceCandidateToServer(candidate: RTCIceCandidate) {
        let url = URL(string: "\(serverURL)/api/webrtc/ice")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "candidate": candidate.sdp,
            "sdpMLineIndex": candidate.sdpMLineIndex,
            "sdpMid": candidate.sdpMid ?? ""
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        URLSession.shared.dataTask(with: request).resume()
    }
}

// MARK: - RTCVideoCapturerDelegate

extension WebRTCManager: RTCVideoCapturerDelegate {
    func capturer(_ capturer: RTCVideoCapturer, didCapture frame: RTCVideoFrame) {
        currentFrame = frame
        onFrameCaptured?(frame)
    }
}

// MARK: - Errors

enum WebRTCError: Error, LocalizedError {
    case noCameraAvailable
    case peerConnectionFailed
    case offerCreationFailed
    case invalidServerResponse
    
    var errorDescription: String? {
        switch self {
        case .noCameraAvailable:
            return "No camera available"
        case .peerConnectionFailed:
            return "Failed to create peer connection"
        case .offerCreationFailed:
            return "Failed to create SDP offer"
        case .invalidServerResponse:
            return "Invalid response from server"
        }
    }
}