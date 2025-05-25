package com.ooblex.sdk

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.webrtc.*
import java.net.URL
import java.util.concurrent.Executors

/**
 * Manages WebRTC connections and video streaming
 */
class WebRTCManager(private val context: Context) {
    
    companion object {
        private const val TAG = "WebRTCManager"
    }
    
    // WebRTC components
    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var videoCapturer: CameraVideoCapturer? = null
    private var localVideoSource: VideoSource? = null
    private var localVideoTrack: VideoTrack? = null
    private var localAudioSource: AudioSource? = null
    private var localAudioTrack: AudioTrack? = null
    
    // Configuration
    private var serverURL: String = ""
    private var currentFrame: VideoFrame? = null
    private var frameCallback: ((VideoFrame) -> Unit)? = null
    
    // Executor for camera operations
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    
    init {
        initializePeerConnectionFactory()
    }
    
    private fun initializePeerConnectionFactory() {
        val options = PeerConnectionFactory.InitializationOptions.builder(context)
            .setEnableInternalTracer(true)
            .createInitializationOptions()
        PeerConnectionFactory.initialize(options)
        
        val encoderFactory = DefaultVideoEncoderFactory(
            EglBase.create().eglBaseContext,
            true,
            true
        )
        val decoderFactory = DefaultVideoDecoderFactory(
            EglBase.create().eglBaseContext
        )
        
        peerConnectionFactory = PeerConnectionFactory.builder()
            .setVideoEncoderFactory(encoderFactory)
            .setVideoDecoderFactory(decoderFactory)
            .createPeerConnectionFactory()
    }
    
    fun configure(serverURL: String) {
        this.serverURL = serverURL
    }
    
    fun startCapture(
        cameraFacing: CameraFacing,
        resolution: VideoResolution,
        fps: Int,
        onFrameCaptured: (VideoFrame) -> Unit
    ) {
        frameCallback = onFrameCaptured
        
        val factory = peerConnectionFactory ?: return
        
        // Create video source
        localVideoSource = factory.createVideoSource(false)
        localVideoTrack = factory.createVideoTrack("video0", localVideoSource)
        localVideoTrack?.setEnabled(true)
        
        // Create audio source
        val audioConstraints = MediaConstraints()
        localAudioSource = factory.createAudioSource(audioConstraints)
        localAudioTrack = factory.createAudioTrack("audio0", localAudioSource)
        localAudioTrack?.setEnabled(true)
        
        // Set up camera capturer
        videoCapturer = createCameraCapturer(cameraFacing)
        videoCapturer?.initialize(
            SurfaceTextureHelper.create("CameraThread", EglBase.create().eglBaseContext),
            context,
            localVideoSource?.capturerObserver
        )
        
        videoCapturer?.startCapture(resolution.width, resolution.height, fps)
        
        // Set up frame observer
        localVideoSource?.capturerObserver?.let { observer ->
            localVideoSource?.setVideoProcessor(object : VideoProcessor {
                override fun onCapturerStarted(success: Boolean) {
                    Log.d(TAG, "Capturer started: $success")
                }
                
                override fun onCapturerStopped() {
                    Log.d(TAG, "Capturer stopped")
                }
                
                override fun onFrameCaptured(frame: VideoFrame) {
                    currentFrame = frame
                    frameCallback?.invoke(frame)
                }
                
                override fun setSink(sink: VideoSink?) {
                    // Not used in this implementation
                }
            })
        }
    }
    
    fun stopCapture() {
        videoCapturer?.stopCapture()
        videoCapturer?.dispose()
        videoCapturer = null
        
        localVideoTrack?.setEnabled(false)
        localVideoTrack?.dispose()
        localVideoTrack = null
        
        localVideoSource?.dispose()
        localVideoSource = null
        
        localAudioTrack?.setEnabled(false)
        localAudioTrack?.dispose()
        localAudioTrack = null
        
        localAudioSource?.dispose()
        localAudioSource = null
    }
    
    suspend fun connect() = withContext(Dispatchers.IO) {
        val factory = peerConnectionFactory ?: throw Exception("PeerConnectionFactory not initialized")
        
        val iceServers = listOf(
            PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer()
        )
        
        val rtcConfig = PeerConnection.RTCConfiguration(iceServers).apply {
            bundlePolicy = PeerConnection.BundlePolicy.MAXBUNDLE
            rtcpMuxPolicy = PeerConnection.RtcpMuxPolicy.REQUIRE
            tcpCandidatePolicy = PeerConnection.TcpCandidatePolicy.DISABLED
            candidateNetworkPolicy = PeerConnection.CandidateNetworkPolicy.ALL
            continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
        }
        
        peerConnection = factory.createPeerConnection(
            rtcConfig,
            object : PeerConnection.Observer {
                override fun onSignalingChange(state: PeerConnection.SignalingState) {
                    Log.d(TAG, "Signaling state changed: $state")
                }
                
                override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
                    Log.d(TAG, "ICE connection state changed: $state")
                }
                
                override fun onIceConnectionReceivingChange(receiving: Boolean) {
                    Log.d(TAG, "ICE connection receiving changed: $receiving")
                }
                
                override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
                    Log.d(TAG, "ICE gathering state changed: $state")
                }
                
                override fun onIceCandidate(candidate: IceCandidate) {
                    sendIceCandidateToServer(candidate)
                }
                
                override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {
                    Log.d(TAG, "ICE candidates removed")
                }
                
                override fun onAddStream(stream: MediaStream) {
                    Log.d(TAG, "Stream added")
                }
                
                override fun onRemoveStream(stream: MediaStream) {
                    Log.d(TAG, "Stream removed")
                }
                
                override fun onDataChannel(dataChannel: DataChannel) {
                    Log.d(TAG, "Data channel created")
                }
                
                override fun onRenegotiationNeeded() {
                    Log.d(TAG, "Renegotiation needed")
                }
                
                override fun onAddTrack(receiver: RtpReceiver, streams: Array<out MediaStream>) {
                    Log.d(TAG, "Track added")
                }
            }
        )
        
        // Add tracks to peer connection
        localVideoTrack?.let { track ->
            peerConnection?.addTrack(track, listOf("stream0"))
        }
        localAudioTrack?.let { track ->
            peerConnection?.addTrack(track, listOf("stream0"))
        }
        
        // Create and send offer
        createAndSendOffer()
    }
    
    fun disconnect() {
        peerConnection?.close()
        peerConnection = null
    }
    
    fun getLocalVideoTrack(): VideoTrack? = localVideoTrack
    
    fun getProcessedVideoTrack(): VideoTrack? {
        // In a real implementation, this would return the processed video track from the server
        return localVideoTrack
    }
    
    fun getCurrentFrame(): VideoFrame? = currentFrame
    
    fun attachRenderer(renderer: SurfaceViewRenderer) {
        localVideoTrack?.addSink(renderer)
    }
    
    fun detachRenderer(renderer: SurfaceViewRenderer) {
        localVideoTrack?.removeSink(renderer)
    }
    
    fun release() {
        stopCapture()
        disconnect()
        peerConnectionFactory?.dispose()
        peerConnectionFactory = null
        cameraExecutor.shutdown()
    }
    
    private fun createCameraCapturer(cameraFacing: CameraFacing): CameraVideoCapturer? {
        val enumerator = Camera2Enumerator(context)
        
        val deviceNames = enumerator.deviceNames
        
        // Try to find the requested camera
        for (deviceName in deviceNames) {
            if (cameraFacing == CameraFacing.FRONT && enumerator.isFrontFacing(deviceName)) {
                return enumerator.createCapturer(deviceName, null)
            } else if (cameraFacing == CameraFacing.BACK && enumerator.isBackFacing(deviceName)) {
                return enumerator.createCapturer(deviceName, null)
            }
        }
        
        // Fall back to first available camera
        return if (deviceNames.isNotEmpty()) {
            enumerator.createCapturer(deviceNames[0], null)
        } else {
            null
        }
    }
    
    private suspend fun createAndSendOffer() = withContext(Dispatchers.IO) {
        val constraints = MediaConstraints().apply {
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "false"))
        }
        
        peerConnection?.createOffer(object : SdpObserver {
            override fun onCreateSuccess(sessionDescription: SessionDescription) {
                peerConnection?.setLocalDescription(object : SdpObserver {
                    override fun onCreateSuccess(p0: SessionDescription?) {}
                    override fun onSetSuccess() {
                        sendOfferToServer(sessionDescription)
                    }
                    override fun onCreateFailure(p0: String?) {}
                    override fun onSetFailure(p0: String?) {}
                }, sessionDescription)
            }
            
            override fun onSetSuccess() {}
            override fun onCreateFailure(error: String?) {
                Log.e(TAG, "Failed to create offer: $error")
            }
            override fun onSetFailure(error: String?) {}
        }, constraints)
    }
    
    private fun sendOfferToServer(sdp: SessionDescription) {
        cameraExecutor.execute {
            try {
                val url = URL("$serverURL/api/webrtc/offer")
                val connection = url.openConnection() as java.net.HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true
                
                val json = JSONObject().apply {
                    put("sdp", sdp.description)
                    put("type", sdp.type.canonicalForm())
                }
                
                connection.outputStream.use { os ->
                    os.write(json.toString().toByteArray())
                }
                
                if (connection.responseCode == 200) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val responseJson = JSONObject(response)
                    
                    val answerSdp = SessionDescription(
                        SessionDescription.Type.fromCanonicalForm(responseJson.getString("type")),
                        responseJson.getString("sdp")
                    )
                    
                    peerConnection?.setRemoteDescription(object : SdpObserver {
                        override fun onCreateSuccess(p0: SessionDescription?) {}
                        override fun onSetSuccess() {
                            Log.d(TAG, "Remote description set successfully")
                        }
                        override fun onCreateFailure(p0: String?) {}
                        override fun onSetFailure(error: String?) {
                            Log.e(TAG, "Failed to set remote description: $error")
                        }
                    }, answerSdp)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send offer to server", e)
            }
        }
    }
    
    private fun sendIceCandidateToServer(candidate: IceCandidate) {
        cameraExecutor.execute {
            try {
                val url = URL("$serverURL/api/webrtc/ice")
                val connection = url.openConnection() as java.net.HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true
                
                val json = JSONObject().apply {
                    put("candidate", candidate.sdp)
                    put("sdpMLineIndex", candidate.sdpMLineIndex)
                    put("sdpMid", candidate.sdpMid)
                }
                
                connection.outputStream.use { os ->
                    os.write(json.toString().toByteArray())
                }
                
                if (connection.responseCode != 200) {
                    Log.e(TAG, "Failed to send ICE candidate: ${connection.responseCode}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send ICE candidate to server", e)
            }
        }
    }
}