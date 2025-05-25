package com.ooblex.sdk

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.*
import org.webrtc.*
import java.util.concurrent.Executors

/**
 * Ooblex SDK - Real-time AI video processing for Android
 */
class OoblexSDK private constructor(private val context: Context) {
    
    companion object {
        private const val TAG = "OoblexSDK"
        
        @Volatile
        private var INSTANCE: OoblexSDK? = null
        
        /**
         * Get singleton instance of OoblexSDK
         */
        fun getInstance(context: Context): OoblexSDK {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: OoblexSDK(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    // Properties
    private var serverURL: String = ""
    private var apiKey: String? = null
    private lateinit var webRTCManager: WebRTCManager
    private lateinit var aiProcessor: AIProcessor
    private var isInitialized = false
    
    // Coroutine scope for async operations
    private val sdkScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // Listeners
    var connectionListener: ConnectionListener? = null
    var frameListener: FrameListener? = null
    var errorListener: ErrorListener? = null
    
    /**
     * Configure the SDK with server URL and optional API key
     */
    fun configure(serverURL: String, apiKey: String? = null) {
        this.serverURL = serverURL
        this.apiKey = apiKey
        
        if (!isInitialized) {
            webRTCManager = WebRTCManager(context)
            aiProcessor = AIProcessor(context)
            isInitialized = true
        }
        
        webRTCManager.configure(serverURL)
        aiProcessor.configure(serverURL, apiKey)
    }
    
    /**
     * Start video capture and processing
     */
    fun startVideoCapture(
        cameraFacing: CameraFacing = CameraFacing.FRONT,
        resolution: VideoResolution = VideoResolution.HD_720P,
        fps: Int = 30
    ) {
        if (!isInitialized) {
            errorListener?.onError(OoblexError.NotConfigured)
            return
        }
        
        webRTCManager.startCapture(cameraFacing, resolution, fps) { frame ->
            processVideoFrame(frame)
        }
    }
    
    /**
     * Stop video capture
     */
    fun stopVideoCapture() {
        webRTCManager.stopCapture()
    }
    
    /**
     * Enable/disable specific AI features
     */
    fun setAIFeatures(features: Set<AIFeature>) {
        aiProcessor.setEnabledFeatures(features)
    }
    
    /**
     * Connect to Ooblex server for real-time processing
     */
    fun connect() {
        if (!isInitialized || serverURL.isEmpty()) {
            errorListener?.onError(OoblexError.NotConfigured)
            return
        }
        
        sdkScope.launch {
            try {
                webRTCManager.connect()
                withContext(Dispatchers.Main) {
                    connectionListener?.onConnected()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    errorListener?.onError(OoblexError.ConnectionFailed(e.message ?: "Unknown error"))
                }
            }
        }
    }
    
    /**
     * Disconnect from server
     */
    fun disconnect() {
        webRTCManager.disconnect()
        connectionListener?.onDisconnected()
    }
    
    /**
     * Get the local video track for display
     */
    fun getLocalVideoTrack(): VideoTrack? {
        return webRTCManager.getLocalVideoTrack()
    }
    
    /**
     * Get the processed video track
     */
    fun getProcessedVideoTrack(): VideoTrack? {
        return webRTCManager.getProcessedVideoTrack()
    }
    
    /**
     * Take a snapshot of the current processed frame
     */
    fun takeSnapshot(callback: (Result<Bitmap>) -> Unit) {
        val currentFrame = webRTCManager.getCurrentFrame()
        if (currentFrame != null) {
            aiProcessor.frameToBitmap(currentFrame) { bitmap ->
                callback(Result.success(bitmap))
            }
        } else {
            callback(Result.failure(Exception("No frame available")))
        }
    }
    
    /**
     * Set a custom SurfaceViewRenderer for video display
     */
    fun attachVideoRenderer(renderer: SurfaceViewRenderer) {
        webRTCManager.attachRenderer(renderer)
    }
    
    /**
     * Remove video renderer
     */
    fun detachVideoRenderer(renderer: SurfaceViewRenderer) {
        webRTCManager.detachRenderer(renderer)
    }
    
    // Private methods
    
    private fun processVideoFrame(frame: VideoFrame) {
        sdkScope.launch {
            try {
                val processedFrame = aiProcessor.processFrame(frame)
                withContext(Dispatchers.Main) {
                    frameListener?.onFrameProcessed(processedFrame)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
                withContext(Dispatchers.Main) {
                    errorListener?.onError(OoblexError.ProcessingFailed(e.message ?: "Unknown error"))
                }
            }
        }
    }
    
    /**
     * Clean up resources
     */
    fun release() {
        disconnect()
        webRTCManager.release()
        aiProcessor.release()
        sdkScope.cancel()
    }
    
    // Interfaces
    
    interface ConnectionListener {
        fun onConnected()
        fun onDisconnected()
    }
    
    interface FrameListener {
        fun onFrameProcessed(frame: ProcessedFrame)
    }
    
    interface ErrorListener {
        fun onError(error: OoblexError)
    }
}

// Supporting types

enum class CameraFacing {
    FRONT, BACK
}

enum class VideoResolution(val width: Int, val height: Int) {
    VGA_480P(640, 480),
    HD_720P(1280, 720),
    FULL_HD_1080P(1920, 1080),
    UHD_4K(3840, 2160)
}

enum class AIFeature {
    FACE_DETECTION,
    EMOTION_RECOGNITION,
    OBJECT_DETECTION,
    BACKGROUND_BLUR,
    VIRTUAL_BACKGROUND,
    BEAUTIFICATION,
    GESTURE_RECOGNITION
}

sealed class OoblexError : Exception() {
    object NotConfigured : OoblexError() {
        override val message = "SDK not configured. Call configure() first."
    }
    
    data class ConnectionFailed(override val message: String) : OoblexError()
    data class ProcessingFailed(override val message: String) : OoblexError()
    
    object NoFrameAvailable : OoblexError() {
        override val message = "No video frame available"
    }
    
    object InvalidAPIKey : OoblexError() {
        override val message = "Invalid API key"
    }
}

// Data classes

data class ProcessedFrame(
    val videoFrame: VideoFrame,
    val metadata: FrameMetadata
)

data class FrameMetadata(
    val timestamp: Long,
    val faces: List<DetectedFace>,
    val objects: List<DetectedObject>,
    val emotions: List<EmotionResult>
)

data class DetectedFace(
    val boundingBox: RectF,
    val confidence: Float,
    val landmarks: List<FaceLandmark>
)

data class FaceLandmark(
    val type: LandmarkType,
    val position: android.graphics.PointF
)

enum class LandmarkType {
    LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH
}

data class DetectedObject(
    val label: String,
    val boundingBox: RectF,
    val confidence: Float
)

data class EmotionResult(
    val faceId: Int,
    val emotions: Map<Emotion, Float>
)

enum class Emotion {
    HAPPY, SAD, ANGRY, SURPRISED, NEUTRAL, DISGUSTED, FEARFUL
}