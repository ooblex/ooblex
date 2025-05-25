package com.ooblex.sdk

import android.content.Context
import android.graphics.*
import android.util.Base64
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import org.webrtc.VideoFrame
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

/**
 * Handles AI processing of video frames
 */
class AIProcessor(private val context: Context) {
    
    companion object {
        private const val TAG = "AIProcessor"
    }
    
    // Configuration
    private var serverURL: String = ""
    private var apiKey: String? = null
    private var enabledFeatures = setOf<AIFeature>()
    
    // ML Kit detectors
    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .enableTracking()
        .build()
    
    private val faceDetector = FaceDetection.getClient(faceDetectorOptions)
    
    private val objectDetectorOptions = ObjectDetectorOptions.Builder()
        .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
        .enableMultipleObjects()
        .enableClassification()
        .build()
    
    private val objectDetector = ObjectDetection.getClient(objectDetectorOptions)
    
    // Processing scope
    private val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    fun configure(serverURL: String, apiKey: String?) {
        this.serverURL = serverURL
        this.apiKey = apiKey
    }
    
    fun setEnabledFeatures(features: Set<AIFeature>) {
        this.enabledFeatures = features
    }
    
    suspend fun processFrame(frame: VideoFrame): ProcessedFrame = coroutineScope {
        val bitmap = frameToBitmapSync(frame)
        val metadata = FrameMetadata(
            timestamp = System.currentTimeMillis(),
            faces = mutableListOf(),
            objects = mutableListOf(),
            emotions = mutableListOf()
        )
        
        // Run detections in parallel
        val jobs = mutableListOf<Deferred<Unit>>()
        
        if (enabledFeatures.contains(AIFeature.FACE_DETECTION)) {
            jobs.add(async {
                val faces = detectFaces(bitmap)
                metadata.faces = faces
            })
        }
        
        if (enabledFeatures.contains(AIFeature.OBJECT_DETECTION)) {
            jobs.add(async {
                val objects = detectObjects(bitmap)
                metadata.objects = objects
            })
        }
        
        // Wait for all detections to complete
        jobs.awaitAll()
        
        // Process emotions if faces were detected
        if (enabledFeatures.contains(AIFeature.EMOTION_RECOGNITION) && metadata.faces.isNotEmpty()) {
            val emotions = recognizeEmotions(bitmap, metadata.faces)
            metadata.emotions = emotions
        }
        
        // Apply visual effects if needed
        val processedBitmap = if (shouldApplyVisualEffects()) {
            applyVisualEffects(bitmap, metadata)
        } else {
            bitmap
        }
        
        // Convert processed bitmap back to VideoFrame
        val processedFrame = bitmapToVideoFrame(processedBitmap, frame)
        
        ProcessedFrame(processedFrame, metadata)
    }
    
    fun frameToBitmap(frame: VideoFrame, callback: (Bitmap) -> Unit) {
        processingScope.launch {
            val bitmap = frameToBitmapSync(frame)
            withContext(Dispatchers.Main) {
                callback(bitmap)
            }
        }
    }
    
    private fun frameToBitmapSync(frame: VideoFrame): Bitmap {
        val buffer = frame.buffer
        val width = buffer.width
        val height = buffer.height
        
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Convert I420 to ARGB
        val i420Buffer = buffer.toI420()
        val yuvBytes = ByteArray(width * height * 3 / 2)
        
        // Copy Y plane
        i420Buffer.dataY.get(yuvBytes, 0, width * height)
        
        // Copy U and V planes
        val uvPixelStride = 2
        for (i in 0 until height / 2) {
            i420Buffer.dataU.position(i * i420Buffer.strideU)
            i420Buffer.dataU.get(yuvBytes, width * height + i * width / 2, width / 2)
            
            i420Buffer.dataV.position(i * i420Buffer.strideV)
            i420Buffer.dataV.get(yuvBytes, width * height * 5 / 4 + i * width / 2, width / 2)
        }
        
        // Convert YUV to RGB using Android's built-in converter
        val yuvImage = YuvImage(yuvBytes, ImageFormat.NV21, width, height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, outputStream)
        val jpegBytes = outputStream.toByteArray()
        
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }
    
    private suspend fun detectFaces(bitmap: Bitmap): List<DetectedFace> = suspendCoroutine { cont ->
        val image = InputImage.fromBitmap(bitmap, 0)
        
        faceDetector.process(image)
            .addOnSuccessListener { faces ->
                val detectedFaces = faces.map { face ->
                    val landmarks = mutableListOf<com.ooblex.sdk.FaceLandmark>()
                    
                    face.getLandmark(FaceLandmark.LEFT_EYE)?.let { landmark ->
                        landmarks.add(
                            com.ooblex.sdk.FaceLandmark(
                                LandmarkType.LEFT_EYE,
                                PointF(landmark.position.x, landmark.position.y)
                            )
                        )
                    }
                    
                    face.getLandmark(FaceLandmark.RIGHT_EYE)?.let { landmark ->
                        landmarks.add(
                            com.ooblex.sdk.FaceLandmark(
                                LandmarkType.RIGHT_EYE,
                                PointF(landmark.position.x, landmark.position.y)
                            )
                        )
                    }
                    
                    face.getLandmark(FaceLandmark.NOSE_BASE)?.let { landmark ->
                        landmarks.add(
                            com.ooblex.sdk.FaceLandmark(
                                LandmarkType.NOSE,
                                PointF(landmark.position.x, landmark.position.y)
                            )
                        )
                    }
                    
                    DetectedFace(
                        boundingBox = RectF(face.boundingBox),
                        confidence = 0.95f, // ML Kit doesn't provide confidence
                        landmarks = landmarks
                    )
                }
                cont.resume(detectedFaces)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed", e)
                cont.resume(emptyList())
            }
    }
    
    private suspend fun detectObjects(bitmap: Bitmap): List<DetectedObject> = suspendCoroutine { cont ->
        val image = InputImage.fromBitmap(bitmap, 0)
        
        objectDetector.process(image)
            .addOnSuccessListener { objects ->
                val detectedObjects = objects.map { obj ->
                    DetectedObject(
                        label = obj.labels.firstOrNull()?.text ?: "Unknown",
                        boundingBox = RectF(obj.boundingBox),
                        confidence = obj.labels.firstOrNull()?.confidence ?: 0f
                    )
                }
                cont.resume(detectedObjects)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Object detection failed", e)
                cont.resume(emptyList())
            }
    }
    
    private suspend fun recognizeEmotions(bitmap: Bitmap, faces: List<DetectedFace>): List<EmotionResult> {
        if (serverURL.isEmpty() || faces.isEmpty()) {
            return emptyList()
        }
        
        return withContext(Dispatchers.IO) {
            try {
                val url = URL("$serverURL/api/ai/emotions")
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                apiKey?.let {
                    connection.setRequestProperty("Authorization", "Bearer $it")
                }
                connection.doOutput = true
                
                // Convert bitmap to base64
                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                val imageBase64 = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
                
                // Create request body
                val facesArray = JSONArray()
                faces.forEach { face ->
                    facesArray.put(JSONObject().apply {
                        put("x", face.boundingBox.left)
                        put("y", face.boundingBox.top)
                        put("width", face.boundingBox.width())
                        put("height", face.boundingBox.height())
                    })
                }
                
                val requestBody = JSONObject().apply {
                    put("image", imageBase64)
                    put("faces", facesArray)
                }
                
                connection.outputStream.use { os ->
                    os.write(requestBody.toString().toByteArray())
                }
                
                if (connection.responseCode == 200) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val responseJson = JSONObject(response)
                    val emotionsArray = responseJson.getJSONArray("emotions")
                    
                    val results = mutableListOf<EmotionResult>()
                    for (i in 0 until emotionsArray.length()) {
                        val emotionJson = emotionsArray.getJSONObject(i)
                        val scores = emotionJson.getJSONObject("scores")
                        
                        val emotions = mutableMapOf<Emotion, Float>()
                        Emotion.values().forEach { emotion ->
                            val score = scores.optDouble(emotion.name.lowercase(), 0.0)
                            emotions[emotion] = score.toFloat()
                        }
                        
                        results.add(EmotionResult(i, emotions))
                    }
                    
                    results
                } else {
                    Log.e(TAG, "Emotion recognition failed: ${connection.responseCode}")
                    emptyList()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to recognize emotions", e)
                emptyList()
            }
        }
    }
    
    private fun shouldApplyVisualEffects(): Boolean {
        return enabledFeatures.any {
            it in setOf(
                AIFeature.BACKGROUND_BLUR,
                AIFeature.VIRTUAL_BACKGROUND,
                AIFeature.BEAUTIFICATION
            )
        }
    }
    
    private suspend fun applyVisualEffects(bitmap: Bitmap, metadata: FrameMetadata): Bitmap {
        if (serverURL.isEmpty()) {
            return bitmap
        }
        
        return withContext(Dispatchers.IO) {
            try {
                val url = URL("$serverURL/api/ai/effects")
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                apiKey?.let {
                    connection.setRequestProperty("Authorization", "Bearer $it")
                }
                connection.doOutput = true
                
                // Convert bitmap to base64
                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                val imageBase64 = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
                
                // Determine which effects to apply
                val effects = JSONArray()
                if (enabledFeatures.contains(AIFeature.BACKGROUND_BLUR)) effects.put("blur")
                if (enabledFeatures.contains(AIFeature.VIRTUAL_BACKGROUND)) effects.put("virtual_bg")
                if (enabledFeatures.contains(AIFeature.BEAUTIFICATION)) effects.put("beautify")
                
                // Create faces array
                val facesArray = JSONArray()
                metadata.faces.forEach { face ->
                    facesArray.put(JSONObject().apply {
                        put("x", face.boundingBox.left)
                        put("y", face.boundingBox.top)
                        put("width", face.boundingBox.width())
                        put("height", face.boundingBox.height())
                    })
                }
                
                val requestBody = JSONObject().apply {
                    put("image", imageBase64)
                    put("effects", effects)
                    put("faces", facesArray)
                }
                
                connection.outputStream.use { os ->
                    os.write(requestBody.toString().toByteArray())
                }
                
                if (connection.responseCode == 200) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val responseJson = JSONObject(response)
                    val processedImageBase64 = responseJson.getString("processed_image")
                    
                    val imageBytes = Base64.decode(processedImageBase64, Base64.NO_WRAP)
                    BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size) ?: bitmap
                } else {
                    Log.e(TAG, "Visual effects failed: ${connection.responseCode}")
                    bitmap
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to apply visual effects", e)
                bitmap
            }
        }
    }
    
    private fun bitmapToVideoFrame(bitmap: Bitmap, originalFrame: VideoFrame): VideoFrame {
        // In a real implementation, this would convert the bitmap back to a VideoFrame
        // For now, return the original frame
        return originalFrame
    }
    
    fun release() {
        processingScope.cancel()
        faceDetector.close()
        objectDetector.close()
    }
}