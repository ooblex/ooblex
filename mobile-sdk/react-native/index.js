import { NativeModules, NativeEventEmitter, Platform } from 'react-native';

const { OoblexSDK: NativeOoblexSDK } = NativeModules;
const eventEmitter = new NativeEventEmitter(NativeOoblexSDK);

/**
 * Ooblex SDK for React Native
 * Real-time AI video processing
 */
class OoblexSDK {
  constructor() {
    this._listeners = {};
    this._isInitialized = false;
    this._setupEventListeners();
  }

  /**
   * Configure the SDK
   * @param {Object} config - Configuration object
   * @param {string} config.serverURL - Ooblex server URL
   * @param {string} [config.apiKey] - Optional API key
   */
  async configure(config) {
    const { serverURL, apiKey } = config;
    
    if (!serverURL) {
      throw new Error('serverURL is required');
    }

    await NativeOoblexSDK.configure(serverURL, apiKey || null);
    this._isInitialized = true;
  }

  /**
   * Start video capture
   * @param {Object} options - Capture options
   * @param {string} [options.cameraFacing='front'] - 'front' or 'back'
   * @param {string} [options.resolution='720p'] - '480p', '720p', '1080p', or '4k'
   * @param {number} [options.fps=30] - Frames per second
   */
  async startVideoCapture(options = {}) {
    this._checkInitialized();
    
    const {
      cameraFacing = 'front',
      resolution = '720p',
      fps = 30
    } = options;

    await NativeOoblexSDK.startVideoCapture(cameraFacing, resolution, fps);
  }

  /**
   * Stop video capture
   */
  async stopVideoCapture() {
    await NativeOoblexSDK.stopVideoCapture();
  }

  /**
   * Set AI features
   * @param {Array<string>} features - Array of feature names
   */
  async setAIFeatures(features) {
    this._checkInitialized();
    
    const validFeatures = [
      'faceDetection',
      'emotionRecognition',
      'objectDetection',
      'backgroundBlur',
      'virtualBackground',
      'beautification',
      'gestureRecognition'
    ];

    const invalidFeatures = features.filter(f => !validFeatures.includes(f));
    if (invalidFeatures.length > 0) {
      console.warn(`Invalid features: ${invalidFeatures.join(', ')}`);
    }

    const validatedFeatures = features.filter(f => validFeatures.includes(f));
    await NativeOoblexSDK.setAIFeatures(validatedFeatures);
  }

  /**
   * Connect to Ooblex server
   */
  async connect() {
    this._checkInitialized();
    await NativeOoblexSDK.connect();
  }

  /**
   * Disconnect from server
   */
  async disconnect() {
    await NativeOoblexSDK.disconnect();
  }

  /**
   * Take a snapshot
   * @returns {Promise<string>} Base64 encoded image
   */
  async takeSnapshot() {
    this._checkInitialized();
    return await NativeOoblexSDK.takeSnapshot();
  }

  /**
   * Set virtual background image
   * @param {string} imageUri - URI of the background image
   */
  async setVirtualBackground(imageUri) {
    this._checkInitialized();
    await NativeOoblexSDK.setVirtualBackground(imageUri);
  }

  /**
   * Remove virtual background
   */
  async removeVirtualBackground() {
    await NativeOoblexSDK.removeVirtualBackground();
  }

  /**
   * Set beauty level
   * @param {number} level - Beauty level (0.0 to 1.0)
   */
  async setBeautyLevel(level) {
    if (level < 0 || level > 1) {
      throw new Error('Beauty level must be between 0.0 and 1.0');
    }
    await NativeOoblexSDK.setBeautyLevel(level);
  }

  /**
   * Enable/disable mirror mode
   * @param {boolean} enabled - Whether to enable mirror mode
   */
  async setMirrorMode(enabled) {
    await NativeOoblexSDK.setMirrorMode(enabled);
  }

  /**
   * Event listeners
   */

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  addEventListener(event, callback) {
    const validEvents = [
      'connected',
      'disconnected',
      'frameProcessed',
      'error',
      'faceDetected',
      'emotionDetected',
      'objectDetected',
      'gestureDetected'
    ];

    if (!validEvents.includes(event)) {
      console.warn(`Invalid event: ${event}`);
      return;
    }

    if (!this._listeners[event]) {
      this._listeners[event] = [];
    }

    this._listeners[event].push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  removeEventListener(event, callback) {
    if (!this._listeners[event]) {
      return;
    }

    this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
  }

  /**
   * Remove all event listeners for an event
   * @param {string} event - Event name
   */
  removeAllListeners(event) {
    if (event) {
      delete this._listeners[event];
    } else {
      this._listeners = {};
    }
  }

  // Private methods

  _checkInitialized() {
    if (!this._isInitialized) {
      throw new Error('SDK not initialized. Call configure() first.');
    }
  }

  _setupEventListeners() {
    // Connection events
    eventEmitter.addListener('OoblexSDKConnected', () => {
      this._emit('connected');
    });

    eventEmitter.addListener('OoblexSDKDisconnected', () => {
      this._emit('disconnected');
    });

    // Frame processed event
    eventEmitter.addListener('OoblexSDKFrameProcessed', (data) => {
      this._emit('frameProcessed', data);
    });

    // Error event
    eventEmitter.addListener('OoblexSDKError', (error) => {
      this._emit('error', error);
    });

    // AI detection events
    eventEmitter.addListener('OoblexSDKFaceDetected', (faces) => {
      this._emit('faceDetected', faces);
    });

    eventEmitter.addListener('OoblexSDKEmotionDetected', (emotions) => {
      this._emit('emotionDetected', emotions);
    });

    eventEmitter.addListener('OoblexSDKObjectDetected', (objects) => {
      this._emit('objectDetected', objects);
    });

    eventEmitter.addListener('OoblexSDKGestureDetected', (gestures) => {
      this._emit('gestureDetected', gestures);
    });
  }

  _emit(event, data) {
    const listeners = this._listeners[event];
    if (listeners && listeners.length > 0) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }
}

// Export singleton instance
export default new OoblexSDK();

// Export types
export const CameraFacing = {
  FRONT: 'front',
  BACK: 'back'
};

export const VideoResolution = {
  VGA_480P: '480p',
  HD_720P: '720p',
  FULL_HD_1080P: '1080p',
  UHD_4K: '4k'
};

export const AIFeature = {
  FACE_DETECTION: 'faceDetection',
  EMOTION_RECOGNITION: 'emotionRecognition',
  OBJECT_DETECTION: 'objectDetection',
  BACKGROUND_BLUR: 'backgroundBlur',
  VIRTUAL_BACKGROUND: 'virtualBackground',
  BEAUTIFICATION: 'beautification',
  GESTURE_RECOGNITION: 'gestureRecognition'
};

export const Emotion = {
  HAPPY: 'happy',
  SAD: 'sad',
  ANGRY: 'angry',
  SURPRISED: 'surprised',
  NEUTRAL: 'neutral',
  DISGUSTED: 'disgusted',
  FEARFUL: 'fearful'
};

export const Gesture = {
  THUMBS_UP: 'thumbsUp',
  THUMBS_DOWN: 'thumbsDown',
  PEACE: 'peace',
  OK: 'ok',
  WAVE: 'wave',
  POINT: 'point',
  FIST: 'fist'
};