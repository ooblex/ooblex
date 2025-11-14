#!/usr/bin/env node

/**
 * NinjaSDK Audio Ingestion Service for Ooblex
 *
 * This service provides a P2P WebRTC alternative to Janus Gateway for low-latency
 * audio ingestion. It uses the NinjaSDK (vdo.ninja) for serverless P2P connections.
 *
 * Features:
 * - Real-time audio capture via WebRTC P2P
 * - PCM16 audio conversion and buffering
 * - Redis integration for audio streaming to workers
 * - WebRTC data channels for bidirectional communication
 * - Room-based multi-participant support
 * - No server infrastructure needed (just signaling)
 *
 * Flow:
 * Browser → NinjaSDK P2P → This Service → Redis → Whisper Worker → LLM → Response
 */

const VDONinjaSDK = require('@vdoninja/sdk/node');
const redis = require('redis');
const amqp = require('amqplib');
const winston = require('winston');
require('dotenv').config();

// Configuration
const CONFIG = {
  // NinjaSDK Configuration
  ninjasdk: {
    host: process.env.NINJASDK_HOST || 'wss://wss.vdo.ninja',
    room: process.env.NINJASDK_ROOM || 'ooblex-audio',
    password: process.env.NINJASDK_PASSWORD || '',
    debug: process.env.DEBUG === 'true' || false,
  },
  // Redis Configuration
  redis: {
    url: process.env.REDIS_URL || 'redis://localhost:6379',
  },
  // RabbitMQ Configuration
  rabbitmq: {
    url: process.env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672',
  },
  // Audio Configuration
  audio: {
    sampleRate: 16000, // 16kHz for Whisper
    channels: 1, // Mono
    chunkDurationMs: 1000, // 1 second chunks
    format: 'pcm16',
  },
};

// Setup logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      ),
    }),
    new winston.transports.File({ filename: 'ninjasdk-audio.log' }),
  ],
});

/**
 * Audio Stream Manager
 * Handles audio capture, conversion, and buffering
 */
class AudioStreamManager {
  constructor(streamID, config) {
    this.streamID = streamID;
    this.config = config;
    this.audioBuffer = [];
    this.lastSampleRate = null;
    this.isActive = true;
    this.bytesReceived = 0;
    this.startTime = Date.now();
  }

  /**
   * Process raw audio data from RTCAudioSink
   * @param {Object} audioData - { samples: Buffer, sampleRate: Number, channels: Number }
   */
  async processAudioData(audioData) {
    if (!this.isActive) return;

    const { samples, sampleRate, channels } = audioData;

    // Log sample rate changes
    if (this.lastSampleRate !== sampleRate) {
      logger.info(`Audio sample rate: ${sampleRate}Hz, channels: ${channels}`);
      this.lastSampleRate = sampleRate;
    }

    // Convert to PCM16 if needed
    const pcm16Buffer = this.toPCM16(samples, sampleRate, channels);

    // Update stats
    this.bytesReceived += pcm16Buffer.length;

    // Add to buffer
    this.audioBuffer.push({
      data: pcm16Buffer,
      timestamp: Date.now(),
      sampleRate: this.config.audio.sampleRate,
      channels: this.config.audio.channels,
    });

    // Check if we have enough data for a chunk
    const chunkSizeBytes = (this.config.audio.sampleRate *
                            this.config.audio.chunkDurationMs / 1000 *
                            2); // 2 bytes per sample (PCM16)

    const totalBuffered = this.audioBuffer.reduce((sum, chunk) => sum + chunk.data.length, 0);

    if (totalBuffered >= chunkSizeBytes) {
      await this.flushBuffer();
    }
  }

  /**
   * Convert audio samples to PCM16 format
   */
  toPCM16(samples, sourceSampleRate, sourceChannels) {
    let pcmBuffer;

    // Handle different input formats
    if (samples instanceof Int16Array) {
      pcmBuffer = Buffer.from(samples.buffer);
    } else if (samples instanceof Float32Array) {
      // Convert Float32 (-1.0 to 1.0) to Int16
      const int16Array = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      pcmBuffer = Buffer.from(int16Array.buffer);
    } else if (Buffer.isBuffer(samples)) {
      pcmBuffer = samples;
    } else {
      logger.warn('Unknown audio sample format, attempting buffer conversion');
      pcmBuffer = Buffer.from(samples);
    }

    // Resample if needed (simple decimation/interpolation)
    if (sourceSampleRate !== this.config.audio.sampleRate) {
      pcmBuffer = this.resample(pcmBuffer, sourceSampleRate, this.config.audio.sampleRate);
    }

    // Convert to mono if needed
    if (sourceChannels > 1 && this.config.audio.channels === 1) {
      pcmBuffer = this.stereoToMono(pcmBuffer);
    }

    return pcmBuffer;
  }

  /**
   * Simple audio resampling (linear interpolation)
   */
  resample(buffer, fromRate, toRate) {
    if (fromRate === toRate) return buffer;

    const ratio = fromRate / toRate;
    const int16Input = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
    const outputLength = Math.floor(int16Input.length / ratio);
    const int16Output = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, int16Input.length - 1);
      const frac = srcIndex - srcIndexFloor;

      // Linear interpolation
      int16Output[i] = Math.round(
        int16Input[srcIndexFloor] * (1 - frac) + int16Input[srcIndexCeil] * frac
      );
    }

    return Buffer.from(int16Output.buffer);
  }

  /**
   * Convert stereo to mono by averaging channels
   */
  stereoToMono(buffer) {
    const int16Input = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
    const outputLength = Math.floor(int16Input.length / 2);
    const int16Output = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      int16Output[i] = Math.round((int16Input[i * 2] + int16Input[i * 2 + 1]) / 2);
    }

    return Buffer.from(int16Output.buffer);
  }

  /**
   * Flush audio buffer to Redis
   */
  async flushBuffer() {
    if (this.audioBuffer.length === 0) return;

    // Concatenate all buffers
    const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.data.length, 0);
    const combinedBuffer = Buffer.concat(
      this.audioBuffer.map(chunk => chunk.data),
      totalLength
    );

    // Store in Redis
    const chunkID = `audio:chunk:${this.streamID}:${Date.now()}`;
    const metadata = {
      streamID: this.streamID,
      timestamp: Date.now(),
      sampleRate: this.config.audio.sampleRate,
      channels: this.config.audio.channels,
      format: this.config.audio.format,
      durationMs: (combinedBuffer.length / 2 / this.config.audio.sampleRate) * 1000,
      bytes: combinedBuffer.length,
    };

    try {
      await global.redisClient.setEx(chunkID, 300, combinedBuffer); // 5 min TTL
      await global.redisClient.setEx(`${chunkID}:meta`, 300, JSON.stringify(metadata));

      // Publish event to RabbitMQ
      await global.rabbitChannel.sendToQueue(
        'audio-chunks',
        Buffer.from(JSON.stringify({
          chunkID,
          metadata,
        })),
        { persistent: false }
      );

      logger.debug(`Flushed audio chunk: ${chunkID} (${combinedBuffer.length} bytes)`);

      // Clear buffer
      this.audioBuffer = [];
    } catch (error) {
      logger.error('Failed to flush audio buffer:', error);
    }
  }

  /**
   * Get stream statistics
   */
  getStats() {
    const duration = (Date.now() - this.startTime) / 1000;
    const bytesPerSecond = this.bytesReceived / duration;
    const kbps = (bytesPerSecond * 8) / 1000;

    return {
      streamID: this.streamID,
      duration,
      bytesReceived: this.bytesReceived,
      kbps: kbps.toFixed(2),
      bufferedChunks: this.audioBuffer.length,
    };
  }

  /**
   * Stop and cleanup
   */
  async stop() {
    this.isActive = false;
    await this.flushBuffer();
    logger.info(`Audio stream stopped: ${this.streamID}`, this.getStats());
  }
}

/**
 * NinjaSDK Audio Service
 * Manages WebRTC connections and audio streams
 */
class NinjaSDKAudioService {
  constructor(config) {
    this.config = config;
    this.sdk = null;
    this.activeStreams = new Map();
    this.dataChannels = new Map();
  }

  /**
   * Initialize the service
   */
  async initialize() {
    logger.info('Initializing NinjaSDK Audio Service...');

    // Connect to Redis
    logger.info('Connecting to Redis...');
    global.redisClient = redis.createClient({ url: this.config.redis.url });
    await global.redisClient.connect();
    logger.info('Redis connected');

    // Connect to RabbitMQ
    logger.info('Connecting to RabbitMQ...');
    global.rabbitConnection = await amqp.connect(this.config.rabbitmq.url);
    global.rabbitChannel = await global.rabbitConnection.createChannel();
    await global.rabbitChannel.assertQueue('audio-chunks', { durable: false });
    await global.rabbitChannel.assertQueue('stt-results', { durable: false });
    logger.info('RabbitMQ connected');

    // Initialize NinjaSDK
    logger.info('Initializing NinjaSDK...');
    this.sdk = new VDONinjaSDK({
      host: this.config.ninjasdk.host,
      room: this.config.ninjasdk.room,
      password: this.config.ninjasdk.password,
      debug: this.config.ninjasdk.debug,
    });

    // Setup event listeners
    this.setupEventListeners();

    // Connect to NinjaSDK
    await this.sdk.connect();
    await this.sdk.joinRoom();

    logger.info(`NinjaSDK Audio Service ready! Room: ${this.config.ninjasdk.room}`);
    logger.info('Waiting for audio streams...');
  }

  /**
   * Setup NinjaSDK event listeners
   */
  setupEventListeners() {
    // New stream available
    this.sdk.on('videoaddedtoroom', async (streamID) => {
      logger.info(`New stream detected: ${streamID}`);
      await this.handleNewStream(streamID);
    });

    // Stream listing
    this.sdk.on('listing', async (streamIDs) => {
      logger.info(`Active streams in room: ${streamIDs.join(', ')}`);
      for (const streamID of streamIDs) {
        if (!this.activeStreams.has(streamID)) {
          await this.handleNewStream(streamID);
        }
      }
    });

    // Data received
    this.sdk.on('dataReceived', (data, streamID) => {
      logger.debug(`Data received from ${streamID}:`, data);
    });

    // Connection state changes
    this.sdk.on('connectionStateChange', (state, streamID) => {
      logger.info(`Connection state for ${streamID}: ${state}`);
    });
  }

  /**
   * Handle new audio stream
   */
  async handleNewStream(streamID) {
    try {
      logger.info(`Connecting to stream: ${streamID}`);

      // Use quickView to establish connection
      const pc = await this.sdk.quickView({
        streamID,
        room: this.config.ninjasdk.room,
        audio: true,
        video: false, // Audio only
      });

      logger.info(`Peer connection established for ${streamID}`);

      // Create audio stream manager
      const streamManager = new AudioStreamManager(streamID, this.config);
      this.activeStreams.set(streamID, streamManager);

      // Setup data channel for responses
      this.setupDataChannel(pc, streamID);

      // Handle audio tracks
      pc.ontrack = (event) => {
        logger.info(`Track received from ${streamID}:`, event.track.kind);

        if (event.track.kind === 'audio') {
          this.handleAudioTrack(event.track, streamID, streamManager);
        }
      };

      // Handle connection closure
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
          logger.warn(`Connection ${pc.connectionState} for ${streamID}`);
          this.handleStreamClosure(streamID);
        }
      };

    } catch (error) {
      logger.error(`Failed to handle stream ${streamID}:`, error);
    }
  }

  /**
   * Setup WebRTC data channel for bidirectional communication
   */
  setupDataChannel(pc, streamID) {
    try {
      const dataChannel = pc.createDataChannel('ooblex-stt', {
        ordered: true,
      });

      dataChannel.onopen = () => {
        logger.info(`Data channel opened for ${streamID}`);
        this.dataChannels.set(streamID, dataChannel);

        // Send welcome message
        this.sendToClient(streamID, {
          type: 'welcome',
          message: 'Connected to Ooblex NinjaSDK Audio Service',
          timestamp: Date.now(),
        });
      };

      dataChannel.onmessage = (event) => {
        logger.debug(`Message from ${streamID}:`, event.data);
        // Handle client messages if needed
      };

      dataChannel.onerror = (error) => {
        logger.error(`Data channel error for ${streamID}:`, error);
      };

      dataChannel.onclose = () => {
        logger.info(`Data channel closed for ${streamID}`);
        this.dataChannels.delete(streamID);
      };

    } catch (error) {
      logger.error(`Failed to setup data channel for ${streamID}:`, error);
    }
  }

  /**
   * Handle audio track with RTCAudioSink
   */
  handleAudioTrack(track, streamID, streamManager) {
    logger.info(`Processing audio track for ${streamID}`);

    try {
      // Use wrtc's RTCAudioSink to capture raw audio
      const { RTCAudioSink } = require('@roamhq/wrtc').nonstandard;
      const audioSink = new RTCAudioSink(track);

      audioSink.ondata = (audioData) => {
        // audioData = { samples: Int16Array|Float32Array, sampleRate: number, channels: number }
        streamManager.processAudioData(audioData);
      };

      // Store sink for cleanup
      streamManager.audioSink = audioSink;

      logger.info(`Audio sink attached for ${streamID}`);

    } catch (error) {
      logger.error(`Failed to setup audio sink for ${streamID}:`, error);
    }
  }

  /**
   * Send message to client via data channel
   */
  sendToClient(streamID, message) {
    const dataChannel = this.dataChannels.get(streamID);
    if (dataChannel && dataChannel.readyState === 'open') {
      try {
        dataChannel.send(JSON.stringify(message));
        logger.debug(`Sent to ${streamID}:`, message);
      } catch (error) {
        logger.error(`Failed to send to ${streamID}:`, error);
      }
    }
  }

  /**
   * Handle stream closure
   */
  async handleStreamClosure(streamID) {
    const streamManager = this.activeStreams.get(streamID);
    if (streamManager) {
      await streamManager.stop();

      if (streamManager.audioSink) {
        streamManager.audioSink.stop();
      }

      this.activeStreams.delete(streamID);
    }

    this.dataChannels.delete(streamID);
    logger.info(`Stream ${streamID} cleaned up`);
  }

  /**
   * Consume STT results and send back to clients
   */
  async startSTTResultConsumer() {
    logger.info('Starting STT result consumer...');

    global.rabbitChannel.consume('stt-results', (msg) => {
      if (msg) {
        try {
          const result = JSON.parse(msg.content.toString());
          const { streamID, text, confidence, timestamp } = result;

          logger.info(`STT Result for ${streamID}: "${text}" (${confidence})`);

          // Send to client via data channel
          this.sendToClient(streamID, {
            type: 'transcription',
            text,
            confidence,
            timestamp,
          });

          global.rabbitChannel.ack(msg);
        } catch (error) {
          logger.error('Failed to process STT result:', error);
          global.rabbitChannel.nack(msg, false, false);
        }
      }
    }, { noAck: false });
  }

  /**
   * Cleanup and shutdown
   */
  async shutdown() {
    logger.info('Shutting down NinjaSDK Audio Service...');

    // Stop all streams
    for (const [streamID, streamManager] of this.activeStreams) {
      await streamManager.stop();
    }

    // Disconnect SDK
    if (this.sdk) {
      await this.sdk.disconnect();
    }

    // Close connections
    if (global.redisClient) {
      await global.redisClient.quit();
    }

    if (global.rabbitConnection) {
      await global.rabbitConnection.close();
    }

    logger.info('Service shut down');
    process.exit(0);
  }
}

// Main entry point
async function main() {
  const service = new NinjaSDKAudioService(CONFIG);

  // Handle graceful shutdown
  process.on('SIGINT', () => service.shutdown());
  process.on('SIGTERM', () => service.shutdown());

  try {
    await service.initialize();
    await service.startSTTResultConsumer();
  } catch (error) {
    logger.error('Fatal error:', error);
    process.exit(1);
  }
}

// Run service
if (require.main === module) {
  main().catch((error) => {
    logger.error('Unhandled error:', error);
    process.exit(1);
  });
}

module.exports = { NinjaSDKAudioService, AudioStreamManager };
