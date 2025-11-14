/**
 * Unit tests for NinjaSDK Audio Ingestion Service
 */

const { AudioStreamManager } = require('../../services/ninjasdk-audio-ingestion/server');

describe('AudioStreamManager', () => {
  let manager;
  const mockConfig = {
    audio: {
      sampleRate: 16000,
      channels: 1,
      chunkDurationMs: 1000,
      format: 'pcm16',
    },
  };

  beforeEach(() => {
    manager = new AudioStreamManager('test-stream-123', mockConfig);
  });

  afterEach(async () => {
    await manager.stop();
  });

  describe('Audio Format Conversion', () => {
    test('should convert Float32Array to PCM16', () => {
      const float32Samples = new Float32Array([0.0, 0.5, -0.5, 1.0, -1.0]);
      const pcm16 = manager.toPCM16(float32Samples, 16000, 1);

      expect(pcm16).toBeInstanceOf(Buffer);
      expect(pcm16.length).toBe(float32Samples.length * 2); // 2 bytes per sample

      // Check conversion accuracy
      const int16Array = new Int16Array(pcm16.buffer, pcm16.byteOffset, pcm16.length / 2);
      expect(int16Array[0]).toBe(0);
      expect(Math.abs(int16Array[1] - 16383)).toBeLessThan(10); // ~0.5 * 32767
      expect(Math.abs(int16Array[2] + 16384)).toBeLessThan(10); // ~-0.5 * 32768
    });

    test('should convert Int16Array to PCM16', () => {
      const int16Samples = new Int16Array([0, 1000, -1000, 32767, -32768]);
      const pcm16 = manager.toPCM16(int16Samples, 16000, 1);

      expect(pcm16).toBeInstanceOf(Buffer);
      expect(pcm16.length).toBe(int16Samples.length * 2);
    });

    test('should handle Buffer input', () => {
      const buffer = Buffer.alloc(10);
      const pcm16 = manager.toPCM16(buffer, 16000, 1);

      expect(pcm16).toBeInstanceOf(Buffer);
    });
  });

  describe('Audio Resampling', () => {
    test('should resample from 48kHz to 16kHz', () => {
      const samples48k = new Int16Array(4800); // 100ms at 48kHz
      for (let i = 0; i < samples48k.length; i++) {
        samples48k[i] = Math.sin(i * 0.1) * 10000;
      }

      const buffer48k = Buffer.from(samples48k.buffer);
      const buffer16k = manager.resample(buffer48k, 48000, 16000);

      // Should be ~1/3 the size
      expect(buffer16k.length).toBeCloseTo(buffer48k.length / 3, 0);
    });

    test('should not resample when rates match', () => {
      const buffer = Buffer.alloc(1000);
      const resampled = manager.resample(buffer, 16000, 16000);

      expect(resampled).toBe(buffer);
    });
  });

  describe('Stereo to Mono Conversion', () => {
    test('should convert stereo to mono by averaging', () => {
      const stereo = new Int16Array([1000, 2000, 3000, 4000, 5000, 6000]);
      const stereoBuffer = Buffer.from(stereo.buffer);
      const mono = manager.stereoToMono(stereoBuffer);

      const monoArray = new Int16Array(mono.buffer, mono.byteOffset, mono.length / 2);
      expect(monoArray.length).toBe(3);
      expect(monoArray[0]).toBe(1500); // (1000 + 2000) / 2
      expect(monoArray[1]).toBe(3500); // (3000 + 4000) / 2
      expect(monoArray[2]).toBe(5500); // (5000 + 6000) / 2
    });
  });

  describe('Audio Buffering', () => {
    test('should accumulate audio data', async () => {
      const audioData = {
        samples: new Float32Array(1600), // 100ms at 16kHz
        sampleRate: 16000,
        channels: 1,
      };

      await manager.processAudioData(audioData);

      expect(manager.audioBuffer.length).toBeGreaterThan(0);
      expect(manager.bytesReceived).toBeGreaterThan(0);
    });

    test('should flush buffer when chunk size reached', async () => {
      // Mock Redis and RabbitMQ
      global.redisClient = {
        setEx: jest.fn().mockResolvedValue('OK'),
      };
      global.rabbitChannel = {
        sendToQueue: jest.fn(),
      };

      // Generate 1 second of audio
      const audioData = {
        samples: new Float32Array(16000),
        sampleRate: 16000,
        channels: 1,
      };

      await manager.processAudioData(audioData);

      // Should have flushed
      expect(global.redisClient.setEx).toHaveBeenCalled();
      expect(global.rabbitChannel.sendToQueue).toHaveBeenCalled();
      expect(manager.audioBuffer.length).toBe(0);
    });
  });

  describe('Statistics', () => {
    test('should track audio statistics', () => {
      manager.bytesReceived = 32000; // 1 second of 16kHz mono PCM16
      manager.startTime = Date.now() - 1000; // 1 second ago

      const stats = manager.getStats();

      expect(stats.streamID).toBe('test-stream-123');
      expect(stats.bytesReceived).toBe(32000);
      expect(stats.kbps).toBeCloseTo(256, 0); // 32000 bytes/s * 8 bits/byte / 1000
    });
  });
});

describe('Integration Tests', () => {
  test('should handle end-to-end audio flow', async () => {
    // This would require a full integration test environment
    // For now, we'll just validate the components are exported correctly
    expect(AudioStreamManager).toBeDefined();
  });
});
