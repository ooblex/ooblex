#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace emscripten;

struct BoundingBox {
    float x, y, width, height;
    float confidence;
};

class FaceDetector {
private:
    // Simplified Viola-Jones cascade classifier
    struct HaarFeature {
        int x, y, width, height;
        float threshold;
        float left_val, right_val;
    };
    
    std::vector<HaarFeature> cascade;
    int window_size = 24;
    
public:
    FaceDetector() {
        // Initialize with basic Haar features
        initializeCascade();
    }
    
    void initializeCascade() {
        // Simplified cascade - in production, load from trained model
        cascade.push_back({2, 2, 10, 10, 0.3f, -1.0f, 1.0f});
        cascade.push_back({8, 8, 8, 8, 0.4f, -1.0f, 1.0f});
        cascade.push_back({4, 4, 16, 16, 0.5f, -1.0f, 1.0f});
    }
    
    std::vector<BoundingBox> detect(val imageData, int width, int height) {
        std::vector<BoundingBox> faces;
        auto data = vecFromJSArray<uint8_t>(imageData);
        
        // Convert to grayscale
        std::vector<float> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            int idx = i * 4;
            gray[i] = 0.299f * data[idx] + 0.587f * data[idx + 1] + 0.114f * data[idx + 2];
        }
        
        // Sliding window detection
        float scale = 1.0f;
        while (window_size * scale < std::min(width, height)) {
            int step = std::max(1, (int)(2 * scale));
            int scaledWindow = (int)(window_size * scale);
            
            for (int y = 0; y < height - scaledWindow; y += step) {
                for (int x = 0; x < width - scaledWindow; x += step) {
                    if (detectWindow(gray, width, x, y, scaledWindow)) {
                        faces.push_back({
                            (float)x, (float)y, 
                            (float)scaledWindow, (float)scaledWindow,
                            0.8f
                        });
                    }
                }
            }
            scale *= 1.2f;
        }
        
        // Non-maximum suppression
        return nonMaxSuppression(faces);
    }
    
private:
    bool detectWindow(const std::vector<float>& gray, int imgWidth, 
                     int x, int y, int windowSize) {
        // Simplified detection - check basic patterns
        float centerBrightness = getRegionMean(gray, imgWidth, 
            x + windowSize/4, y + windowSize/4, windowSize/2, windowSize/2);
        float edgeBrightness = getRegionMean(gray, imgWidth, 
            x, y, windowSize, windowSize) - centerBrightness;
        
        // Basic face pattern: center brighter than edges
        return centerBrightness > 100 && edgeBrightness < -20;
    }
    
    float getRegionMean(const std::vector<float>& gray, int imgWidth,
                       int x, int y, int w, int h) {
        float sum = 0;
        int count = 0;
        for (int dy = 0; dy < h; dy++) {
            for (int dx = 0; dx < w; dx++) {
                int idx = (y + dy) * imgWidth + (x + dx);
                if (idx < gray.size()) {
                    sum += gray[idx];
                    count++;
                }
            }
        }
        return count > 0 ? sum / count : 0;
    }
    
    std::vector<BoundingBox> nonMaxSuppression(std::vector<BoundingBox>& boxes) {
        if (boxes.empty()) return boxes;
        
        // Sort by confidence
        std::sort(boxes.begin(), boxes.end(), 
            [](const BoundingBox& a, const BoundingBox& b) {
                return a.confidence > b.confidence;
            });
        
        std::vector<BoundingBox> result;
        std::vector<bool> suppressed(boxes.size(), false);
        
        for (size_t i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            
            result.push_back(boxes[i]);
            
            // Suppress overlapping boxes
            for (size_t j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                
                float iou = computeIoU(boxes[i], boxes[j]);
                if (iou > 0.5f) {
                    suppressed[j] = true;
                }
            }
        }
        
        return result;
    }
    
    float computeIoU(const BoundingBox& a, const BoundingBox& b) {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.width, b.x + b.width);
        float y2 = std::min(a.y + a.height, b.y + b.height);
        
        if (x2 < x1 || y2 < y1) return 0;
        
        float intersection = (x2 - x1) * (y2 - y1);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        
        return intersection / (areaA + areaB - intersection);
    }
};

// Binding code
EMSCRIPTEN_BINDINGS(face_detection) {
    value_object<BoundingBox>("BoundingBox")
        .field("x", &BoundingBox::x)
        .field("y", &BoundingBox::y)
        .field("width", &BoundingBox::width)
        .field("height", &BoundingBox::height)
        .field("confidence", &BoundingBox::confidence);
    
    register_vector<BoundingBox>("BoundingBoxVector");
    
    class_<FaceDetector>("FaceDetector")
        .constructor()
        .function("detect", &FaceDetector::detect);
}