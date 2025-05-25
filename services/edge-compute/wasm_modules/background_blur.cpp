#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace emscripten;

class BackgroundBlur {
private:
    // Segmentation model weights (simplified)
    struct SegmentationModel {
        std::vector<float> edge_weights;
        std::vector<float> color_weights;
        float threshold;
    };
    
    SegmentationModel model;
    int blur_radius = 15;
    float blur_strength = 0.8f;
    
public:
    BackgroundBlur() {
        initializeModel();
    }
    
    void initializeModel() {
        // Initialize simple segmentation model
        // In production, this would be a trained neural network
        model.threshold = 0.5f;
        model.edge_weights = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        model.color_weights = {0.299f, 0.587f, 0.114f}; // RGB to luminance
    }
    
    val process(val imageData, int width, int height) {
        auto data = vecFromJSArray<uint8_t>(imageData);
        std::vector<uint8_t> output(data.size());
        
        // Generate person mask
        auto mask = generatePersonMask(data, width, height);
        
        // Apply blur to background
        auto blurred = gaussianBlur(data, width, height, blur_radius);
        
        // Composite based on mask
        for (int i = 0; i < width * height; i++) {
            float mask_val = mask[i];
            float inv_mask = 1.0f - mask_val;
            
            for (int c = 0; c < 3; c++) {
                int idx = i * 4 + c;
                output[idx] = (uint8_t)(data[idx] * mask_val + 
                                       blurred[idx] * inv_mask * blur_strength +
                                       data[idx] * inv_mask * (1 - blur_strength));
            }
            output[i * 4 + 3] = data[i * 4 + 3]; // Alpha
        }
        
        return val(typed_memory_view(output.size(), output.data()));
    }
    
    val processWithDepth(val imageData, val depthData, int width, int height) {
        auto data = vecFromJSArray<uint8_t>(imageData);
        auto depth = vecFromJSArray<float>(depthData);
        std::vector<uint8_t> output(data.size());
        
        // Find foreground depth threshold
        float depth_threshold = findDepthThreshold(depth);
        
        // Generate mask from depth
        std::vector<float> mask(width * height);
        for (int i = 0; i < width * height; i++) {
            mask[i] = depth[i] < depth_threshold ? 1.0f : 0.0f;
        }
        
        // Refine mask edges
        mask = refineMask(mask, width, height);
        
        // Variable blur based on depth
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float mask_val = mask[idx];
                
                if (mask_val < 0.9f) {
                    // Background - apply blur
                    int blur_size = (int)(blur_radius * (1.0f - mask_val));
                    auto pixel = getBlurredPixel(data, width, height, x, y, blur_size);
                    
                    for (int c = 0; c < 3; c++) {
                        output[idx * 4 + c] = (uint8_t)(
                            data[idx * 4 + c] * mask_val +
                            pixel[c] * (1.0f - mask_val)
                        );
                    }
                } else {
                    // Foreground - keep original
                    for (int c = 0; c < 3; c++) {
                        output[idx * 4 + c] = data[idx * 4 + c];
                    }
                }
                output[idx * 4 + 3] = data[idx * 4 + 3]; // Alpha
            }
        }
        
        return val(typed_memory_view(output.size(), output.data()));
    }
    
    void setBlurRadius(int radius) {
        blur_radius = std::max(1, std::min(50, radius));
    }
    
    void setBlurStrength(float strength) {
        blur_strength = std::max(0.0f, std::min(1.0f, strength));
    }
    
private:
    std::vector<float> generatePersonMask(const std::vector<uint8_t>& data, 
                                         int width, int height) {
        std::vector<float> mask(width * height, 0.0f);
        
        // Simple center-weighted person detection
        // In production, use a proper segmentation model
        int centerX = width / 2;
        int centerY = height / 2;
        
        // Find skin-tone regions
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 4;
                float r = data[idx] / 255.0f;
                float g = data[idx + 1] / 255.0f;
                float b = data[idx + 2] / 255.0f;
                
                // Simple skin detection
                bool isSkin = (r > 0.35f && g > 0.25f && b > 0.15f) &&
                             (std::abs(r - g) < 0.15f) &&
                             (r > b) && (g > b);
                
                // Distance from center
                float dx = (x - centerX) / (float)width;
                float dy = (y - centerY) / (float)height;
                float dist = sqrt(dx * dx + dy * dy);
                
                // Combine skin detection with center bias
                float confidence = isSkin ? 0.7f : 0.3f;
                confidence *= (1.0f - dist * 0.5f);
                
                mask[y * width + x] = confidence;
            }
        }
        
        // Apply morphological operations
        mask = dilate(mask, width, height, 3);
        mask = erode(mask, width, height, 2);
        
        // Smooth mask
        mask = gaussianBlurMask(mask, width, height, 5);
        
        return mask;
    }
    
    float findDepthThreshold(const std::vector<float>& depth) {
        // Find depth histogram
        std::vector<int> histogram(256, 0);
        float min_depth = *std::min_element(depth.begin(), depth.end());
        float max_depth = *std::max_element(depth.begin(), depth.end());
        
        for (float d : depth) {
            int bin = (int)((d - min_depth) / (max_depth - min_depth) * 255);
            histogram[std::max(0, std::min(255, bin))]++;
        }
        
        // Find first significant peak (likely foreground)
        int peak_threshold = depth.size() / 100; // 1% of pixels
        for (int i = 0; i < 256; i++) {
            if (histogram[i] > peak_threshold) {
                return min_depth + (i + 20) * (max_depth - min_depth) / 255.0f;
            }
        }
        
        return (min_depth + max_depth) / 2.0f;
    }
    
    std::vector<float> refineMask(const std::vector<float>& mask, 
                                 int width, int height) {
        std::vector<float> refined = mask;
        
        // Edge-aware smoothing
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                float center = mask[idx];
                float sum = center * 4;
                float weight_sum = 4;
                
                // Check neighbors
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nidx = (y + dy) * width + (x + dx);
                        float neighbor = mask[nidx];
                        float diff = std::abs(neighbor - center);
                        float weight = exp(-diff * diff * 10);
                        
                        sum += neighbor * weight;
                        weight_sum += weight;
                    }
                }
                
                refined[idx] = sum / weight_sum;
            }
        }
        
        return refined;
    }
    
    std::vector<uint8_t> gaussianBlur(const std::vector<uint8_t>& data, 
                                     int width, int height, int radius) {
        std::vector<uint8_t> output(data.size());
        
        // Generate Gaussian kernel
        int kernel_size = radius * 2 + 1;
        std::vector<float> kernel(kernel_size * kernel_size);
        float sigma = radius / 3.0f;
        float sum = 0;
        
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
                kernel[(y + radius) * kernel_size + (x + radius)] = val;
                sum += val;
            }
        }
        
        // Normalize kernel
        for (auto& k : kernel) k /= sum;
        
        // Apply blur
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float r = 0, g = 0, b = 0;
                
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = std::max(0, std::min(width - 1, x + kx));
                        int ny = std::max(0, std::min(height - 1, y + ky));
                        int idx = (ny * width + nx) * 4;
                        float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
                        
                        r += data[idx] * weight;
                        g += data[idx + 1] * weight;
                        b += data[idx + 2] * weight;
                    }
                }
                
                int out_idx = (y * width + x) * 4;
                output[out_idx] = (uint8_t)std::min(255.0f, r);
                output[out_idx + 1] = (uint8_t)std::min(255.0f, g);
                output[out_idx + 2] = (uint8_t)std::min(255.0f, b);
                output[out_idx + 3] = data[out_idx + 3];
            }
        }
        
        return output;
    }
    
    std::vector<float> gaussianBlurMask(const std::vector<float>& mask, 
                                       int width, int height, int radius) {
        std::vector<float> output(mask.size());
        float sigma = radius / 3.0f;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0;
                float weight_sum = 0;
                
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = std::max(0, std::min(width - 1, x + kx));
                        int ny = std::max(0, std::min(height - 1, y + ky));
                        
                        float weight = exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
                        sum += mask[ny * width + nx] * weight;
                        weight_sum += weight;
                    }
                }
                
                output[y * width + x] = sum / weight_sum;
            }
        }
        
        return output;
    }
    
    std::vector<float> getBlurredPixel(const std::vector<uint8_t>& data,
                                      int width, int height,
                                      int x, int y, int radius) {
        float r = 0, g = 0, b = 0;
        float weight_sum = 0;
        float sigma = radius / 3.0f;
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = std::max(0, std::min(width - 1, x + dx));
                int ny = std::max(0, std::min(height - 1, y + dy));
                int idx = (ny * width + nx) * 4;
                
                float weight = exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                r += data[idx] * weight;
                g += data[idx + 1] * weight;
                b += data[idx + 2] * weight;
                weight_sum += weight;
            }
        }
        
        return {r / weight_sum, g / weight_sum, b / weight_sum};
    }
    
    std::vector<float> dilate(const std::vector<float>& mask, 
                             int width, int height, int radius) {
        std::vector<float> output(mask.size());
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float max_val = 0;
                
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = std::max(0, std::min(width - 1, x + dx));
                        int ny = std::max(0, std::min(height - 1, y + dy));
                        max_val = std::max(max_val, mask[ny * width + nx]);
                    }
                }
                
                output[y * width + x] = max_val;
            }
        }
        
        return output;
    }
    
    std::vector<float> erode(const std::vector<float>& mask, 
                            int width, int height, int radius) {
        std::vector<float> output(mask.size());
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float min_val = 1.0f;
                
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = std::max(0, std::min(width - 1, x + dx));
                        int ny = std::max(0, std::min(height - 1, y + dy));
                        min_val = std::min(min_val, mask[ny * width + nx]);
                    }
                }
                
                output[y * width + x] = min_val;
            }
        }
        
        return output;
    }
};

// Binding code
EMSCRIPTEN_BINDINGS(background_blur) {
    class_<BackgroundBlur>("BackgroundBlur")
        .constructor()
        .function("process", &BackgroundBlur::process)
        .function("processWithDepth", &BackgroundBlur::processWithDepth)
        .function("setBlurRadius", &BackgroundBlur::setBlurRadius)
        .function("setBlurStrength", &BackgroundBlur::setBlurStrength);
}