#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace emscripten;

class StyleTransferLite {
private:
    // Simplified neural style transfer using separable convolutions
    struct ConvKernel {
        std::vector<float> weights;
        int size;
        float bias;
    };
    
    std::vector<ConvKernel> style_kernels;
    float style_strength = 0.7f;
    
public:
    StyleTransferLite() {
        initializeKernels();
    }
    
    void initializeKernels() {
        // Initialize with basic edge detection and blur kernels
        // In production, these would be learned from style images
        
        // Edge detection kernel
        ConvKernel edge;
        edge.size = 3;
        edge.weights = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        edge.bias = 0.0f;
        style_kernels.push_back(edge);
        
        // Gaussian blur kernel
        ConvKernel blur;
        blur.size = 3;
        blur.weights = {1/16.0f, 2/16.0f, 1/16.0f, 
                       2/16.0f, 4/16.0f, 2/16.0f,
                       1/16.0f, 2/16.0f, 1/16.0f};
        blur.bias = 0.0f;
        style_kernels.push_back(blur);
    }
    
    val transfer(val imageData, int width, int height, const std::string& style) {
        auto data = vecFromJSArray<uint8_t>(imageData);
        std::vector<uint8_t> output(data.size());
        
        // Extract RGB channels
        std::vector<std::vector<float>> channels(3, std::vector<float>(width * height));
        for (int i = 0; i < width * height; i++) {
            channels[0][i] = data[i * 4];     // R
            channels[1][i] = data[i * 4 + 1]; // G
            channels[2][i] = data[i * 4 + 2]; // B
        }
        
        // Apply style-specific transformations
        if (style == "sketch") {
            applySketchStyle(channels, width, height);
        } else if (style == "oil_painting") {
            applyOilPaintingStyle(channels, width, height);
        } else if (style == "watercolor") {
            applyWatercolorStyle(channels, width, height);
        }
        
        // Convert back to RGBA
        for (int i = 0; i < width * height; i++) {
            output[i * 4] = clamp(channels[0][i]);
            output[i * 4 + 1] = clamp(channels[1][i]);
            output[i * 4 + 2] = clamp(channels[2][i]);
            output[i * 4 + 3] = data[i * 4 + 3]; // Alpha
        }
        
        return val(typed_memory_view(output.size(), output.data()));
    }
    
    void setStyleStrength(float strength) {
        style_strength = std::max(0.0f, std::min(1.0f, strength));
    }
    
private:
    void applySketchStyle(std::vector<std::vector<float>>& channels, 
                         int width, int height) {
        // Convert to grayscale
        std::vector<float> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            gray[i] = 0.299f * channels[0][i] + 
                     0.587f * channels[1][i] + 
                     0.114f * channels[2][i];
        }
        
        // Apply edge detection
        auto edges = applyConvolution(gray, width, height, style_kernels[0]);
        
        // Invert and threshold
        for (int i = 0; i < edges.size(); i++) {
            float edge_val = 255.0f - std::abs(edges[i]);
            edge_val = edge_val > 200 ? 255 : 0;
            
            // Apply to all channels
            for (int c = 0; c < 3; c++) {
                channels[c][i] = channels[c][i] * (1 - style_strength) + 
                                edge_val * style_strength;
            }
        }
    }
    
    void applyOilPaintingStyle(std::vector<std::vector<float>>& channels, 
                               int width, int height) {
        int radius = 4;
        std::vector<std::vector<float>> output(3, std::vector<float>(width * height));
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::vector<int> intensity_count(256, 0);
                std::vector<std::vector<float>> intensity_sum(256, std::vector<float>(3, 0));
                
                // Sample neighborhood
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = std::max(0, std::min(width - 1, x + dx));
                        int ny = std::max(0, std::min(height - 1, y + dy));
                        int idx = ny * width + nx;
                        
                        int intensity = (int)(channels[0][idx] * 0.299f + 
                                            channels[1][idx] * 0.587f + 
                                            channels[2][idx] * 0.114f);
                        intensity = std::max(0, std::min(255, intensity));
                        
                        intensity_count[intensity]++;
                        for (int c = 0; c < 3; c++) {
                            intensity_sum[intensity][c] += channels[c][idx];
                        }
                    }
                }
                
                // Find most frequent intensity
                int max_count = 0;
                int max_intensity = 0;
                for (int i = 0; i < 256; i++) {
                    if (intensity_count[i] > max_count) {
                        max_count = intensity_count[i];
                        max_intensity = i;
                    }
                }
                
                // Set pixel to average of most frequent intensity
                int idx = y * width + x;
                for (int c = 0; c < 3; c++) {
                    float new_val = intensity_sum[max_intensity][c] / max_count;
                    output[c][idx] = channels[c][idx] * (1 - style_strength) + 
                                    new_val * style_strength;
                }
            }
        }
        
        channels = output;
    }
    
    void applyWatercolorStyle(std::vector<std::vector<float>>& channels, 
                             int width, int height) {
        // Apply multiple passes of bilateral filtering
        for (int pass = 0; pass < 3; pass++) {
            auto filtered = bilateralFilter(channels, width, height, 5, 30.0f, 30.0f);
            
            // Blend with original
            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < width * height; i++) {
                    channels[c][i] = channels[c][i] * (1 - style_strength * 0.3f) + 
                                    filtered[c][i] * style_strength * 0.3f;
                }
            }
        }
        
        // Add paper texture
        for (int i = 0; i < width * height; i++) {
            float noise = (rand() % 20 - 10) * style_strength * 0.5f;
            for (int c = 0; c < 3; c++) {
                channels[c][i] += noise;
            }
        }
    }
    
    std::vector<float> applyConvolution(const std::vector<float>& input, 
                                       int width, int height, 
                                       const ConvKernel& kernel) {
        std::vector<float> output(width * height);
        int half = kernel.size / 2;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0;
                int weight_idx = 0;
                
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int nx = std::max(0, std::min(width - 1, x + kx));
                        int ny = std::max(0, std::min(height - 1, y + ky));
                        sum += input[ny * width + nx] * kernel.weights[weight_idx++];
                    }
                }
                
                output[y * width + x] = sum + kernel.bias;
            }
        }
        
        return output;
    }
    
    std::vector<std::vector<float>> bilateralFilter(
        const std::vector<std::vector<float>>& channels,
        int width, int height, int radius, 
        float sigma_color, float sigma_space) {
        
        std::vector<std::vector<float>> output(3, std::vector<float>(width * height));
        
        // Precompute spatial weights
        std::vector<float> spatial_weights((2 * radius + 1) * (2 * radius + 1));
        int idx = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                float dist = sqrt(dx * dx + dy * dy);
                spatial_weights[idx++] = exp(-dist * dist / (2 * sigma_space * sigma_space));
            }
        }
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    float sum = 0;
                    float weight_sum = 0;
                    float center_val = channels[c][y * width + x];
                    
                    idx = 0;
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int nx = std::max(0, std::min(width - 1, x + dx));
                            int ny = std::max(0, std::min(height - 1, y + dy));
                            
                            float neighbor_val = channels[c][ny * width + nx];
                            float color_dist = std::abs(neighbor_val - center_val);
                            float color_weight = exp(-color_dist * color_dist / 
                                                   (2 * sigma_color * sigma_color));
                            
                            float weight = spatial_weights[idx++] * color_weight;
                            sum += neighbor_val * weight;
                            weight_sum += weight;
                        }
                    }
                    
                    output[c][y * width + x] = sum / weight_sum;
                }
            }
        }
        
        return output;
    }
    
    uint8_t clamp(float val) {
        return (uint8_t)std::max(0.0f, std::min(255.0f, val));
    }
};

// Binding code
EMSCRIPTEN_BINDINGS(style_transfer_lite) {
    class_<StyleTransferLite>("StyleTransferLite")
        .constructor()
        .function("transfer", &StyleTransferLite::transfer)
        .function("setStyleStrength", &StyleTransferLite::setStyleStrength);
}