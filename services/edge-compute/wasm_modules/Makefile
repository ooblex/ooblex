# Makefile for building WebAssembly modules
EMCC = emcc
CXXFLAGS = -O3 -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
           -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=512MB \
           -s USE_ES6_IMPORT_META=1 -s SINGLE_FILE=1 \
           --bind -std=c++17

# Output directory
BUILD_DIR = ../../../html/wasm

# Source files
SOURCES = face_detection.cpp style_transfer_lite.cpp background_blur.cpp
TARGETS = $(patsubst %.cpp,$(BUILD_DIR)/%.js,$(SOURCES))

.PHONY: all clean create-dir

all: create-dir $(TARGETS)

create-dir:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.js: %.cpp
	$(EMCC) $(CXXFLAGS) -s EXPORT_NAME='$*Module' $< -o $@
	@echo "Built $@ successfully"

# Build with debugging symbols
debug: CXXFLAGS += -g -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=2
debug: all

# Build with profiling
profile: CXXFLAGS += --profiling
profile: all

# Build optimized for size
size: CXXFLAGS = -Os -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 \
                  -s ALLOW_MEMORY_GROWTH=1 --bind -std=c++17 \
                  -s ENVIRONMENT='web,worker' -s FILESYSTEM=0
size: all

clean:
	rm -f $(BUILD_DIR)/*.js $(BUILD_DIR)/*.wasm

# Individual targets
face_detection: create-dir $(BUILD_DIR)/face_detection.js
style_transfer: create-dir $(BUILD_DIR)/style_transfer_lite.js
background_blur: create-dir $(BUILD_DIR)/background_blur.js

# Test build
test: all
	@echo "WebAssembly modules built successfully!"
	@echo "Modules available at: $(BUILD_DIR)/"
	@ls -lh $(BUILD_DIR)/*.js