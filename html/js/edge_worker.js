// Edge Worker for WebAssembly AI Processing
// This worker handles AI processing tasks using WebAssembly modules

let config = {};
let modules = {};
let initialized = false;

// Message handler
self.onmessage = async function(event) {
    const { type, id } = event.data;
    
    try {
        switch (type) {
            case 'init':
                await handleInit(event.data.config);
                self.postMessage({ type: 'init_complete', id });
                break;
                
            case 'process':
                const result = await handleProcess(event.data);
                self.postMessage({ type: 'result', id, result });
                break;
                
            case 'preload':
                await handlePreload(event.data.modules);
                self.postMessage({ type: 'preload_complete', id });
                break;
                
            case 'cleanup':
                handleCleanup();
                self.postMessage({ type: 'cleanup_complete', id });
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        self.postMessage({ 
            type: 'error', 
            id, 
            error: error.message || 'Processing failed' 
        });
    }
};

// Initialize worker
async function handleInit(workerConfig) {
    config = workerConfig;
    initialized = true;
    
    // Register as edge worker
    try {
        const response = await fetch(`${config.serverUrl}/api/edge/workers/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                worker_id: getWorkerId(),
                capabilities: ['face_detection', 'style_transfer', 'background_blur'],
                device_info: {
                    userAgent: self.navigator.userAgent,
                    hardwareConcurrency: self.navigator.hardwareConcurrency,
                    platform: self.navigator.platform
                }
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('Edge worker registered:', data);
            
            // Start heartbeat
            startHeartbeat();
        }
    } catch (error) {
        console.error('Failed to register edge worker:', error);
    }
}

// Process task
async function handleProcess(data) {
    const { taskType, module: moduleName, data: processData, options } = data;
    
    // Load module if not cached
    if (!modules[moduleName]) {
        modules[moduleName] = await loadModule(moduleName);
    }
    
    const module = modules[moduleName];
    const startTime = performance.now();
    
    try {
        let result;
        
        switch (taskType) {
            case 'face_detection':
                result = await processFaceDetection(module, processData, options);
                break;
                
            case 'style_transfer':
                result = await processStyleTransfer(module, processData, options);
                break;
                
            case 'background_blur':
                result = await processBackgroundBlur(module, processData, options);
                break;
                
            default:
                throw new Error(`Unknown task type: ${taskType}`);
        }
        
        const processingTime = performance.now() - startTime;
        
        // Report metrics
        reportMetrics(taskType, processingTime, true);
        
        return result;
        
    } catch (error) {
        const processingTime = performance.now() - startTime;
        reportMetrics(taskType, processingTime, false);
        throw error;
    }
}

// Face detection processing
async function processFaceDetection(module, imageData, options) {
    const detector = new module.FaceDetector();
    
    // Convert image data to format expected by WASM
    const width = options.width || 640;
    const height = options.height || 480;
    
    // Process with WASM
    const faces = detector.detect(imageData, width, height);
    
    // Convert result to JavaScript array
    const result = [];
    for (let i = 0; i < faces.size(); i++) {
        const face = faces.get(i);
        result.push({
            x: face.x,
            y: face.y,
            width: face.width,
            height: face.height,
            confidence: face.confidence
        });
    }
    
    // Clean up
    faces.delete();
    detector.delete();
    
    return result;
}

// Style transfer processing
async function processStyleTransfer(module, data, options) {
    const styleTransfer = new module.StyleTransferLite();
    
    // Set style strength if provided
    if (options.strength !== undefined) {
        styleTransfer.setStyleStrength(options.strength);
    }
    
    const width = options.width || 640;
    const height = options.height || 480;
    const style = data.style || 'sketch';
    
    // Process with WASM
    const result = styleTransfer.transfer(data.imageData, width, height, style);
    
    // Convert to Uint8Array
    const resultArray = new Uint8Array(result);
    
    // Clean up
    styleTransfer.delete();
    
    return resultArray;
}

// Background blur processing
async function processBackgroundBlur(module, imageData, options) {
    const backgroundBlur = new module.BackgroundBlur();
    
    // Set blur parameters
    if (options.blurRadius !== undefined) {
        backgroundBlur.setBlurRadius(options.blurRadius);
    }
    if (options.blurStrength !== undefined) {
        backgroundBlur.setBlurStrength(options.blurStrength);
    }
    
    const width = options.width || 640;
    const height = options.height || 480;
    
    // Process with WASM
    let result;
    if (options.depthData) {
        result = backgroundBlur.processWithDepth(
            imageData, 
            options.depthData, 
            width, 
            height
        );
    } else {
        result = backgroundBlur.process(imageData, width, height);
    }
    
    // Convert to Uint8Array
    const resultArray = new Uint8Array(result);
    
    // Clean up
    backgroundBlur.delete();
    
    return resultArray;
}

// Preload modules
async function handlePreload(moduleNames) {
    const promises = moduleNames.map(name => loadModule(name));
    await Promise.all(promises);
}

// Allowed module names whitelist for security
const ALLOWED_MODULES = new Set([
    'face_detection',
    'style_transfer',
    'background_blur'
]);

// Validate module name to prevent path traversal
function isValidModuleName(name) {
    // Only allow alphanumeric characters and underscores
    return ALLOWED_MODULES.has(name) && /^[a-zA-Z0-9_]+$/.test(name);
}

// Load WASM module
async function loadModule(moduleName) {
    if (modules[moduleName]) {
        return modules[moduleName];
    }

    // Validate module name to prevent injection attacks
    if (!isValidModuleName(moduleName)) {
        throw new Error(`Invalid module name: ${moduleName}`);
    }

    try {
        // Import module dynamically using WebAssembly
        const moduleUrl = `${config.serverUrl}/api/edge/modules/${encodeURIComponent(moduleName)}.wasm`;
        const response = await fetch(moduleUrl);

        // Validate response content type
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/wasm')) {
            throw new Error(`Invalid content type for module ${moduleName}: ${contentType}`);
        }

        if (!response.ok) {
            throw new Error(`Failed to fetch module ${moduleName}: ${response.status}`);
        }

        // Use WebAssembly.instantiateStreaming for safe module loading
        const wasmModule = await WebAssembly.instantiateStreaming(response);

        modules[moduleName] = wasmModule.instance.exports;
        console.log('Loaded WASM module:', moduleName);

        return modules[moduleName];
    } catch (error) {
        console.error('Failed to load module %s:', moduleName, error.message);
        throw error;
    }
}

// Clean up resources
function handleCleanup() {
    // Delete all WASM module instances
    for (const module of Object.values(modules)) {
        if (module && typeof module.delete === 'function') {
            module.delete();
        }
    }
    modules = {};
}

// Get or generate worker ID
function getWorkerId() {
    let workerId = self.workerId;
    if (!workerId) {
        // Use crypto.getRandomValues for secure random ID generation
        const array = new Uint8Array(12);
        crypto.getRandomValues(array);
        workerId = 'worker_' + Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('').slice(0, 9);
        self.workerId = workerId;
    }
    return workerId;
}

// Start heartbeat to maintain worker registration
function startHeartbeat() {
    setInterval(async () => {
        try {
            const response = await fetch(
                `${config.serverUrl}/api/edge/workers/${getWorkerId()}/heartbeat`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        metrics: {
                            load: getWorkerLoad(),
                            processing_time: getAverageProcessingTime()
                        }
                    })
                }
            );
            
            if (!response.ok) {
                console.error('Heartbeat failed');
            }
        } catch (error) {
            console.error('Failed to send heartbeat:', error);
        }
    }, 30000); // Every 30 seconds
}

// Report processing metrics
async function reportMetrics(taskType, processingTime, success) {
    // Store metrics locally
    if (!self.metrics) {
        self.metrics = {
            processing_times: [],
            success_count: 0,
            failure_count: 0
        };
    }
    
    self.metrics.processing_times.push(processingTime);
    if (self.metrics.processing_times.length > 100) {
        self.metrics.processing_times.shift();
    }
    
    if (success) {
        self.metrics.success_count++;
    } else {
        self.metrics.failure_count++;
    }
    
    // Report to server periodically
    if ((self.metrics.success_count + self.metrics.failure_count) % 10 === 0) {
        try {
            const successRate = self.metrics.success_count / 
                               (self.metrics.success_count + self.metrics.failure_count);
            
            await fetch(
                `${config.serverUrl}/api/edge/workers/${getWorkerId()}/report`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        success_rate: successRate,
                        processing_stats: {
                            task_type: taskType,
                            avg_processing_time: getAverageProcessingTime(),
                            total_processed: self.metrics.success_count + self.metrics.failure_count
                        }
                    })
                }
            );
        } catch (error) {
            console.error('Failed to report metrics:', error);
        }
    }
}

// Get worker load estimate
function getWorkerLoad() {
    // Simple load estimation based on recent processing
    if (!self.lastProcessingTimes) {
        self.lastProcessingTimes = [];
    }
    
    const now = Date.now();
    self.lastProcessingTimes = self.lastProcessingTimes.filter(
        time => now - time < 60000 // Last minute
    );
    
    // Estimate load as percentage of time spent processing
    const processingTime = self.lastProcessingTimes.length * 100; // Assume 100ms avg
    const load = Math.min(1.0, processingTime / 60000);
    
    return load;
}

// Get average processing time
function getAverageProcessingTime() {
    if (!self.metrics || !self.metrics.processing_times.length) {
        return 0;
    }
    
    const sum = self.metrics.processing_times.reduce((a, b) => a + b, 0);
    return sum / self.metrics.processing_times.length;
}

// Performance monitoring
let performanceMonitor = null;

function startPerformanceMonitoring() {
    if (performanceMonitor) return;
    
    performanceMonitor = setInterval(() => {
        const memory = performance.memory;
        if (memory) {
            const usedMemory = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
            if (usedMemory > 0.9) {
                console.warn('High memory usage in edge worker:', usedMemory);
                // Consider clearing some caches
                cleanupOldModules();
            }
        }
    }, 10000); // Every 10 seconds
}

function cleanupOldModules() {
    // Simple cleanup - in production, implement LRU cache
    const moduleNames = Object.keys(modules);
    if (moduleNames.length > 5) {
        // Remove oldest modules
        const toRemove = moduleNames.slice(0, moduleNames.length - 3);
        for (const name of toRemove) {
            if (modules[name] && typeof modules[name].delete === 'function') {
                modules[name].delete();
            }
            delete modules[name];
        }
        console.log('Cleaned up old modules:', toRemove);
    }
}

// Start monitoring
startPerformanceMonitoring();

console.log('Edge worker initialized');