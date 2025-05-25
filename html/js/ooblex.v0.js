var Ooblex = {};

// Edge Computing Module
Ooblex.Edge = new (function(){
	var edge = {};
	
	// Configuration
	edge.config = {
		serverUrl: 'https://api.ooblex.com:8090',
		wasmPath: '/wasm/',
		enableEdge: true,
		fallbackToCloud: true,
		performanceThreshold: 100, // ms
		modules: {}
	};
	
	// WASM module cache
	edge.moduleCache = {};
	edge.workers = {};
	edge.capabilities = new Set();
	
	// Initialize edge computing
	edge.init = async function(config = {}) {
		Object.assign(edge.config, config);
		
		// Check WebAssembly support
		if (!edge.checkWasmSupport()) {
			console.warn('WebAssembly not supported, falling back to cloud processing');
			edge.config.enableEdge = false;
			return false;
		}
		
		// Check available modules from server
		try {
			const response = await fetch(`${edge.config.serverUrl}/api/edge/modules`);
			const data = await response.json();
			
			for (const module of data.modules) {
				edge.capabilities.add(...module.capabilities);
			}
			
			console.log('Edge computing initialized with capabilities:', Array.from(edge.capabilities));
			return true;
		} catch (error) {
			console.error('Failed to initialize edge computing:', error);
			edge.config.enableEdge = false;
			return false;
		}
	};
	
	// Check WebAssembly support
	edge.checkWasmSupport = function() {
		try {
			if (typeof WebAssembly === 'object' &&
				typeof WebAssembly.instantiate === 'function') {
				const module = new WebAssembly.Module(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));
				if (module instanceof WebAssembly.Module) {
					return new WebAssembly.Instance(module) instanceof WebAssembly.Instance;
				}
			}
		} catch (e) {}
		return false;
	};
	
	// Load WASM module
	edge.loadModule = async function(moduleName) {
		if (edge.moduleCache[moduleName]) {
			return edge.moduleCache[moduleName];
		}
		
		try {
			const response = await fetch(`${edge.config.serverUrl}/api/edge/modules/${moduleName}`);
			if (!response.ok) throw new Error(`Failed to load module: ${moduleName}`);
			
			const moduleText = await response.text();
			const moduleFunction = eval(`(${moduleText})`);
			const module = await moduleFunction();
			
			edge.moduleCache[moduleName] = module;
			console.log(`Loaded WASM module: ${moduleName}`);
			
			return module;
		} catch (error) {
			console.error(`Failed to load module ${moduleName}:`, error);
			throw error;
		}
	};
	
	// Create or get worker for processing
	edge.getWorker = function(taskType) {
		if (!edge.workers[taskType]) {
			edge.workers[taskType] = new Worker('/js/edge_worker.js');
			edge.workers[taskType].postMessage({ 
				type: 'init', 
				config: edge.config 
			});
		}
		return edge.workers[taskType];
	};
	
	// Process with edge computing
	edge.process = async function(taskType, data, options = {}) {
		if (!edge.config.enableEdge || !edge.capabilities.has(taskType)) {
			// Fallback to cloud
			return edge.processCloud(taskType, data, options);
		}
		
		const startTime = performance.now();
		
		try {
			// Get processing instructions from server
			const coordResponse = await fetch(`${edge.config.serverUrl}/api/edge/process`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ type: taskType, priority: options.priority || 'normal' })
			});
			
			const coordination = await coordResponse.json();
			
			if (coordination.mode === 'cloud') {
				return edge.processCloud(taskType, data, options);
			}
			
			// Process on edge
			const result = await edge.processEdge(taskType, data, coordination, options);
			
			const processingTime = performance.now() - startTime;
			
			// Report performance
			edge.reportPerformance(taskType, processingTime, true);
			
			// Check if we should fallback based on performance
			if (processingTime > edge.config.performanceThreshold && edge.config.fallbackToCloud) {
				console.warn(`Edge processing took ${processingTime}ms, consider cloud fallback`);
			}
			
			return result;
			
		} catch (error) {
			console.error('Edge processing failed:', error);
			
			// Report failure
			edge.reportPerformance(taskType, performance.now() - startTime, false);
			
			// Fallback to cloud if enabled
			if (edge.config.fallbackToCloud) {
				return edge.processCloud(taskType, data, options);
			}
			
			throw error;
		}
	};
	
	// Process on edge
	edge.processEdge = function(taskType, data, coordination, options) {
		return new Promise((resolve, reject) => {
			const worker = edge.getWorker(taskType);
			const messageId = Math.random().toString(36).substr(2, 9);
			
			const timeout = setTimeout(() => {
				worker.removeEventListener('message', messageHandler);
				reject(new Error('Edge processing timeout'));
			}, coordination.fallback?.timeout || 5000);
			
			const messageHandler = (event) => {
				if (event.data.id === messageId) {
					clearTimeout(timeout);
					worker.removeEventListener('message', messageHandler);
					
					if (event.data.error) {
						reject(new Error(event.data.error));
					} else {
						resolve(event.data.result);
					}
				}
			};
			
			worker.addEventListener('message', messageHandler);
			
			worker.postMessage({
				id: messageId,
				type: 'process',
				taskType: taskType,
				module: coordination.module,
				data: data,
				options: options
			});
		});
	};
	
	// Process on cloud
	edge.processCloud = async function(taskType, data, options) {
		const response = await fetch(`${edge.config.serverUrl}/api/process`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				type: taskType,
				data: data,
				options: options
			})
		});
		
		if (!response.ok) {
			throw new Error(`Cloud processing failed: ${response.statusText}`);
		}
		
		return response.json();
	};
	
	// Report performance metrics
	edge.reportPerformance = async function(taskType, processingTime, success) {
		try {
			// Get worker ID from local storage or generate new one
			let workerId = localStorage.getItem('edge_worker_id');
			if (!workerId) {
				workerId = 'browser_' + Math.random().toString(36).substr(2, 9);
				localStorage.setItem('edge_worker_id', workerId);
			}
			
			await fetch(`${edge.config.serverUrl}/api/edge/workers/${workerId}/report`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					processing_stats: {
						task_type: taskType,
						processing_time: processingTime,
						success: success,
						timestamp: new Date().toISOString()
					}
				})
			});
		} catch (error) {
			console.error('Failed to report performance:', error);
		}
	};
	
	// Specific edge processing functions
	edge.detectFaces = function(imageData, options = {}) {
		return edge.process('face_detection', imageData, options);
	};
	
	edge.applyStyleTransfer = function(imageData, style, options = {}) {
		return edge.process('style_transfer', { 
			imageData: imageData, 
			style: style 
		}, options);
	};
	
	edge.blurBackground = function(imageData, options = {}) {
		return edge.process('background_blur', imageData, options);
	};
	
	// Preload modules for better performance
	edge.preloadModules = async function(moduleNames) {
		const promises = moduleNames.map(name => edge.loadModule(name));
		return Promise.all(promises);
	};
	
	// Clean up resources
	edge.cleanup = function() {
		// Terminate workers
		for (const worker of Object.values(edge.workers)) {
			worker.terminate();
		}
		edge.workers = {};
		edge.moduleCache = {};
	};
	
	return edge;
})();

Ooblex.Media = new (function(){
	
	var session = {};

	session.onReady = function(){
		console.log("Stream Publishing");
	}

	session.generateToken = function(){
	  var text = "";
	  var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
	  for (var i = 0; i < 12; i++){
	    text += possible.charAt(Math.floor(Math.random() * possible.length));
	  }
	  return text;
	};
	
	session.connect = function(token = null, callback = session.createStream){
		if (token == null){
			token = session.generateToken(); // If token was not provided, generate one.
		}
		session.ws = new WebSocket("wss://api.ooblex.com:8100");		
		session.pc = new RTCPeerConnection({'iceServers': [{urls: "stun:stun.l.google.com:19302"}, {urls: "stun:numb.viagenie.ca:3478"}]});
		session.pc.onclose = function(){console.log("pc closed");};
		session.ws.onopen = function(){
		        console.log("connected to video server");
		        var data = {};
		        data.token = token;
		        session.ws.send(JSON.stringify(data));
		}
		session.ws.onmessage = function (evt) {
		        var msg = evt.data;
		        msg = JSON.parse(msg);
		        console.log("incoming: "+msg);
		        if (msg.request){
		                if (msg.request=="offerSDP"){
		                        callback(); // Need to create stream before an SDP offer can be generated
		                } else if (msg.request=="publish"){
		                        if (msg.jsep){
		                                session.publishStream(msg.jsep)
		                        } else {
		                                console.log("No SDP provided; error");
		                        }
		                } else if (msg.request=="nothing"){
					session.onReady();
				}
		        }
		}
		return session;
	};	
	session.createStream = function(videoElement = null){ // stream is used to generated an SDP
		if (videoElement == null){
			var videoElement = document.createElement('video');
			videoElement.autoplay = true;
			videoElement.muted = true;
			var body = document.getElementsByTagName('body')[0];
			body.appendChild(videoElement);
		}
	        navigator.getUserMedia = (  navigator.getUserMedia || navigator.mediaDevices.getUserMedia);
	        navigator.mediaDevices.getUserMedia({
	                video: {frameRate: { ideal: 15, max: 15 }, width: {ideal: 640}, height: {ideal: 480}},
	                audio: true
	        }).then(function success(stream) {
	                videoElement.srcObject = stream;
	                
	                // Initialize edge processing if available
	                if (Ooblex.Edge.config.enableEdge) {
	                	session.setupEdgeProcessing(videoElement, stream);
	                }
	                
	                session.offerSDP(stream);
	        });
		return videoElement;
	};
	
	session.setupEdgeProcessing = function(videoElement, stream) {
		// Create canvas for edge processing
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');
		canvas.width = 640;
		canvas.height = 480;
		
		// Edge processing state
		session.edgeProcessing = {
			enabled: false,
			mode: null,
			canvas: canvas,
			ctx: ctx,
			stream: stream
		};
		
		// Process frame with edge computing
		session.processFrame = async function() {
			if (!session.edgeProcessing.enabled) return;
			
			// Draw current frame to canvas
			ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
			const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
			
			try {
				let processedData;
				
				switch (session.edgeProcessing.mode) {
					case 'face_detection':
						const faces = await Ooblex.Edge.detectFaces(imageData.data);
						// Draw face boundaries
						session.drawFaceDetection(ctx, faces);
						break;
						
					case 'background_blur':
						processedData = await Ooblex.Edge.blurBackground(imageData.data);
						const processedImageData = new ImageData(
							new Uint8ClampedArray(processedData),
							canvas.width,
							canvas.height
						);
						ctx.putImageData(processedImageData, 0, 0);
						break;
						
					case 'style_transfer':
						processedData = await Ooblex.Edge.applyStyleTransfer(
							imageData.data, 
							session.edgeProcessing.style || 'sketch'
						);
						const styledImageData = new ImageData(
							new Uint8ClampedArray(processedData),
							canvas.width,
							canvas.height
						);
						ctx.putImageData(styledImageData, 0, 0);
						break;
				}
			} catch (error) {
				console.error('Edge processing error:', error);
			}
			
			// Continue processing
			if (session.edgeProcessing.enabled) {
				requestAnimationFrame(session.processFrame);
			}
		};
		
		// Draw face detection results
		session.drawFaceDetection = function(ctx, faces) {
			ctx.strokeStyle = '#00ff00';
			ctx.lineWidth = 2;
			
			for (const face of faces) {
				ctx.strokeRect(face.x, face.y, face.width, face.height);
				
				// Draw confidence
				ctx.fillStyle = '#00ff00';
				ctx.font = '12px Arial';
				ctx.fillText(
					`${(face.confidence * 100).toFixed(1)}%`,
					face.x,
					face.y - 5
				);
			}
		};
	};
	
	// Enable edge face detection
	session.enableEdgeFaceDetection = async function() {
		if (!Ooblex.Edge.config.enableEdge) {
			console.warn('Edge computing not available');
			return false;
		}
		
		await Ooblex.Edge.preloadModules(['face_detection']);
		session.edgeProcessing.enabled = true;
		session.edgeProcessing.mode = 'face_detection';
		session.processFrame();
		return true;
	};
	
	// Enable edge background blur
	session.enableEdgeBackgroundBlur = async function() {
		if (!Ooblex.Edge.config.enableEdge) {
			console.warn('Edge computing not available');
			return false;
		}
		
		await Ooblex.Edge.preloadModules(['background_blur']);
		session.edgeProcessing.enabled = true;
		session.edgeProcessing.mode = 'background_blur';
		session.processFrame();
		return true;
	};
	
	// Enable edge style transfer
	session.enableEdgeStyleTransfer = async function(style = 'sketch') {
		if (!Ooblex.Edge.config.enableEdge) {
			console.warn('Edge computing not available');
			return false;
		}
		
		await Ooblex.Edge.preloadModules(['style_transfer_lite']);
		session.edgeProcessing.enabled = true;
		session.edgeProcessing.mode = 'style_transfer';
		session.edgeProcessing.style = style;
		session.processFrame();
		return true;
	};
	
	// Disable edge processing
	session.disableEdgeProcessing = function() {
		session.edgeProcessing.enabled = false;
		session.edgeProcessing.mode = null;
	};
	session.offerSDP = function(stream){  // publisher
			session.pc.addStream(stream);
			session.pc.onicecandidate = session.onIceCandidate;
			session.pc.createOffer(function(description){
					session.pc.setLocalDescription(description, function (){
							session.publishOffer(description);
					}, function(){});
			}, session.errorHandler);
	};
	session.publishOffer = function(description){
			console.log("publishing SDP Offer");
			console.log(description);
			var data = {};
			data.message = "Offering Requested SDP"
			data.jsep = description;
			console.log(data);
			session.ws.send(JSON.stringify(data));
	};
	session.onIceCandidate = function(event){ // deprecated, but chrome still uses it.
			console.log("onIceCandidate Event");
			console.log(event);
			if (event.candidate==null){console.log("Ignoring Ice Event");return;}
			var data = {};
			data.candidate = event.candidate;
			console.log(data);
			session.ws.send(JSON.stringify(data));
	};
	session.publishStream = function(description){
			session.pc.setRemoteDescription(new RTCSessionDescription(description), function(){
					console.log("Starting Video Stream Publishing");
					}, session.errorHandler);

	};
	session.errorHandler = function(error){
		console.log("Error:");
		console.log(error);
	};
	return session;
})();


Ooblex.Brain = new (function(){
	var session = {};
	
	session.onOpen = function(evt){console.log("Connected to Ooblex Brain Service.");};
	session.onClose = function(evt){console.log("Closed connection with Ooblex Brain Service");};
	session.onError = function(evt){console.log("Error: " + evt.data + '\n');}

	session.onMessage = function(evt){console.log("Mesage: "+ evt.data + '\n');};

	session.connect = function (){
	    session.ws = new WebSocket("wss://api.ooblex.com:8800/");
	    session.ws.onopen = function(evt) { session.onOpen(evt) };
	    session.ws.onclose = function(evt) { session.onClose(evt) };
	    session.ws.onmessage = function(evt) { session.onMessage(evt) };
	    session.ws.onerror = function(evt) { session.onError(evt) };
	}

  	session.onError = function (evt){
    		console.log('error: ' + evt.data + '\n');
		session.ws.close();
		document.myform.disconnectButton.disabled = true;
 	}

  	session.doSend = function (message){
    		console.log(message + '\n'); 
    		session.ws.send(message);
  	}


  	session.doFace =  function (token){ // continue off code by repeating this update. getting it working; rabbit MQ simultanous issue. writing email to IBM. finishing up doc and module. then get the scaling going. Not sure about custom 3d models yet. 
			session.ws.send("FaceOn:"+(token));
    		console.log("Toggling Face Detection State");
  	}
	session.doTrump =  function (token){ // continue off code by repeating this update. getting it working; rabbit MQ simultanous issue. writing email to IBM. finishing up doc and module. then get the scaling going. Not sure about custom 3d models yet.
                        session.ws.send("TrumpOn:"+(token));
                console.log("Toggling Face Detection State");
        }

	session.doSwift =  function (token){ // continue off code by repeating this update. getting it working; rabbit MQ simultanous issue. writing email to IBM. finishing up doc and module. then get the scaling going. Not sure about custom 3d models yet.
                        session.ws.send("SwiftOn:"+(token));
                console.log("Toggling Face Detection State");
        }



  objectState = false;
  session.doObject = function(token){
                session.ws.send("ObjectOn:"+(token));
                objectState = true;
                document.myform.objectOnButton.value="Turn Object Detect Off";
		document.myform.objectOnButton.className = "onstate";
        console.log("Toggling Object Detection State");
  }

  var speedState = false;
  session.doSpeed = function(token){
                session.ws.send("speedFast:"+(token));
                speedState = true;
                document.myform.speedOnButton.value="Slow Down";
                document.myform.speedOnButton.className = "onstate";
        console.log("Toggling Object Detection State");
  }


  session.doImage = function(token){
	session.ws.send("saveImage:"+(token));
	console.log("Requesting Image");
  }


   session.doDisconnect = function(){
		session.ws.close();
		//pc.close();
		//ws.close();
   }

	return session;
})();
