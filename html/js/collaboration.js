class CollaborationClient {
    constructor(options = {}) {
        this.options = {
            wsUrl: options.wsUrl || 'ws://localhost:8765',
            sessionId: options.sessionId,
            streamId: options.streamId,
            userId: options.userId || this.generateUserId(),
            userName: options.userName || 'Anonymous',
            userColor: options.userColor || this.generateUserColor(),
            onReady: options.onReady || (() => {}),
            onError: options.onError || ((err) => console.error(err)),
            ...options
        };
        
        this.ws = null;
        this.canvas = null;
        this.ctx = null;
        this.videoElement = options.videoElement;
        this.annotations = new Map();
        this.users = new Map();
        this.cursors = new Map();
        this.isDrawing = false;
        this.currentTool = 'pen';
        this.currentColor = '#FF0000';
        this.currentLineWidth = 2;
        this.playbackState = null;
        this.recordingEnabled = false;
        
        this.init();
    }
    
    init() {
        // Create canvas overlay
        this.createCanvasOverlay();
        
        // Set up WebSocket connection
        this.connect();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize UI components
        this.initializeUI();
    }
    
    createCanvasOverlay() {
        // Create canvas element
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'collaboration-canvas';
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.zIndex = '1000';
        
        // Insert canvas after video element
        if (this.videoElement) {
            this.videoElement.parentNode.insertBefore(this.canvas, this.videoElement.nextSibling);
            this.resizeCanvas();
        }
        
        this.ctx = this.canvas.getContext('2d');
    }
    
    resizeCanvas() {
        const rect = this.videoElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        this.redrawAnnotations();
    }
    
    connect() {
        this.ws = new WebSocket(this.options.wsUrl);
        
        this.ws.onopen = () => {
            // Send initialization message
            this.send({
                type: 'init',
                session_id: this.options.sessionId,
                stream_id: this.options.streamId,
                user_id: this.options.userId,
                user_info: {
                    name: this.options.userName,
                    color: this.options.userColor
                }
            });
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.ws.onerror = (error) => {
            this.options.onError(error);
        };
        
        this.ws.onclose = () => {
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connect(), 3000);
        };
    }
    
    handleMessage(message) {
        const handlers = {
            'initial_state': this.handleInitialState.bind(this),
            'user_joined': this.handleUserJoined.bind(this),
            'user_left': this.handleUserLeft.bind(this),
            'annotation_added': this.handleAnnotationAdded.bind(this),
            'annotation_deleted': this.handleAnnotationDeleted.bind(this),
            'annotations_cleared': this.handleAnnotationsCleared.bind(this),
            'cursor_update': this.handleCursorUpdate.bind(this),
            'chat_message': this.handleChatMessage.bind(this),
            'playback_update': this.handlePlaybackUpdate.bind(this),
            'recording_started': this.handleRecordingStarted.bind(this),
            'recording_stopped': this.handleRecordingStopped.bind(this)
        };
        
        // Validate message type against known handlers to prevent prototype pollution
        if (Object.prototype.hasOwnProperty.call(handlers, message.type)) {
            const handler = handlers[message.type];
            handler(message);
        }
    }
    
    handleInitialState(message) {
        // Set up users
        for (const [userId, userInfo] of Object.entries(message.users)) {
            this.users.set(userId, userInfo);
            this.updateUserList();
        }
        
        // Load existing annotations
        message.annotations.forEach(annotation => {
            this.annotations.set(annotation.id, annotation);
        });
        
        // Set playback state
        this.playbackState = message.playback_state;
        if (this.playbackState && this.videoElement) {
            this.syncPlaybackState();
        }
        
        // Set recording state
        this.recordingEnabled = message.recording_enabled;
        this.updateRecordingUI();
        
        // Redraw canvas
        this.redrawAnnotations();
        
        // Callback
        this.options.onReady();
    }
    
    handleUserJoined(message) {
        this.users.set(message.user_id, message.user_info);
        this.updateUserList();
        this.showNotification(`${message.user_info.name} joined the session`);
    }
    
    handleUserLeft(message) {
        const user = this.users.get(message.user_id);
        if (user) {
            this.showNotification(`${user.name} left the session`);
            this.users.delete(message.user_id);
            this.cursors.delete(message.user_id);
            this.updateUserList();
            this.redrawAnnotations();
        }
    }
    
    handleAnnotationAdded(message) {
        this.annotations.set(message.annotation.id, message.annotation);
        this.drawAnnotation(message.annotation);
        
        if (message.annotation.user_id !== this.options.userId) {
            this.showNotification(`${message.user_info.name} added an annotation`);
        }
    }
    
    handleAnnotationDeleted(message) {
        this.annotations.delete(message.annotation_id);
        this.redrawAnnotations();
    }
    
    handleAnnotationsCleared(message) {
        this.annotations.clear();
        this.redrawAnnotations();
        
        if (message.user_id !== this.options.userId) {
            const user = this.users.get(message.user_id);
            this.showNotification(`${user.name} cleared all annotations`);
        }
    }
    
    handleCursorUpdate(message) {
        this.cursors.set(message.cursor.user_id, message.cursor);
        this.drawCursors();
    }
    
    handleChatMessage(message) {
        this.addChatMessage(message.message, message.user_info);
        
        // Check for mentions
        if (message.message.mentions.includes(this.options.userId)) {
            this.showNotification(`${message.user_info.name} mentioned you`, 'mention');
        }
    }
    
    handlePlaybackUpdate(message) {
        this.playbackState = message.playback_state;
        
        if (message.playback_state.user_id !== this.options.userId) {
            this.syncPlaybackState();
            const user = this.users.get(message.playback_state.user_id);
            const action = message.playback_state.is_playing ? 'started playback' : 'paused playback';
            this.showNotification(`${user.name} ${action}`);
        }
    }
    
    handleRecordingStarted(message) {
        this.recordingEnabled = true;
        this.updateRecordingUI();
        
        const user = this.users.get(message.user_id);
        this.showNotification(`${user.name} started recording the session`);
    }
    
    handleRecordingStopped(message) {
        this.recordingEnabled = false;
        this.updateRecordingUI();
        
        const user = this.users.get(message.user_id);
        this.showNotification(`${user.name} stopped recording (${Math.round(message.duration)}s)`);
    }
    
    setupEventListeners() {
        // Canvas events
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this));
        
        // Video events
        if (this.videoElement) {
            this.videoElement.addEventListener('play', this.handleVideoPlay.bind(this));
            this.videoElement.addEventListener('pause', this.handleVideoPause.bind(this));
            this.videoElement.addEventListener('seeked', this.handleVideoSeeked.bind(this));
            this.videoElement.addEventListener('ratechange', this.handleVideoRateChange.bind(this));
        }
        
        // Window resize
        window.addEventListener('resize', this.resizeCanvas.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
    
    handleMouseDown(e) {
        if (this.currentTool === 'cursor') return;
        
        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.startAnnotation(x, y);
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        
        // Send cursor position
        this.send({
            type: 'cursor_move',
            x: x,
            y: y
        });
        
        // Continue drawing if in progress
        if (this.isDrawing && this.currentTool !== 'cursor') {
            this.continueAnnotation(e.clientX - rect.left, e.clientY - rect.top);
        }
    }
    
    handleMouseUp(e) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.finishAnnotation();
        }
    }
    
    handleMouseLeave(e) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.finishAnnotation();
        }
    }
    
    handleTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.canvas.dispatchEvent(mouseEvent);
    }
    
    handleTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.canvas.dispatchEvent(mouseEvent);
    }
    
    handleTouchEnd(e) {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        this.canvas.dispatchEvent(mouseEvent);
    }
    
    handleVideoPlay() {
        if (!this.playbackState || !this.playbackState.is_playing) {
            this.send({
                type: 'playback_control',
                control_type: 'play'
            });
        }
    }
    
    handleVideoPause() {
        if (this.playbackState && this.playbackState.is_playing) {
            this.send({
                type: 'playback_control',
                control_type: 'pause'
            });
        }
    }
    
    handleVideoSeeked() {
        this.send({
            type: 'playback_control',
            control_type: 'seek',
            position: this.videoElement.currentTime
        });
    }
    
    handleVideoRateChange() {
        this.send({
            type: 'playback_control',
            control_type: 'speed',
            speed: this.videoElement.playbackRate
        });
    }
    
    handleKeyDown(e) {
        // Keyboard shortcuts
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'z':
                    e.preventDefault();
                    this.undo();
                    break;
                case 'c':
                    e.preventDefault();
                    this.clearAnnotations();
                    break;
            }
        }
    }
    
    startAnnotation(x, y) {
        this.currentAnnotation = {
            type: this.currentTool,
            data: {
                color: this.currentColor,
                lineWidth: this.currentLineWidth,
                points: [{x: x, y: y}]
            },
            stream_time: this.videoElement ? this.videoElement.currentTime : 0
        };
        
        if (this.currentTool === 'rectangle' || this.currentTool === 'circle') {
            this.currentAnnotation.data.startX = x;
            this.currentAnnotation.data.startY = y;
        }
    }
    
    continueAnnotation(x, y) {
        if (!this.currentAnnotation) return;
        
        if (this.currentTool === 'pen' || this.currentTool === 'arrow') {
            this.currentAnnotation.data.points.push({x: x, y: y});
            
            // Draw incremental update
            this.ctx.strokeStyle = this.currentAnnotation.data.color;
            this.ctx.lineWidth = this.currentAnnotation.data.lineWidth;
            this.ctx.lineCap = 'round';
            this.ctx.lineJoin = 'round';
            
            const points = this.currentAnnotation.data.points;
            if (points.length >= 2) {
                this.ctx.beginPath();
                this.ctx.moveTo(points[points.length - 2].x, points[points.length - 2].y);
                this.ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
                this.ctx.stroke();
            }
        } else if (this.currentTool === 'rectangle' || this.currentTool === 'circle') {
            this.currentAnnotation.data.endX = x;
            this.currentAnnotation.data.endY = y;
            this.redrawAnnotations();
            this.drawTempShape();
        }
    }
    
    finishAnnotation() {
        if (!this.currentAnnotation) return;
        
        // Normalize coordinates to relative values
        const rect = this.canvas.getBoundingClientRect();
        const normalizedAnnotation = this.normalizeAnnotation(this.currentAnnotation, rect);
        
        // Send annotation to server
        this.send({
            type: 'annotation',
            annotation_type: normalizedAnnotation.type,
            data: normalizedAnnotation.data,
            stream_time: normalizedAnnotation.stream_time
        });
        
        this.currentAnnotation = null;
    }
    
    normalizeAnnotation(annotation, rect) {
        const normalized = JSON.parse(JSON.stringify(annotation));
        
        if (annotation.data.points) {
            normalized.data.points = annotation.data.points.map(p => ({
                x: p.x / rect.width,
                y: p.y / rect.height
            }));
        }
        
        if (annotation.data.startX !== undefined) {
            normalized.data.startX = annotation.data.startX / rect.width;
            normalized.data.startY = annotation.data.startY / rect.height;
            normalized.data.endX = annotation.data.endX / rect.width;
            normalized.data.endY = annotation.data.endY / rect.height;
        }
        
        return normalized;
    }
    
    drawAnnotation(annotation) {
        const rect = this.canvas.getBoundingClientRect();
        const user = this.users.get(annotation.user_id);
        
        this.ctx.save();
        this.ctx.strokeStyle = annotation.data.color || '#FF0000';
        this.ctx.lineWidth = annotation.data.lineWidth || 2;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        switch(annotation.type) {
            case 'pen':
                this.drawPath(annotation.data.points, rect);
                break;
            case 'arrow':
                this.drawArrow(annotation.data.points, rect);
                break;
            case 'rectangle':
                this.drawRectangle(annotation.data, rect);
                break;
            case 'circle':
                this.drawCircle(annotation.data, rect);
                break;
            case 'text':
                this.drawText(annotation.data, rect);
                break;
        }
        
        this.ctx.restore();
    }
    
    drawPath(points, rect) {
        if (points.length < 2) return;
        
        this.ctx.beginPath();
        this.ctx.moveTo(points[0].x * rect.width, points[0].y * rect.height);
        
        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i].x * rect.width, points[i].y * rect.height);
        }
        
        this.ctx.stroke();
    }
    
    drawArrow(points, rect) {
        if (points.length < 2) return;
        
        // Draw line
        this.drawPath(points, rect);
        
        // Draw arrowhead
        const lastPoint = points[points.length - 1];
        const secondLastPoint = points[points.length - 2];
        
        const angle = Math.atan2(
            lastPoint.y - secondLastPoint.y,
            lastPoint.x - secondLastPoint.x
        );
        
        const arrowLength = 15;
        const arrowAngle = Math.PI / 6;
        
        this.ctx.beginPath();
        this.ctx.moveTo(lastPoint.x * rect.width, lastPoint.y * rect.height);
        this.ctx.lineTo(
            lastPoint.x * rect.width - arrowLength * Math.cos(angle - arrowAngle),
            lastPoint.y * rect.height - arrowLength * Math.sin(angle - arrowAngle)
        );
        this.ctx.moveTo(lastPoint.x * rect.width, lastPoint.y * rect.height);
        this.ctx.lineTo(
            lastPoint.x * rect.width - arrowLength * Math.cos(angle + arrowAngle),
            lastPoint.y * rect.height - arrowLength * Math.sin(angle + arrowAngle)
        );
        this.ctx.stroke();
    }
    
    drawRectangle(data, rect) {
        const x = data.startX * rect.width;
        const y = data.startY * rect.height;
        const width = (data.endX - data.startX) * rect.width;
        const height = (data.endY - data.startY) * rect.height;
        
        this.ctx.strokeRect(x, y, width, height);
    }
    
    drawCircle(data, rect) {
        const centerX = (data.startX + data.endX) / 2 * rect.width;
        const centerY = (data.startY + data.endY) / 2 * rect.height;
        const radiusX = Math.abs(data.endX - data.startX) / 2 * rect.width;
        const radiusY = Math.abs(data.endY - data.startY) / 2 * rect.height;
        
        this.ctx.beginPath();
        this.ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
        this.ctx.stroke();
    }
    
    drawText(data, rect) {
        this.ctx.font = `${data.fontSize || 16}px Arial`;
        this.ctx.fillStyle = data.color;
        this.ctx.fillText(data.text, data.x * rect.width, data.y * rect.height);
    }
    
    drawTempShape() {
        if (!this.currentAnnotation) return;
        
        const rect = this.canvas.getBoundingClientRect();
        this.ctx.save();
        this.ctx.strokeStyle = this.currentAnnotation.data.color;
        this.ctx.lineWidth = this.currentAnnotation.data.lineWidth;
        this.ctx.setLineDash([5, 5]);
        
        if (this.currentAnnotation.type === 'rectangle') {
            this.drawRectangle({
                startX: this.currentAnnotation.data.startX / rect.width,
                startY: this.currentAnnotation.data.startY / rect.height,
                endX: this.currentAnnotation.data.endX / rect.width,
                endY: this.currentAnnotation.data.endY / rect.height
            }, rect);
        } else if (this.currentAnnotation.type === 'circle') {
            this.drawCircle({
                startX: this.currentAnnotation.data.startX / rect.width,
                startY: this.currentAnnotation.data.startY / rect.height,
                endX: this.currentAnnotation.data.endX / rect.width,
                endY: this.currentAnnotation.data.endY / rect.height
            }, rect);
        }
        
        this.ctx.restore();
    }
    
    drawCursors() {
        const rect = this.canvas.getBoundingClientRect();
        
        // Clear cursor layer (we'll need a separate canvas for this in production)
        this.cursors.forEach((cursor, userId) => {
            if (userId === this.options.userId) return;
            
            const user = this.users.get(userId);
            if (!user) return;
            
            const x = cursor.x * rect.width;
            const y = cursor.y * rect.height;
            
            // Draw cursor
            this.ctx.save();
            this.ctx.fillStyle = user.color;
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;
            
            // Draw pointer shape
            this.ctx.beginPath();
            this.ctx.moveTo(x, y);
            this.ctx.lineTo(x + 10, y + 10);
            this.ctx.lineTo(x + 4, y + 10);
            this.ctx.closePath();
            
            this.ctx.stroke();
            this.ctx.fill();
            
            // Draw user name
            this.ctx.font = '12px Arial';
            this.ctx.fillStyle = '#000000';
            this.ctx.fillText(user.name, x + 15, y + 15);
            
            this.ctx.restore();
        });
    }
    
    redrawAnnotations() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw all annotations
        this.annotations.forEach(annotation => {
            this.drawAnnotation(annotation);
        });
        
        // Draw cursors
        this.drawCursors();
    }
    
    syncPlaybackState() {
        if (!this.playbackState || !this.videoElement) return;
        
        if (this.playbackState.is_playing && this.videoElement.paused) {
            this.videoElement.play();
        } else if (!this.playbackState.is_playing && !this.videoElement.paused) {
            this.videoElement.pause();
        }
        
        if (Math.abs(this.videoElement.currentTime - this.playbackState.position) > 1) {
            this.videoElement.currentTime = this.playbackState.position;
        }
        
        if (this.videoElement.playbackRate !== this.playbackState.speed) {
            this.videoElement.playbackRate = this.playbackState.speed;
        }
    }
    
    initializeUI() {
        // Create toolbar
        this.toolbar = this.createToolbar();
        
        // Create user list
        this.userList = this.createUserList();
        
        // Create chat panel
        this.chatPanel = this.createChatPanel();
        
        // Create notification area
        this.notificationArea = this.createNotificationArea();
    }
    
    createToolbar() {
        const toolbar = document.createElement('div');
        toolbar.className = 'collab-toolbar';
        toolbar.innerHTML = `
            <div class="tool-group">
                <button class="tool-btn" data-tool="cursor" title="Cursor">
                    <i class="icon-cursor"></i>
                </button>
                <button class="tool-btn active" data-tool="pen" title="Pen">
                    <i class="icon-pen"></i>
                </button>
                <button class="tool-btn" data-tool="arrow" title="Arrow">
                    <i class="icon-arrow"></i>
                </button>
                <button class="tool-btn" data-tool="rectangle" title="Rectangle">
                    <i class="icon-rectangle"></i>
                </button>
                <button class="tool-btn" data-tool="circle" title="Circle">
                    <i class="icon-circle"></i>
                </button>
                <button class="tool-btn" data-tool="text" title="Text">
                    <i class="icon-text"></i>
                </button>
            </div>
            <div class="tool-group">
                <input type="color" class="color-picker" value="#FF0000">
                <input type="range" class="line-width" min="1" max="10" value="2">
            </div>
            <div class="tool-group">
                <button class="action-btn" id="clear-btn" title="Clear All">
                    <i class="icon-clear"></i>
                </button>
                <button class="action-btn" id="record-btn" title="Record Session">
                    <i class="icon-record"></i>
                </button>
            </div>
        `;
        
        // Add event listeners
        toolbar.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                toolbar.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentTool = btn.dataset.tool;
                this.canvas.style.cursor = this.currentTool === 'cursor' ? 'default' : 'crosshair';
            });
        });
        
        toolbar.querySelector('.color-picker').addEventListener('change', (e) => {
            this.currentColor = e.target.value;
        });
        
        toolbar.querySelector('.line-width').addEventListener('input', (e) => {
            this.currentLineWidth = parseInt(e.target.value);
        });
        
        toolbar.querySelector('#clear-btn').addEventListener('click', () => {
            this.clearAnnotations();
        });
        
        toolbar.querySelector('#record-btn').addEventListener('click', () => {
            this.toggleRecording();
        });
        
        document.body.appendChild(toolbar);
        return toolbar;
    }
    
    createUserList() {
        const userList = document.createElement('div');
        userList.className = 'collab-user-list';
        userList.innerHTML = '<h3>Active Users</h3><ul id="user-list-items"></ul>';
        document.body.appendChild(userList);
        return userList;
    }
    
    createChatPanel() {
        const chatPanel = document.createElement('div');
        chatPanel.className = 'collab-chat-panel';
        chatPanel.innerHTML = `
            <h3>Chat</h3>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="chat-input" placeholder="Type a message...">
                <button class="chat-send" id="chat-send">Send</button>
            </div>
        `;
        
        const input = chatPanel.querySelector('#chat-input');
        const sendBtn = chatPanel.querySelector('#chat-send');
        
        const sendMessage = () => {
            const text = input.value.trim();
            if (text) {
                // Extract mentions
                const mentions = [];
                const mentionRegex = /@(\w+)/g;
                let match;
                while ((match = mentionRegex.exec(text)) !== null) {
                    const userName = match[1];
                    this.users.forEach((user, userId) => {
                        if (user.name.toLowerCase() === userName.toLowerCase()) {
                            mentions.push(userId);
                        }
                    });
                }
                
                this.send({
                    type: 'chat',
                    text: text,
                    mentions: mentions
                });
                
                input.value = '';
            }
        };
        
        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        document.body.appendChild(chatPanel);
        return chatPanel;
    }
    
    createNotificationArea() {
        const notificationArea = document.createElement('div');
        notificationArea.className = 'collab-notifications';
        document.body.appendChild(notificationArea);
        return notificationArea;
    }
    
    updateUserList() {
        const listContainer = document.querySelector('#user-list-items');
        listContainer.innerHTML = '';
        
        this.users.forEach((user, userId) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="user-color" style="background-color: ${user.color}"></span>
                <span class="user-name">${user.name}</span>
                ${userId === this.options.userId ? '<span class="user-you">(You)</span>' : ''}
            `;
            listContainer.appendChild(li);
        });
    }
    
    addChatMessage(message, userInfo) {
        const messagesContainer = document.querySelector('#chat-messages');
        const messageEl = document.createElement('div');
        messageEl.className = 'chat-message';
        messageEl.innerHTML = `
            <span class="chat-user" style="color: ${userInfo.color}">${userInfo.name}:</span>
            <span class="chat-text">${this.escapeHtml(message.message)}</span>
        `;
        
        messagesContainer.appendChild(messageEl);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    showNotification(text, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = text;
        
        this.notificationArea.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    updateRecordingUI() {
        const recordBtn = document.querySelector('#record-btn');
        if (this.recordingEnabled) {
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<i class="icon-stop"></i>';
        } else {
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<i class="icon-record"></i>';
        }
    }
    
    clearAnnotations() {
        if (confirm('Clear all annotations?')) {
            this.send({
                type: 'clear_annotations'
            });
        }
    }
    
    toggleRecording() {
        if (this.recordingEnabled) {
            this.send({
                type: 'stop_recording'
            });
        } else {
            this.send({
                type: 'start_recording'
            });
        }
    }
    
    undo() {
        // Find last annotation by current user
        const userAnnotations = Array.from(this.annotations.values())
            .filter(a => a.user_id === this.options.userId)
            .sort((a, b) => b.timestamp - a.timestamp);
        
        if (userAnnotations.length > 0) {
            this.send({
                type: 'delete_annotation',
                annotation_id: userAnnotations[0].id
            });
        }
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
    
    generateUserId() {
        // Use crypto.getRandomValues for secure random ID generation
        const array = new Uint8Array(12);
        crypto.getRandomValues(array);
        return 'user_' + Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('').slice(0, 9);
    }

    generateUserColor() {
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE'];
        // Use crypto.getRandomValues for secure random selection
        const array = new Uint32Array(1);
        crypto.getRandomValues(array);
        return colors[array[0] % colors.length];
    }
    
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
    
    destroy() {
        // Clean up WebSocket
        if (this.ws) {
            this.ws.close();
        }
        
        // Remove UI elements
        if (this.canvas) this.canvas.remove();
        if (this.toolbar) this.toolbar.remove();
        if (this.userList) this.userList.remove();
        if (this.chatPanel) this.chatPanel.remove();
        if (this.notificationArea) this.notificationArea.remove();
        
        // Remove event listeners
        window.removeEventListener('resize', this.resizeCanvas);
        document.removeEventListener('keydown', this.handleKeyDown);
    }
}

// CSS styles
const collaborationStyles = `
.collaboration-canvas {
    cursor: crosshair;
}

.collab-toolbar {
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    background: white;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    display: flex;
    gap: 20px;
    z-index: 1001;
}

.tool-group {
    display: flex;
    gap: 5px;
    align-items: center;
}

.tool-btn, .action-btn {
    width: 40px;
    height: 40px;
    border: none;
    background: #f0f0f0;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
}

.tool-btn:hover, .action-btn:hover {
    background: #e0e0e0;
}

.tool-btn.active {
    background: #4ECDC4;
    color: white;
}

.color-picker {
    width: 40px;
    height: 40px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.line-width {
    width: 100px;
}

.collab-user-list {
    position: fixed;
    top: 80px;
    right: 20px;
    background: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    width: 200px;
    z-index: 1001;
}

.collab-user-list h3 {
    margin: 0 0 10px 0;
    font-size: 16px;
}

.collab-user-list ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.collab-user-list li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
}

.user-color {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.user-you {
    font-size: 12px;
    color: #999;
    margin-left: auto;
}

.collab-chat-panel {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    width: 300px;
    height: 400px;
    display: flex;
    flex-direction: column;
    z-index: 1001;
}

.collab-chat-panel h3 {
    margin: 0 0 10px 0;
    font-size: 16px;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 10px;
    padding: 10px;
    background: #f9f9f9;
    border-radius: 4px;
}

.chat-message {
    margin-bottom: 8px;
}

.chat-user {
    font-weight: bold;
    margin-right: 5px;
}

.chat-input-container {
    display: flex;
    gap: 5px;
}

.chat-input {
    flex: 1;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.chat-send {
    padding: 8px 15px;
    background: #4ECDC4;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.collab-notifications {
    position: fixed;
    top: 80px;
    left: 20px;
    z-index: 1002;
}

.notification {
    background: white;
    padding: 10px 15px;
    border-radius: 4px;
    margin-bottom: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s;
}

.notification-mention {
    background: #FFF3CD;
    border-left: 4px solid #FFA07A;
}

.notification.fade-out {
    opacity: 0;
    transform: translateX(-20px);
}

.recording {
    background: #FF6B6B !important;
    color: white;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = collaborationStyles;
document.head.appendChild(styleSheet);

// Export for use
window.CollaborationClient = CollaborationClient;