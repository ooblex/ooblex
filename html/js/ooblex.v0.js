var Ooblex = {};

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
	                session.offerSDP(stream);
	        });
		return videoElement;
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
