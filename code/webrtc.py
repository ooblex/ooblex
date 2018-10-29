from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import binascii
import threading
from multiprocessing import Process
import time
import sys
import ssl
import random
import asyncio
import websockets
import json
import uuid
import logging
import datetime
import socket
from concurrent.futures._base import TimeoutError
import config
#logging.basicConfig(level=logging.DEBUG)

global peers, sessions, rooms, KEEPALIVE_TIMEOUT, SERVER_ADDR, queue, ices, plugins

############### Global data ###############

# Format: {uid: (Peer WebSocketServerProtocol,
#                remote_address,
#                <'session'|room_id|None>)}
peers = dict()
# Format: {caller_uid: callee_uid,
#          callee_uid: caller_uid}
# Bidirectional mapping between the two peers
sessions = None
piduid = dict()
# Format: {room_id: {peer1_id, peer2_id, peer3_id, ...}}
# Room dict with a set of peers in each room
rooms = dict()
trans = dict()
plugins = dict()
sdps = dict()
states = dict()
ices = dict()
pubids = dict()
codec = dict()
rtsp = dict()
queue = []

global sock

server_address = "/root/ooblex/ux-janusapi"
try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.connect(server_address)
except:
        print("Can't connect socket server yet")
        sys.exit()


async def hello_peer(ws):
    '''
    Exchange Token, register peer
    '''
    global peers, sessions, rooms, trans
    raddr = ws.remote_address
    jdata = await ws.recv()
    try:
        data = json.loads(jdata)
    except:
        await ws.close(code=1002, reason='invalid protocol; not JSON')
        raise Exception("Not JSON from {!r}".format(raddr))
    #print(data)
    try:
    	uid = str(data['token'])
    except:
        await ws.close(code=1003, reason='no token provided on initial connect')
        raise Exception("Invalid hello from {!r}".format(raddr))
    if "video_codec" in data:
        codec[uid] = data['video_codec']
    if not uid or uid in peers or uid.split() != [uid]: # no whitespace
        await ws.close(code=1004, reason='invalid peer uid; maybe already in use')
        raise Exception("Invalid uid {!r} from {!r}".format(uid, raddr))
    attachPlugin(uid)
    await ws.send(json.dumps({"message":'Welcome', "request":"offerSDP"}))
    return uid, tid

async def cleanup_session(uid):
	wso, oaddr, _ = peers[uid]
	del peers[uid]
	await wso.close()

async def recv_msg_ping(ws, raddr, uid):
    '''
    Wait for a message forever, and send a regular ping to prevent bad routers
    from closing the connection.
    '''
    msg = None
    while msg is None:
        try:
            msg = await asyncio.wait_for(ws.recv(), 30)
        except Exception as e:
            if not e:
                print("Was the connection lost?")
                return "None"
            print("EXCEPTION: ")
            print(e)
            print('Sending keepalive ping to {!r} in recv'.format(raddr))
            print("pinging!")
            await ws.ping()
            isActive = pingJanus(uid)
            if (isActive == False):
                return "None"
    return msg

async def disconnect(ws, peer_id):
    '''
    Remove @peer_id from the list of sessions and close our connection to it.
    This informs the peer that the session and all calls have ended, and it
    must reconnect.
    '''
    # Close connection
    if ws and ws.open:
        # Don't care about errors
        asyncio.ensure_future(ws.close(reason='hangup'))

def send_msg(uid, msg):
	global peers, sessions, rooms
	print("trying to send message to peer")
	if uid in peers:
		try:
			ws, raddr, status = peers[uid]
		except:
			print("couldn't send message")
		if ws and ws.open:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			future = asyncio.ensure_future(ws.send(json.dumps(msg)))
			loop.run_until_complete(future)
			#print("sent?")
		else:
			print("ws not open or not valid")
	else:
		print("uid not found in peers")

async def remove_peer(uid):
    global peers, sessions, rooms
    await cleanup_session(uid)
    if uid in peers:
        ws, raddr, status = peers[uid]
        if status and status != 'session':
            await cleanup_room(uid, status)
        del peers[uid]
        await ws.close()
        print("Disconnected from peer {!r} at {!r}".format(uid, raddr))

async def connection_handler(ws, uid):
	global peers, sessions, rooms
	raddr = ws.remote_address
	states[uid] = None
	peers[uid] = [ws, raddr, False]
	print("Registered peer {!r} at {!r}".format(uid, raddr))
	while True:
        # Receive command, wait forever if necessary
		msg = await recv_msg_ping(ws, raddr, uid)
		if msg == "None":
			print("No longer valid.")
			break
		#print(msg)
		try:
			data = json.loads(msg)
		except:
			raise Exception("Not JSON from {!r}".format(raddr))
			continue
		if "jsep" in data:  ## INTIAL RTC SETUP BY REMOTE CLIENT
			if "sdp" in data['jsep']:
				sdps[uid] = data['jsep']['sdp']
				configureStream(uid)
				pushICE(uid, None)
		elif "candidate" in data:
			print("Push real ICE")
			pushICE(uid, data)
		elif "ice" in data:
			print("Push real ICE (gst version)")
			data['candidate']=data['ice']
			print(data['candidate'])
			del(data['ice'])
			pushICE(uid,data)
		 

async def handler(ws, path):
    '''
    All incoming messages are handled here. @path is unused.
    '''
    raddr = ws.remote_address
    print("Connected to {!r}".format(raddr))
    peer_id, trans_id  = await hello_peer(ws)
    try:
        await connection_handler(ws, peer_id)
    except websockets.ConnectionClosed:
        print("Connection to peer {!r} closed, exiting handler".format(raddr))
    finally:
        await remove_peer(peer_id)

def sendToJanus(msg):
	global queue
	queue.append(msg)
	#pushQueue()

def pushQueue():
	global queue, sock
	while True:
		while len(queue):
			print("Sending Message to Janus")
			msg = queue.pop(0)
			msg = json.dumps(msg).encode("utf-8")
			try:
				#print(msg)
				sock.sendall(msg)
			except Exception as e:
				print(e)
				print("expeption with trying to send messge")
				sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
				sock.connect(server_address)
				try:
					print("resending")
					sock.sendall(msg)
					
				except:
					print("can't send message")
		time.sleep(0.01)

def pingJanus(uid):
        print("pinging Janus...")
        msg = {}
        msg['janus'] = "keepalive"
        try:
               msg['session_id'] = sessions
        except:
               print("No session to keepalive.")
               return False
        tid = str(uuid.uuid4().hex)[:12]
        msg['transaction'] = tid
        trans[tid] = uid, "ping"
        sendToJanus(msg)
        return True
#{
#   "janus": "success",
#   "transaction": "b5850384a8aa",
#   "data": {
#      "id": 480125329109731
#   }
#}
def matchUID(data):
	if "transaction" in data:
		if data['transaction'] in trans:
			return trans[data['transaction']]
	return False, False

def process(data):
	global sessions, plugins
#	try:
#		print("decoding Json")
#		data = json.loads(jdata)
#	except:
#		print("failed to load json")
#		return False
	if "janus" in data:
		if "success" in data["janus"]:
			uid, act  = matchUID(data)
			print("UID matched", uid, act)
			if act == "sid":
				sid = getID(data)
				if sid:
					sessions = sid
					#attachPlugin()
			elif act == "pid":
				pid = getID(data)
				if pid:
					plugins[uid] = pid
					piduid[pid] = uid
					print("Plugin id:",pid)
					createRoom(uid)
			elif act == "create":
				print("Room created")
				joinRoom(uid)
		elif "ack" in data["janus"]:
			print("ack")
		elif "event" in data["janus"]:
			uid, act  = matchUID(data)
			if uid == False:
				print("Unrequested Event")
				return	
			print("UID matched", uid, act)
			if act == "offer":
				print("SDP Answer Ready")
				answerSDP(uid, data)
				states[uid] = "ready" ## I suppose this should be already set to ready already, but this is the "real" ready
				pushICE(uid, None)
			elif act == "join":
				print("joined room")
				try:
					pubids[uid] = data['plugindata']['data']['id']
				except:
					print("publisher ID not found. error")
					return
				states[uid] = "ready" ## we can start pushing ICE the moment we join the room -- albeit this perhaps seems aggressive.
				configureStream(uid)
		elif "server_info" in data['janus']:
			print(" **************************** server info obtained")
		elif "webrtcup" in data["janus"]:
			print("WebRTC Good")
			msg = {}
			msg["messasge"] = "WebRTC connection is good"
			msg["request"] = "nothing"
			#return ## FIX ME
			uid = getUIDfromPID(data)
			send_msg(uid, msg)
		elif "webrtcdown" in data["janus"]:
			print("WebRTC bad")
			msg = {}
			msg["messasge"] = "WebRTC connection is bad"
			msg["request"] = "improve"
			#return ## FIZX ME
			uid = getUIDfromPID(data)
			send_msg(uid, msg)
		elif "hangup" in data["janus"]:
			print("WebRTC Disconnected")
			msg = {}
			msg["messasge"] = "WebRTC disconnected"
			msg["request"] = "reconnect"
			#return ## FIX ME
			uid = getUIDfromPID(data)
			send_msg(uid, msg)
			states[uid] = "ended"

			if uid in rtsp:        
				connSocket = rtsp[uid]
				connSocket.shutdown(socket.SHUT_RDWR)
				connSocket.close()
				print("closing")
			else:
				print("can't close")
		else:
			print("unknown response from janus")
	else:
		print("not a reply")

def getID(data):
	if "data" in data:
		if "id" in data['data']:
			return data['data']['id']
	return False
#{
#        "janus" : "attach",
#        "session_id" : <the session identifier>,               
#        "plugin" : "<the plugin's unique package name>",
#        "transaction" : "<random string>"
#}
def attachPlugin(uid):
	global sessions
	print("attaching plugin...")
	msg = {}
	msg['janus'] = "attach"
	msg['session_id'] = sessions
	msg['plugin'] = "janus.plugin.videoroom"
	tid = str(uuid.uuid4().hex)[:12]
	msg['transaction'] = tid
	trans[tid] = uid, "pid"
	sendToJanus(msg)

def answerSDP(uid,data):
	print("send sdp answer")
	#print(data)
	msg = {}
	try:
		msg["messasge"] = "process sdp answer and ensure ICE is being pushed"
		msg["audio_codec"] = data["plugindata"]["data"]["audio_codec"]
		msg["video_codec"] = data["plugindata"]["data"]["video_codec"]
		codec[uid] = data["plugindata"]["data"]["video_codec"]  ## update reality, despite what might of been provided by client
		msg["jsep"] = data["jsep"]
		msg["request"] = "publish"
	except:
		print("data missing jsep or something")
		return
	send_msg(uid, msg)
	

def createRoom(uid):
	global sessions, plugins
	print("create room")
	msg = {}
	msg['janus'] = "message"
	msg['session_id'] = sessions
	msg['handle_id'] = plugins[uid]
	body = {}
	body['request'] = "create"
	rooms[uid] = random.randint(1,10000)
	body['room'] = rooms[uid]
	body['publishers'] = 1
	if uid in codec:
		body['videocodec'] = codec[uid]	
	else:
		body['videocodec'] = "h264,vp8"
	body['fir_freq'] = 2
	body['admin_key'] = "supersecret"
	msg['body'] = body
	tid = str(uuid.uuid4().hex)[:12]
	msg['transaction'] = tid
	trans[tid] = uid, "create"
	sendToJanus(msg)

#{"janus":"trickle","candidate":{"completed":true},"transaction":"ABgvd2ruuIbd","session_id":6481667507767800,"handle_id":1812202103772506} 

def pushICE(uid, data):
        print("PUSHING ICE")
        global ices, sessions, plugins, trans, states
        if data != None:
                if (states[uid]!="ready"):
                        print("Not ready to forward candidates")
                        if uid in ices:
                                ices[uid].append(data['candidate'])
                        else:
                                ices[uid] = [data['candidate']]
                        return

                if uid in ices:
                        ices[uid].append(data['candidate'])
                else:
                        ices[uid] = [data['candidate']]
        elif uid not in ices:
                print("ignore ice attempt")
                return
		
        print("forwarding ICE candidate")
        msg = {}
        msg['janus'] = "trickle"
        msg['session_id'] = sessions
        msg['handle_id'] = plugins[uid]
        msg['candidates'] = ices[uid]
        del ices[uid]
        tid = str(uuid.uuid4().hex)[:12]
        msg['transaction'] = tid
        trans[tid] = uid, "ice"
        sendToJanus(msg)


# {"janus":"message","body":{"request":"configure","audio":true,"video":true},"transaction":"qJ32ZcgNmmzZ","jsep":{"type":"offer","sdp":"v=0\r\n"},"session_id":6242440443971610,"handle_id":5865380017130524}

def configureStream(uid):
        global sessions, plugins
        print("configure stream")

        if states[uid] != "ready":
                print("Not yet ready to configure stream")
                return False
        try:
                sdp = sdps[uid]
        except:
                print("no sdp provided yet")
                return

        msg = {}
        msg['janus'] = "message"
        msg['session_id'] = sessions
        msg['handle_id'] = plugins[uid]
        body = {}
        body['request'] = "configure"
        body['video'] = True
        body['audio'] = True
        msg["body"] = body
        jsep = {}
        jsep['type'] = "offer"
        jsep['sdp'] = sdp
        msg['jsep'] = jsep
        tid = str(uuid.uuid4().hex)[:12]
        msg['transaction'] = tid
        trans[tid] = uid, "offer"
        sendToJanus(msg)

def getUIDfromPID(data):
	if "sender" in data:
		if data["sender"] in piduid:
			uid = piduid[data["sender"]]
			return uid
	return False

# {"janus":"message","body":{"request":"join","room":1234,"ptype":"publisher","display":"asdfasdf"},"transaction":"nJlckz0sucs8","session_id":6242440443971610,"handle_id":5865380017130524}
def joinRoom(uid):  ### I SHOULD CONVERT THIS TO JOINANDCONFIGURE. Probably shouldn't create the room until I get the SDP and settings too.
	global plugins, sessions
	print("join room")
	msg = {}
	msg['janus'] = "message"
	msg['session_id'] = sessions
	msg['handle_id'] = plugins[uid]
	body = {}
	body['request'] = "join"
	body['room'] = rooms[uid]
	body['ptype'] = "publisher"
	body['display'] = "anon"
	msg["body"] = body
	tid = str(uuid.uuid4().hex)[:12]
	msg['transaction'] = tid
	trans[tid] = uid, "join"
	sendToJanus(msg)


def rtp_forward(uid, remote_ip, video_port, audio_port):
	global plugins, sessions
	print("forward video")
	msg = {}
	msg["janus"] = "message"
	body = {}
	body["request"] = "rtp_forward"
	if uid not in rooms:
		print("Video Room for RTP forward not setup yet")
		return
	body["room"] = rooms[uid]
	body["publisher_id"] = pubids[uid]
	body["host"] = remote_ip
	body["video_port"] = video_port
	body["audio_port"] = audio_port
	body["video_pt"] = 96
	body["audio_pt"] = 111
	msg["body"] = body
	
	msg['session_id'] = sessions
	msg['handle_id'] = plugins[uid]

	tid = str(uuid.uuid4().hex)[:12]
	msg['transaction'] = tid
	trans[tid] = uid, "rtp"
	sendToJanus(msg)
	

def rtspServerMain(rtspSocketMain):
	time.sleep(1)

	print("RTSP SERVER STARTED")
	while True:
		print("waiting for RTSP connection")
		clientInfo = {}
		clientInfo['rtspSocket'] = rtspSocketMain.accept()   # this accept {SockID,tuple object},tuple object = {clinet_addr,intNum}!!
		print("Socket connection made")
		ServerWorker(clientInfo).run()


class ServerWorker:
	INIT = 0
	READY = 1
	PLAYING = 2
	RECORDING = 3
	state = INIT

	OK_200 = 0
	FILE_NOT_FOUND_404 = 1
	CON_ERR_500 = 2

	clientInfo = {}

	def __init__(self, clientInfo):
		self.clientInfo = clientInfo

	def run(self):
		threading.Thread(target=self.recvRtspRequest).start()

	def recvRtspRequest(self):
		"""Receive RTSP request from the client."""
		connSocket = self.clientInfo['rtspSocket'][0]
		stateCon = True
		while stateCon:
			try:
				data = connSocket.recv(4096).decode('utf-8')  ###
			except:
				break
			if data:
				#print('-'*60 + "\nData received:\n" + '-'*60)
				#print(data)
				stateCon = self.processRtspRequest(data)
			else:
				print("IDELEEEEEEEEEEEEEEEEEE!!")
				time.sleep(1);

	def processRtspRequest(self, data):  ## RTSP services don't need to redirect -- most importantly, we don't know where to redirect to since no IP is provided
		"""Process RTSP request sent from the client."""
		# Get the request type
		try:
			request = data.split('\r\n')
			line1 = request[0].split(' ')
			requestType = line1[0]
			if (request[0] == "RTSP/1.0 200 OK"):
				print("Connection likely Terminated")
				return False
			# Get the media file name
			filename = line1[1]
			# Get the RTSP sequence number
			seq = data.split('CSeq: ')[1]
			seq = str(seq.split('\n')[0])
			self.clientInfo['stream_token'] = filename.split("/")[3]
		except:
			try:
				data = "SETUP rtsp" + data.split("SETUP rtsp")[1]
				request = data.split('\n')
				line1 = request[0].split(' ')
				requestType = line1[0]
				filename = line1[1]
				seq = data.split('CSeq: ')[1]
				seq = str(seq.split('\n')[0])
				self.clientInfo['stream_token'] = filename.split("/")[3]
			except:
				print("I Don't understand the request")
				raise Exception("Not JSON from {!r}".format(raddr))
				return False;
		# Process SETUP request

		rtsp[self.clientInfo['stream_token']] = self.clientInfo['rtspSocket'][0]

		if self.clientInfo['stream_token'] not in states:
			print("This Token is not valid.")
			return False
		elif states[self.clientInfo['stream_token']] == "ended":
			print("This token has expired")
			return False

		if requestType == "SETUP":
			if self.state == self.INIT: ## VIDEO
				# Update state
				print('-'*60 + "\nSETUP 1 Request Received\n")
				self.state = self.READY
				try:	
					self.clientInfo['rtpPort'] = data.split('client_port=')[1].split("-")[0]
				except:
					print("No port provided")
					if 'rtpPort' not in self.clientInfo:
						return True
					
				reply = 'RTSP/1.0 200 OK\r\nCSeq: ' + seq + '\nSession: ' + str(self.clientInfo['session']) +'\r\nTransport: RTP/AVP/UDP;unicast;client_port='+str(self.clientInfo['rtpPort'])+';\r\n\r\n'
				print(reply)	
				try: ##
					self.clientInfo['stream_token'] = filename.split("/")[3]
					print("STREAM TOKEN:",self.clientInfo['stream_token']) ## ALSO IS THE UID
				except: ##
					print("*******!!!!*****")
					print("stream token does not exist or other error occured")
					return False
	
				connSocket = self.clientInfo['rtspSocket'][0]
				connSocket.send(reply.encode("utf-8"))

				print("sequenceNum is " + seq)
				# Get the RTP/UDP port from the last line
				print('-'*60 + "\nrtpPort is: " + str(self.clientInfo['rtpPort']) + "\n" + '-'*60)
				print("filename is " + filename + " ... " + filename.split("/")[3])
						
			else: ## AUDIO
				# Update state
				print('-'*60 + "\nSETUP 2 Request Received\n")
				try:
					self.clientInfo['rtpPort_audio'] = data.split('client_port=')[1].split("-")[0]
				except:
					print("No port provided - audio")
					if 'rtpPort_audio' not in self.clientInfo:
						return True
				reply = 'RTSP/1.0 200 OK\r\nCSeq: ' + seq + '\nSession: ' + str(self.clientInfo['session']) +'\r\nTransport: RTP/AVP/UDP;unicast;client_port='+str(self.clientInfo['rtpPort_audio'])+';\r\n\r\n'
				print(reply)
				connSocket = self.clientInfo['rtspSocket'][0]
				connSocket.send(reply.encode("utf-8"))

				print("sequenceNum is " + seq)
				# Get the RTP/UDP port from the last line
				print('-'*60 + "\nrtpPort Audio is: " + str(self.clientInfo['rtpPort_audio']) + "\n" + '-'*60)
				print("filename or Token requested is " + filename + " ... " + filename.split("/")[3])
						
		# Process PLAY request
		elif requestType == "PLAY":
			print("PLAY STATE")
			if self.state == self.READY:
				print('-'*60 + "\nPLAY Request Received\n" + '-'*60)
				self.state = self.PLAYING

                                # Create a new thread and start sending RTP packets
				self.clientInfo['event'] = threading.Event()
				self.clientInfo['worker']= threading.Thread(target=self.sendRtp)
				self.clientInfo['worker'].start()

				self.replyRtsp(self.OK_200, seq)
				print('-'*60 + "\nSequence Number ("+ seq + ")\nReplied to client\n" + '-'*60)

		elif requestType == "TEARDOWN":
			print('-'*60 + "\nTEARDOWN Request Received\n" + '-'*60)
			if self.state == self.PLAYING:
				self.clientInfo['event'].set()
			self.replyRtsp(self.OK_200, seq)

		elif requestType == "OPTIONS":
			self.clientInfo['session'] = random.randint(100000, 999999)
			print('-'*60 + "\nOPTIONS Request Received\n" + '-'*60)
			reply = "RTSP/1.0 200 OK\r\nCSeq: " + seq + "\nPublic: OPTIONS, RECORD, ANNOUNCE, DESCRIBE, SETUP, PLAY\r\nServer: Python RTSP server\r\n\r\n"
			print(reply)
			connSocket = self.clientInfo['rtspSocket'][0]
			connSocket.send(reply.encode("utf-8"))
			print('-'*60 + "\nSequence Number ("+ seq + ")\nReplied to client\n" + '-'*60)

		elif requestType == "DESCRIBE":  ## VP8 + OPUS support only currently
			print('-'*60 + "\nDESCRIBE Request Received\n" + '-'*60)
			print(self.clientInfo)
			if codec[self.clientInfo['stream_token']] == "h264":
				sdp = "v=0\r\n\
o=- 0 0 IN IP4 127.0.0.1\r\n\
s=Ooblex H264 OPUS WebRTC To RTSP\r\n\
c=IN IP4 "+ str(self.clientInfo['rtspSocket'][1][0])+"\r\n\
t=0 0\r\n\
m=video 0 RTP/AVP 96\r\n\
a=rtpmap:96 H264/90000\r\n\
a=control:streamid=0\r\n\
a=fmtp: 96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f\r\n\
a=recvonly\r\n\
m=audio 0 RTP/AVP 111\r\n\
a=rtpmap:111 OPUS/48000/\r\n\
a=control:streamid=1\r\n\
a=recvonly\r\n"
			else:
				sdp = "v=0\r\n\
o=- 0 0 IN IP4 127.0.0.1\r\n\
s=Ooblex WebRTC To RTSP\r\n\
c=IN IP4 "+ str(self.clientInfo['rtspSocket'][1][0])+"\r\n\
t=0 0\r\n\
m=video 0 RTP/AVP 96\r\n\
a=rtpmap:96 VP8/90000\r\n\
a=control:streamid=0\r\n\
a=recvonly\r\n\
m=audio 0 RTP/AVP 111\r\n\
a=rtpmap:111 OPUS/48000/2\r\n\
a=control:streamid=1\r\n\
a=recvonly\r\n"
 
			reply = 'RTSP/1.0 200 OK\r\nCSeq: ' + seq + '\nContent-Type: application/sdp\r\nContent-Length: '+str(len(sdp))+'\r\n\r\n'
			reply = reply+sdp
			print(reply)
			connSocket = self.clientInfo['rtspSocket'][0]
			connSocket.send(reply.encode("utf-8"))
			print('-'*60 + "\nSequence Number ("+ seq + ")\nReplied to client\n" + '-'*60)

		elif requestType == "ANNOUNCE":
			print('-'*60 + "\nANNOUNCE Request Received\n" + '-'*60)
			reply = 'RTSP/1.0 200 OK\r\nCSeq: ' + seq+'\r\n\r\n'
			print(reply)
			connSocket = self.clientInfo['rtspSocket'][0]
			connSocket.send(reply.encode("utf-8"))
			print('-'*60 + "\nSequence Number ("+ seq + ")\nReplied to client\n" + '-'*60)

		elif requestType == "RECORD":
			self.state = self.RECORDING
			print('-'*60 + "\nRECORD Request Received\n" + '-'*60)
			self.replyRtsp(self.OK_200, seq)
			reply = 'RTSP/1.0 200 OK\r\nCSeq: ' + seq+'\r\nSession: '+ str(self.clientInfo["session"]) +'\r\n\r\n'
			print(reply)
			connSocket = self.clientInfo['rtspSocket'][0]
			connSocket.send(reply.encode("utf-8"))
			print('-'*60 + "\nSequence Number ("+ seq + ")\nReplied to client\n" + '-'*60)
			print("STREAM TOKEN: "+self.clientInfo['stream_token'])
		return True
			
	def sendRtp(self):
		"""Send RTP packets over UDP."""
		#port = int(self.clientInfo['rtpPort'])
		#subprocess.call('ffmpeg -re -i /home/ubuntu/rtsp/videoClip.mov -an -vcodec h264 -f rtp rtp://'+str(self.clientInfo['rtspSocket'][1][0])+':'+str(port), shell=True)
		video_port = int(self.clientInfo['rtpPort'])
		try:
			audio_port = int(self.clientInfo['rtpPort_audio'])
			print("audio port seems correct")
		except:
			print("defaulting AUdio port to guesstimate")
			audio_port = int(self.clientInfo['rtpPort'])+2

		ip_address = str(self.clientInfo['rtspSocket'][1][0])
		rtp_forward(self.clientInfo['stream_token'], ip_address, video_port, audio_port)
		return True

	def replyRtsp(self, code, seq):
		"""Send RTSP reply to the client."""
		if code == self.OK_200:
			#print "200 OK"
			reply = 'RTSP/1.0 200 OK\r\nCSeq: ' + seq + '\nSession: ' + str(self.clientInfo['session']) +'\r\n\r\n'
			print(reply)
			connSocket = self.clientInfo['rtspSocket'][0]
			connSocket.send(reply.encode("utf-8"))
		elif code == self.FILE_NOT_FOUND_404:
			print("404 NOT FOUND")
		elif code == self.CON_ERR_500:
			print("500 CONNECTION ERROR")


SERVERIP="127.0.0.1"
SERVER_PORT=554

rtspSocketMain = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#rtspSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
while True:
	try:
		rtspSocketMain.bind((SERVERIP, SERVER_PORT))
		rtspSocketMain.listen(5)
		print("connect")
		break
	except:
		print("Couldnt connect; retrying")
		time.sleep(3)
		continue
#rtspSocketMain.setblocking(False)


t = threading.Thread(target=rtspServerMain, args=(rtspSocketMain,))
t.start()

def consumeListener():
	global sock
	decoder = json.JSONDecoder(strict=False)
	buffer = b''
	while True:
		print("waiting for more data")
		data = sock.recv(65000)
		if data == b'':
			raise RuntimeError("socket connection broken")
			#sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
			try:
				print("reconnecting")
				sock.connect(server_address)
			except:
				print("can't reconnected")
		print('received "%s"' % data)
		data = data.decode("utf-8")
		while len(data):
			try:
				result, index = decoder.raw_decode(data)
				print("Enough data!")
				threading.Thread(target=process, args=(result,)).start()
				print(index)
				data = data[index:]
			except:
				print("Not enough to do Json")
				break


###
print('connecting to unix socket server..')


threading.Thread(target=consumeListener).start()
threading.Thread(target=pushQueue).start()


msg = {}
msg['janus'] = "info"
msg['transaction'] = str(uuid.uuid4().hex)[:12]
msg['correlation_id'] =  str(uuid.uuid1())
sendToJanus(msg)

msg = {}
msg['janus'] = "create"
tid = str(uuid.uuid4().hex)[:12]
msg['transaction'] = tid
trans[tid] = False, "sid"
sendToJanus(msg) ## create session





print("loading websockets")

####

chain_pem = "/etc/letsencrypt/live/"+config.DOMAIN_CONFIG['domain']+"/fullchain.pem'
key_pem = "/etc/letsencrypt/live/"+config.DOMAIN_CONFIG['domain']+"/privkey.pem'

sslctx = ssl.create_default_context()
try:
	sslctx.load_cert_chain(chain_pem, keyfile=key_pem)
except FileNotFoundError:
	print("SSL Certificates not found.")
	sys.exit(1)

sslctx.check_hostname = False
sslctx.verify_mode = ssl.CERT_NONE

wsd = websockets.serve(handler, '0.0.0.0', 8100, ssl=sslctx,
                       # Maximum number of messages that websockets will pop
                       # off the asyncio and OS buffers per connection. See:
                       # https://websockets.readthedocs.io/en/stable/api.html#websockets.protocol.WebSocketCommonProtocol
                       max_queue=16)

logger = logging.getLogger('websockets.server')

logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler())

asyncio.get_event_loop().run_until_complete(wsd)
asyncio.get_event_loop().run_forever()

