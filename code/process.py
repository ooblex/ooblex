from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.chdir("/root/ooblex/")
import re
import tensorflow as tf
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import numpy as np
import cv2
import binascii
import threading
from multiprocessing import Process
import time
import sys
import detect_face
import ssl
from SimpleWebSocketServer import WebSocket, SimpleSSLWebSocketServer
from amqpstorm import UriConnection
from amqpstorm import Message
import uuid
import logging
import datetime
import json
#logging.basicConfig(level=logging.DEBUG)

class NodeLookup(object):
  def __init__(self, label_lookup_path=None, uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = './inception/imagenet_2012_challenge_label_map_proto.pbtxt'
    if not uid_lookup_path:
      uid_lookup_path = './inception/imagenet_synset_to_human_label_map.txt'
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string
    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]
    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def create_graph():
  with tf.gfile.FastGFile("./inception/classify_image_graph_def.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image_data,sess_thing):
    print("inference running")
    predictions = sess_thing.run(softmax_tensor, {'DecodeJpeg:0': image_data})
    predictions = np.squeeze(predictions)
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-1:][::-1]  ## top 3 predictions
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
    return "found something"

# object detect model
create_graph()
#sess_thing = tf.Session()
#softmax_tensor = sess_thing.graph.get_tensor_by_name('softmax:0')

#run_inference_on_image(image_data) ###!!!!!!!

# face detect model
minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
factor = 0.709 # scale factor


########

clients = []
processes = []
tokens = {}

class SimpleChat(WebSocket):

    def handleMessage(self):
       if (self.data.startswith("start process:")):
          if (self.pState == True):
              self.sendMessage("Already started")
              return
          streamKey = self.data.split("start process:")
          if (len(streamKey)!=2):
              print("No stream key provided!")
              self.sendMessage("No streamKey provided")
              return
          self.sendMessage("Starting Video Processing")
          #idetector = threading.Thread(target=process_stream)
          self.pState = True
          self.streamKey = str(streamKey[1])

          tokens[streamKey[1]] = self

          print(streamKey[1])

          mainchannelin.queue.declare(streamKey[1])

          p = Process(target=process_stream, args=(streamKey[1],))
          #p = threading.Thread(target=process_stream)
          self.process = p
          p.start()
          print("Done starting")
       else:
          print(str(self.data))
          message = Message.create(mainchannelin, str(self.data))
          message.publish(self.streamKey)

 
    def handleConnected(self):
       print(self.address, 'Connected to API server')
       for client in clients:
          client.sendMessage(self.address[0] + u' - connected')

       self.pState = False
       clients.append(self)

    def handleClose(self):
       print("handle close")
       if (self.pState==True):
          self.process.terminate()
          print("Stopping process")
       clients.remove(self)
       print(self.address, 'closed')


def broadcast(data):
	print("Trying to send message")
	print(data)
	data = json.loads(data)
	if "key" in data:
		print(data["key"])
	else:
		print("No Key provided. error");
		return
	if data['key'] in tokens:
		try:
			tokens[data['key']].sendMessage(data['msg'])
		except:
			print("no client to broadcast to found !!!!!!!!")

GObject.threads_init()
Gst.init(None)
Gst.debug_set_active(True)
Gst.debug_set_default_threshold(1)

def message_handler( bus, message):
	struct = message.get_structure()
	if message.type == Gst.MessageType.EOS:
		## delete appsink from stream list
		print('Stream ended.')
		run=False




def process_stream(streamKey):
	print("connecting to rmq server")
	while True:
		try:
			connection = UriConnection("amqps://admin:QYGTGMPLVLFYOPRH@portal-ssl494-22.bmix-dal-yp-42bf8654-c98e-426b-b8e4-a9d19926bfde.steve-seguin-email.composedb.com:39186/bmix-dal-yp-42bf8654-c98e-426b-b8e4-a9d19926bfde")
			break
		except:
			continue
		
	channel = connection.channel() # start a channel

	
	#exchange = rabbitpy.DirectExchange(channel, 'main-exchange')
	#exchange.declare()

	#queue = rabbitpy.Queue(channel, 'broadcast-all')
	#queue.declare()
	
	#queue.bind(exchange, 'to-broadcast')

	def sendMessage(msg):
		data = {}
		data['msg'] = msg
		data['key'] = str(streamKey)
		msg = json.dumps(data)
		print("sending: ",msg)
		msg1 = Message.create(channel, msg)
		msg1.publish('broadcast-all')

        # start consuming (blocks
	#pchnl = threading.Thread(target=channel.start_consuming)
	#pchnl.start()

	global thingDetection, faceDetection, processSpeed, img
	thingDetection = False
	faceDetection = False
	img = None
	processSpeed = 1.0

	def checkMessages(message):
			global thingDetection, faceDetection, processSpeed, img
			print('starting to check messages')
			#queue2 = rabbitpy.Queue(channel, str(pid))
			#queue2.declare()
			#for message in rabbitpy.Queue(channel, str(pid)):
			print("message from websocket to process")
			message.ack()
			msg = message.body
			print(msg)
			if (msg=="ObjectOn"):
				thingDetection = True
			elif (msg=="ObjectOff"):
				thingDetection = False
			elif (msg=="FaceOn"):
				print("face detection turning on")
				faceDetection = True
			elif (msg=="FaceOff"):
				faceDetection = False
			elif (msg=="speedFast"):
				processSpeed = 0.0
			elif (msg=="speedSlow"):
				processSpeed = 1.0
			elif (msg=="saveImage"):
				if (type(img) == type(None)):
					print("can't take snapshot")
					sendMessage("not yet ready")
					return
				try:
					filename = uuid.uuid4().hex+".jpg"
					cv2.imwrite("/var/www/html/images/"+filename,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
					print("https://api.ooblex.com/images/"+filename)
					sendMessage("https://api.ooblex.com/images/"+filename)
				except:
					print("error!")
					sendMessage("error with request")
			else:
				print(msg)
				

	def consumeListener():
		print("process starting Listener")
		channel.queue.declare(streamKey)
		channel.basic.consume(checkMessages, str(streamKey), no_ack=False)
		channel.start_consuming()
		print("listening done")

	chkmsg = threading.Thread(target=consumeListener)
	chkmsg.start()

	print("process stream started")
	sendMessage('Trying to connect to video stream...')

	CLI = ' rtspsrc location=rtsp://127.0.0.1:554/'+str(streamKey)+' name=r ! application/x-rtp,media=video ! rtph264depay ! video/x-h264, stream-fromat=byte-stream, alignment=nal ! h264parse ! avdec_h264  ! videoconvert ! video/x-raw, format=RGB ! appsink max_buffers=1 drop=true sync=false name=as1 '

	pipline=Gst.parse_launch(CLI)

	message_bus = pipline.get_bus()
	message_bus.add_signal_watch()
	message_bus.connect('message', message_handler)
	
	appsink=pipline.get_by_name("as1") 
	appsink.set_property('emit-signals',False) 
	pipline.set_state(Gst.State.PLAYING)

	print("loading object detector")
	sess_thing = tf.Session()
	softmax_tensor = sess_thing.graph.get_tensor_by_name('softmax:0')

	print("loading face detector")
	sess_face = tf.Session()
	pnet, rnet, onet = detect_face.create_mtcnn(sess_face, None)
	
	print("loading smile detector")
	sess_smile = tf.Session()
	with tf.gfile.FastGFile("emotions.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')
	frame_window = 10
	emotion_offsets = (20, 40)
	smile_input_layer = sess_smile.graph.get_tensor_by_name('output_node0:0')
	

	print("starting loop")
	while True:
		sample = appsink.emit('pull-sample')
		try:
			buf=sample.get_buffer()
		except:
			print("Failed. exiting")
			return
		if (buf.get_size()<100):
			print("buffer empty")
			continue
		caps = sample.get_caps()

		format = caps.get_structure(0).get_value('format')
		height = caps.get_structure(0).get_value('height')
		width =  caps.get_structure(0).get_value('width')
		print("\n\nNew Frame: ",width,height, format)
		data=buf.extract_dup(0,buf.get_size())

		if (type(img) == type(None)):
			sendMessage("Video Successful: "+ str(width)+"x"+str(height))

		try:
			img = np.frombuffer(data, np.uint8).reshape(int(height),width,3)
		except:
			continue

		if (faceDetection==True):
			bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
			print("Faces detected: "+str(len(bounding_boxes)))
			sendMessage("Faces detected: "+str(len(bounding_boxes)))
			if (len(bounding_boxes)>0):
				if (bounding_boxes[0][3]-bounding_boxes[0][1]<10):
					print("not tall enough")
				if (bounding_boxes[0][2]-bounding_boxes[0][0]<10):
                                        print("not width enough")
				if (bounding_boxes[0][2]>len(img[:,0])):
					print("larger than img 1")
				if (bounding_boxes[0][3]>len(img[0,:])):
                                        print("larger than img 2")
				img = img[int(bounding_boxes[0][1]):int(bounding_boxes[0][3]+1),int(bounding_boxes[0][0]):int(bounding_boxes[0][2]+1),:]
				ss = img.shape
				if (ss[0]*ss[1])==0:
					print("Image dimensions are incorrect")
					return
				print(bounding_boxes[0][4], ss)
				face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (64, 64)).reshape((1,64,64,1)).astype('float32')
				face /=127.5
				face -= 1.0

				predictions = sess_smile.run(smile_input_layer, {'input_1:0': face})
				predictions = np.squeeze(predictions)
				emote = np.where(predictions==np.amax(predictions))[0]
				if (emote==3):
					print("happy score: "+str(predictions[3]))
					if (predictions[3]>0.5):
						sendMessage("Smile Detected!")
						filename = uuid.uuid4().hex+".jpg"
						cv2.imwrite("/var/www/html/images/"+filename,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
						print("https://api.ooblex.com/images/"+filename)
						sendMessage("https://api.ooblex.com/images/"+filename)

				#print(predictions)

		if (thingDetection==True):
			predictions = sess_thing.run(softmax_tensor, {'DecodeJpeg:0': img})
			predictions = np.squeeze(predictions)
			node_lookup = NodeLookup()
			top_k = predictions.argsort()[-1:][::-1]  ## top 3 predictions
			for node_id in top_k:
				human_string = node_lookup.id_to_string(node_id)
				score = predictions[node_id]
				print('%s (score = %.5f)' % (human_string, score))
				output = '%s (score = %.5f)' % (human_string, score)
				sendMessage("Detected: "+output)
		time.sleep(processSpeed)
	connection.close()

#process_stream()

while True:
	try:
		mainconnection = UriConnection("amqps://admin:QYGTGMPLVLFYOPRH@portal-ssl494-22.bmix-dal-yp-42bf8654-c98e-426b-b8e4-a9d19926bfde.steve-seguin-email.composedb.com:39186/bmix-dal-yp-42bf8654-c98e-426b-b8e4-a9d19926bfde")
		break
	except:
		continue
print('connecting..')
mainchannelout =  mainconnection.channel()
mainchannelin = mainconnection.channel()
print("connected")
#queue = Queue(mainchannel, 'broadcast-all')
#queue.declare()

def startMQ(message):# Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
	#for message in Queue(mainchannel,'broadcast-all'):
	print("message from process")
	message.ack()
	broadcast(message.body)

def startMQ1():
                print("starting to listen")
                mainchannelout.basic.consume(startMQ, 'broadcast-all', no_ack=False)
                mainchannelout.start_consuming()
                print("listening done")

mq = threading.Thread(target=startMQ1)
mq.start()

server = SimpleSSLWebSocketServer("", 8800, SimpleChat, "/etc/letsencrypt/live/api.ooblex.com/fullchain.pem", "/etc/letsencrypt/live/api.ooblex.com/privkey.pem", version=ssl.PROTOCOL_TLSv1)
server.serveforever()

	
