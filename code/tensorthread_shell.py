from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.chdir("/root/ooblex/")
import re
import tensorflow as tf
import numpy as np
import cv2
import binascii
import threading
from multiprocessing import Process
import time
import sys
import detect_face
import ssl
from amqpstorm import UriConnection
from amqpstorm import Message
import uuid
import logging
import datetime
import json
import math
import urllib
import config 

## CONFIGURE THE REDIS SERVER AS NEEDED 

graph = tf.get_default_graph()
r = redis.Redis.from_url(config.REDIS_CONFIG['uri'])

### IMPORT ANY EXTENRAL RESOURCES HERE  

def tensorThread(something):

	def url_to_image(url):  ## GET IMAGE FROM VIDEO SERVER, if data is an image
		try:
			resp = urllib.request.urlopen(url)
		except:
			print("Invalid Image")
			return False
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image
    
	def sendMessage(msg,token):  ## THIS IS USED FOR COMMUNICATION ; needed to provide a response
		global mainchannel_out
		data = {}
		data['msg'] = msg
		data['key'] = token
		msg = json.dumps(data)
		#print("sending message:",msg)
		msg = Message.create(mainChannel_out, msg)
		msg.publish('broadcast-all')

	childChannel = mainConnection.channel()
	childChannel.queue.declare("tf-task", arguments={'x-message-ttl':1000})
	childChannel.basic.qos( 1 , global_ = True )

## PUT YOUR MACHINE LEARNING CODE OR WHATEVER YOU WANT PROCESSED INTO THE TENSORTHREAD PROCESSING FUNCTION BELOW
## Some Sample code for Image grabbing and processing are included

	def processTask(msg0): ### 
		childChannel.stop_consuming()
		msg = msg0.body
		msg = json.loads(msg)
		if (msg['task']=="SOMETASKNAMEHERE"):
			try:	
				image = np.asarray(bytearray(r.get(msg['redisID'])), dtype="uint8")
			except:
				return
			image = cv2.imdecode(image, cv2.IMREAD_COLOR)
			
			if type(image) == type(False):
				print("couldnt process 2")
				return
			faceDetection(image, msg['streamKey'], msg['redisID'], "trump")
      
		else:
			print("unknown task: ",msg['task'])
		msg0.ack()
		return
		
	def faceDetection(img, token, rid, dectype="trump"):  ## RESULT IS IMAGE BASED TRANSFORMATION
		global graph, r
		img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		## DO SOMETHING
		with graph.as_default():
			### DO SOMETHING
			jpg = cv2.imencode(".jpg", img).tostring()
		rid = rid+":face"
		r.set(rid, jpg, ex=30)
		r.publish(token+"face", rid)
		return
    
	def objectDetection(img, token):  ## AN EXAMPLE WHERE RESULT IS JUST TEXT
		predictions = sess_thing.run(softmax_tensor, {'DecodeJpeg:0': img})
		predictions = np.squeeze(predictions)
		return
		node_lookup = NodeLookup()
		top_k = predictions.argsort()[-1:][::-1]  ## top 3 predictions
		for node_id in top_k:
			human_string = str(node_lookup.node_lookup[str(node_id)])
			score = predictions[node_id]
			print('%s (score = %.5f)' % (human_string, score))
			output = '%s (score = %.5f)' % (human_string, score)
			sendMessage("Object Detected: "+output, token)




############### JUST TASK MANAGMENT --- NOT IMPORTANT BELOW THIS LINE

	while True:
		result = childChannel.basic.get("tf-task", no_ack=False)
		if not result:
			childChannel.basic.consume(processTask, "tf-task", no_ack=False)
			childChannel.start_consuming()
			continue
		processTask(result)

while True:
	try:
		mainConnection = UriConnection(config.REDDITMQ_CONFIG['uri'])
		break
	except:
		print("Couldn't connect to rabbitMQ server. Retrying")
		time.sleep(3)
		continue

global mainchannel_out
mainChannel_out = mainConnection.channel()
mainChannel_in = mainConnection.channel()

def processMessage(message):
	message.ack()
	#print(message.body)
	tensorThread(message.body) ## 

## NUMBER OF THREADS TO RUN AT A TIME.  Depends on the number of CORES your server has, but configured for a larger CPU Server

#p = Process(target=tensorThread, args=(None,))
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()
p = threading.Thread(target=tensorThread, args=(None,))
p.start()


print("TensorFlow Server started")
mainChannel_in.queue.declare("tf-controller", arguments={'x-message-ttl':3600000}) ### Spin up more threads if needed; this should be un-used I'd suspect
mainChannel_in.basic.consume(processMessage, 'tf-controller', no_ack=False)
mainChannel_in.basic.qos( 1 , global_ = True )
mainChannel_in.start_consuming()
