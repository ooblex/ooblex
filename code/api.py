from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import binascii
import threading
from multiprocessing import Process
import time
import sys
import ssl
from SimpleWebSocketServer import WebSocket, SimpleSSLWebSocketServer
from amqpstorm import UriConnection
from amqpstorm import Message
import uuid
import logging
import datetime
import json
import config
#logging.basicConfig(level=logging.DEBUG)

global tokens, clients, keys
clients = []
tokens = {}
keys = {}

class SimpleChat(WebSocket):

    def handleMessage(self):
       global tokens, keys, clients
       print("INCOMING **************** ")
       if (self.data.startswith("start process:")):
          streamKey = self.data.split("start process:")
          if (len(streamKey)!=2):
              self.sendMessage("No streamKey provided")
              return
          else:
              streamKey = streamKey[1]
          self.sendMessage("Starting Video Processing")
          self.state = "active"
          tokens[streamKey] = self
          self.sendMessage("...")
          message = Message.create(mainChannel_out, streamKey)
          message.publish("gst-launcher")
          self.sendMessage("!")
       else:
             data = self.data.split(":")
             if (len(data)!=2):
                 self.sendMessage("No streamKey provided")
                 return
             streamKey = data[1]
             task = data[0]
             print("trying to create Task")
             try:
                msg = {}
                msg['streamKey'] = streamKey
                msg['task'] = task
                msg = json.dumps(msg)
                message = Message.create(mainChannel_out, msg) ## channel, queue-name
                message.publish(streamKey)
             except Exception as e:
                self.sendMessage("Error: "+str(e))
 
    def handleConnected(self):
       global tokens, keys, clients
       print(self.address, 'Connected to API server')
       clients.append(self)

    def handleClose(self):
       global tokens, keys, clients
       print("Client Connection Closed")
       clients.remove(self)
       print(self.address, 'closed')


def sendClient(data):
	global tokens, keys, clients
	data = json.loads(data)
	print("sending message:",data)
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

while True:
	try:
		mainConnection = UriConnection(config.RABBITMQ_CONFIG['uri'])
		break
	except:
		continue

mainChannel_out =  mainConnection.channel()
mainChannel_in = mainConnection.channel()

def processMessage(message):
	print("message from process")
	message.ack()
	try:
		sendClient(message.body)
	except:
		print("error with sendClient")

def checkMessages():
	mainChannel_in.basic.consume(processMessage, 'broadcast-all', no_ack=False)
	mainChannel_in.start_consuming()

mq = threading.Thread(target=checkMessages)
mq.start()

print("Main API server starting")
server = SimpleSSLWebSocketServer("", 8800, SimpleChat, "/etc/letsencrypt/live/"+config.DOMAIN_CONFIG['domain']+"/fullchain.pem", "/etc/letsencrypt/live/"+config.DOMAIN_CONFIG['domain']+"/privkey.pem", version=ssl.PROTOCOL_TLS)
server.serveforever()
print("API.py STOPPED!!!!!!!!")
