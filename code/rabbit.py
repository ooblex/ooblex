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
from amqpstorm import UriConnection
from amqpstorm import Message
import uuid
import logging
import datetime
import json
import math
import urllib
import redis
import random

			

def sendMessage(msg,token):
	global mainchannel_out
	data = {}
	data['msg'] = msg
	data['key'] = token
	msg = json.dumps(data)
	#print("sending message:",msg)
	msg = Message.create(mainChannel_out, msg)
	msg.publish('broadcast-all')


def processTask(msg):
	msg.stop_consuming()
	print(msg.body)
	time.sleep(random.randint(0,10)/10.0)
	msg.ack()
	

def tensorThread(something):
	childChannel = mainConnection.channel()
	childChannel.queue.declare("tf-task", arguments={'x-message-ttl':1000})
	childChannel.basic.qos( 1 , global_ = True )
	def processTask(msg):
		childChannel.stop_consuming()
		print(msg.body)
		time.sleep(random.randint(0,10)/10.0)
		msg.ack()
	while True:
		result = childChannel.basic.get("tf-task", no_ack=False)
		if not result:
			print("Channel Empty.")
        		# We are done, lets break the loop and stop the application.
			childChannel.basic.consume(processTask, "tf-task", no_ack=True)
			childChannel.start_consuming()
			continue
		print(result.body)
		time.sleep(random.randint(0,10)/10.0)
		result.ack()
#	childChannel.basic.qos( 1 , global_ = True )
#	childChannel.start_consuming()

mainConnection = UriConnection("amqps://admin:QYGTGMPLVLFYOPRH@portal-ssl494-22.bmix-dal-yp-42bf8654-c98e-426b-b8e4-a9d19926bfde.steve-seguin-email.composedb.com:39186/bmix-dal-yp-42bf8654-c98e-426b-b8e4-a9d19926bfde")

global mainchannel_out
mainChannel_out = mainConnection.channel()
mainChannel_in = mainConnection.channel()

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



