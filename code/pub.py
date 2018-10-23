import binascii
import threading
import time
import sys
import struct
from base64 import b64encode
from hashlib import sha1
import os

from http.server import BaseHTTPRequestHandler, HTTPServer

import redis
global r
redisUrl = "rediss://admin:GSURJCGDVFWYOLYD@portal1369-16.bmix-dal-yp-cbfabb84-69f1-472d-87d7-343fd70e44c6.steve-seguin-email.composedb.com:40299"
r = redis.Redis.from_url(redisUrl)
for key in r.keys('*'):
	print(key)
						

pubsub = r.pubsub()  
pubsub.psubscribe('*')

print('Starting message loop')  
while True:  
    message = pubsub.get_message()
    if message:
        print(message)
    else:
        time.sleep(0.01)	

