import binascii
import threading
import time
import ssl
import sys
import struct
from base64 import b64encode
from hashlib import sha1
import os
import json

from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer

import redis
global r, pixel
redisUrl = "rediss://admin:GSURJCGDVFWYOLYD@portal1369-16.bmix-dal-yp-cbfabb84-69f1-472d-87d7-343fd70e44c6.steve-seguin-email.composedb.com:40299"
r = redis.Redis.from_url(redisUrl)
for key in r.keys('*'):
	print(key)
				
path = '/root/ooblex/code/jpeg.jpg'
jpeg = open(path,'rb')
jpeg = jpeg.read()
print(jpeg)

# This class will handles any incoming request from the browser
class myHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		global r,jpeg
		print("SDFSDF")

		try:
			streamKey = self.path.split(".")[0].split("/")[1]
			videotype =  self.path.split(".")[1].split("?")[0]
			print("good", streamKey)
		except:
			print('bad')
			return

		if videotype == 'mjpg':
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			pubsub = r.pubsub()
			pubsub.subscribe(streamKey.encode("utf-8"))
			counter = 0
			lasttime = int(round(time.time() * 1000))

			lastimg = None
			lastcid = 0

			for item in pubsub.listen():
				try:
					if item['type'] == "subscribe":
						continue
					cid = int(item['data'].decode().split(":")[1])
					im = item['data']
					if cid < counter:
						continue
					elif cid<lastcid:
						im=r.get(im)
						if type(im) == type(None):
							continue
						self.wfile.write("--jpgboundary".encode('utf-8'))
						self.send_header('Content-type','image/jpeg')
						self.send_header('Content-length',str(len(im)))
						self.end_headers()
						self.wfile.write(im)
						counter = cid
					else:
						lastimg = r.get(lastimg)
						if type(lastimg)!=type(None):
							self.wfile.write("--jpgboundary".encode('utf-8'))
							self.send_header('Content-type','image/jpeg')
							self.send_header('Content-length',str(len(lastimg)))
							self.end_headers()
							self.wfile.write(lastimg)
							counter = lastcid
						lastimg = im
						lastcid = cid
				except KeyboardInterrupt:
					break
			return
		elif self.path.endswith('.html'):
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.wfile.write('<html><head></head><body>'.encode('utf-8'))
			self.wfile.write(('<img src="'+streamKey+'.mjpg" width=320 height=180/>').encode('utf-8'))
			self.wfile.write('</body></html>'.encode('utf-8'))
			return

chain_pem = '/etc/letsencrypt/live/api.ooblex.com/fullchain.pem'
key_pem = '/etc/letsencrypt/live/api.ooblex.com/privkey.pem'

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Handle requests in a separate thread."""
        pass

def remote_threader():
    remote_Server = ThreadedHTTPServer(("", 81), myHandler)
    remote_Server.socket = ssl.wrap_socket(remote_Server.socket, keyfile=key_pem, certfile=chain_pem, server_side=True)
    remote_Server.serve_forever()
    print("end")


remote_http_thread = threading.Thread(target=remote_threader)
remote_http_thread.start()
print("end2")
