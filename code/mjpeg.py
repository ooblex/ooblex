import binascii
import json
import os
import ssl
import struct
import sys
import threading
import time
from base64 import b64encode
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import config
import redis

global r, pixel
r = redis.Redis.from_url(config.REDIS_CONFIG["uri"])
for key in r.keys("*"):
    print(key)

path = "/root/ooblex/assets/jpeg.jpg"
jpeg = open(path, "rb")
jpeg = jpeg.read()
print(jpeg)


# This class will handles any incoming request from the browser
class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global r, jpeg
        print("SDFSDF")

        try:
            streamKey = self.path.split(".")[0].split("/")[1]
            videotype = self.path.split(".")[1].split("?")[0]
            print("good", streamKey)
        except:
            print("bad")
            return

        if videotype == "mjpg":
            self.send_response(200)
            self.send_header(
                "Content-type", "multipart/x-mixed-replace; boundary=--jpgboundary"
            )
            self.end_headers()
            pubsub = r.pubsub()
            pubsub.subscribe(streamKey.encode("utf-8"))
            counter = 0
            lasttime = int(round(time.time() * 1000))

            lastimg = None
            lastcid = 0

            for item in pubsub.listen():
                try:
                    if item["type"] == "subscribe":
                        continue
                    cid = int(item["data"].decode().split(":")[1])
                    im = item["data"]
                    if cid < counter:
                        continue
                    elif cid < lastcid:
                        im = r.get(im)
                        if type(im) == type(None):
                            continue
                        self.wfile.write("--jpgboundary".encode("utf-8"))
                        self.send_header("Content-type", "image/jpeg")
                        self.send_header("Content-length", str(len(im)))
                        self.end_headers()
                        self.wfile.write(im)
                        counter = cid
                    else:
                        lastimg = r.get(lastimg)
                        if type(lastimg) != type(None):
                            self.wfile.write("--jpgboundary".encode("utf-8"))
                            self.send_header("Content-type", "image/jpeg")
                            self.send_header("Content-length", str(len(lastimg)))
                            self.end_headers()
                            self.wfile.write(lastimg)
                            counter = lastcid
                        lastimg = im
                        lastcid = cid
                except KeyboardInterrupt:
                    break
            return
        elif self.path.endswith(".html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write("<html><head></head><body>".encode("utf-8"))
            self.wfile.write(
                ('<img src="' + streamKey + '.mjpg" width=320 height=180/>').encode(
                    "utf-8"
                )
            )
            self.wfile.write("</body></html>".encode("utf-8"))
            return


chain_pem = "/etc/letsencrypt/live/" + config.DOMAIN_CONFIG["domain"] + "/fullchain.pem"
key_pem = "/etc/letsencrypt/live/" + config.DOMAIN_CONFIG["domain"] + "/privkey.pem"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    pass


def remote_threader():
    remote_Server = ThreadedHTTPServer(("", 81), myHandler)
    remote_Server.socket = ssl.wrap_socket(
        remote_Server.socket, keyfile=key_pem, certfile=chain_pem, server_side=True
    )
    remote_Server.serve_forever()
    print("end")


remote_http_thread = threading.Thread(target=remote_threader)
remote_http_thread.start()
print("end2")
