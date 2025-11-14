from __future__ import absolute_import, division, print_function

import os

os.chdir("/root/ooblex/")
import re

import gi

gi.require_version("Gst", "1.0")
import binascii
import datetime
import json
import logging
import ssl
import sys
import threading
import time
import uuid
from multiprocessing import Process

import config
import cv2
import numpy as np
import redis
from amqpstorm import Message, UriConnection
from gi.repository import GObject, Gst

r = redis.Redis.from_url(config.REDIS_CONFIG["uri"])
# logging.basicConfig(level=logging.DEBUG)

GObject.threads_init()
Gst.init(None)
# Gst.debug_set_active(True)
# Gst.debug_set_default_threshold(2)


def gstreamer_process(body):
    print("gst process starting")
    childChannel_in = mainConnection.channel()  # start a channel
    childChannel_out = mainConnection.channel()  # start a channel
    streamKey = str(body)
    processSpeed = 1.0
    global tasks, connected
    tasks = []
    connected = True

    def message_handler(bus, message):
        print("MEssage handler fired")
        struct = message.get_structure()
        t = message.type
        ## TODO: SEND MESSAGE BACK TO MAIN API TO LET IT KNOW THE STREAM HAS ENDED; CLEAN UP?
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print("ERROR:", message.src.get_name(), " ", err.message)
            if dbg:
                print("debugging info:", dbg)
        elif t == Gst.MessageType.EOS:
            print("End-Of-Stream reached")
        else:
            # this should not happen. we only asked for ERROR and EOS
            print("ERROR: Unexpected message received.")
        connected = False

    def sendMessage(msg):
        data = {}
        data["msg"] = msg
        data["key"] = str(streamKey)
        msg = json.dumps(data)
        # print("sending: ",msg)
        msg = Message.create(mainChannel_out, msg)
        msg.publish("broadcast-all")

    def processMessage(msg):
        global tasks
        msg.ack()
        msg = msg.body
        msg = json.loads(msg)
        # print("Gst Process: ", msg)
        tasks = []  ## this is temporary
        if "task" in msg:
            tasks.append(msg["task"])
        else:
            print("I do not undersatnd this request")

    def on_new_buffer(appsink1):
        global tasks, connected
        counter = 0
        while connected == True:
            try:
                sample = appsink1.emit("pull-sample")
                buf = sample.get_buffer()
                if buf.get_size() < 100:
                    print("buffer empty")
                    continue

                # delta = not (buf.mini_object.flags & Gst.BufferFlags.DELTA_UNIT)
                # print(delta)
                caps = sample.get_caps()

                format = caps.get_structure(0).get_value("format")
                height = caps.get_structure(0).get_value("height")
                width = caps.get_structure(0).get_value("width")
                # print("\n\nNew Frame: ",width,height, format)
                sendMessage("Video Successful: " + str(width) + "x" + str(height))

                img = buf.extract_dup(0, buf.get_size())
                # img = np.frombuffer(data1, np.uint8).reshape(int(height),width,3)
                counter += 1
                rid = streamKey + ":" + str(counter)
                r.set(rid, img, ex=30)
                r.publish(streamKey, rid)

                if len(tasks) == 0:
                    # sendMessage("No tasks")
                    # time.sleep(0.2)
                    # appsink.set_property('emit-signals',True)
                    continue

                # filename = uuid.uuid4().hex+".jpg"
                # newFile = open("/var/www/html/images/"+filename, "wb")
                # newFile.write(img)
                # newFile.close()
                # cv2.imwrite("/var/www/html/images/"+filename, img)
                ## x-max-length
                for task in tasks:
                    task_msg = {}
                    task_msg["streamKey"] = streamKey
                    task_msg["timestamp"] = time.time()
                    task_msg["redisID"] = rid
                    task_msg["task"] = task
                    task_msg = json.dumps(task_msg)
                    task_msg = Message.create(mainChannel_out, task_msg)
                    task_msg.publish("tf-task")
                # time.sleep(0.1)
            except:
                print("ENDED")
                connected = False
                return False

    print("Gstreamer Loop Starting")
    sendMessage("Trying to connect to video stream...")

    CLI = (
        " rtspsrc location=rtsp://127.0.0.1:554/"
        + str(streamKey)
        + " name=r  buffer-mode=3 retry=1 do-rtcp=FALSE udp-reconnect=false ! application/x-rtp,media=video ! rtph264depay ! video/x-h264, stream-format=byte-stream, alignment=nal ! h264parse ! avdec_h264 output-corrupt=false ! queue max-size-buffers=0 max-size-bytes=0 max-size-time=0 ! videoconvert ! video/x-raw, format=I420 ! jpegenc ! queue leaky=1 ! appsink drop=TRUE max-buffers=1 sync=FALSE name=as1 "
    )
    print("gst-launch-1.0" + CLI)

    pipline = Gst.parse_launch(CLI)

    message_bus = pipline.get_bus()
    message_bus.add_signal_watch()
    message_bus.enable_sync_message_emission()
    message_bus.connect("message", message_handler)

    def probe(pad, info):
        if self._semaphore.acquire(blocking=False):
            return Gst.PadProbeReturn.PASS
        else:
            return Gst.PadProbeReturn.DROP

    # 	selector = pipline.get_by_name("probepad")
    # 	probe_pad = selector.get_static_pad("src")
    # 	probe = probe_pad.add_probe(Gst.PadProbeType.BUFFER, gst_buffer_is_keyframe, None)

    # 	selector2 = pipline.get_by_name("probepad2")
    # 	probe_pad2 = selector2.get_static_pad("src")
    # 	probe_pad.link(probe_pad2)

    appsink = pipline.get_by_name("as1")
    appsink.set_property("emit-signals", False)

    # 	appsink2=pipline.get_by_name("as2")
    # 	probe_pad = appsink2.get_static_pad("sink")
    # 	pipline.get_by_name("as2").srcpads[0].add_probe(Gst.PadProbeType.BUFFER | Gst.PadProbeType.BLOCK, probe)
    # 	probe_pad.add_probe(Gst.PadProbeType.BUFFER, gst_buffer_is_keyframe, None)
    # 	appsink.set_property('drop',True)
    # 	appsink.set_property('max-buffers',1)
    # 	appsink.set_property('sync',False)
    # appsink.connect('new-sample', on_new_buffer)
    pipline.set_state(Gst.State.PLAYING)

    t1 = threading.Thread(target=on_new_buffer, args=(appsink,))
    t1.setDaemon(True)
    t1.start()

    childChannel_in.queue.declare(streamKey, arguments={"x-message-ttl": 10000})
    childChannel_in.basic.consume(processMessage, str(streamKey), no_ack=False)
    childChannel_in.start_consuming()  ## STOP THIS ON END OF GSTREAMER
    print("gst end")
    connected = False


while True:
    try:
        mainConnection = UriConnection(config.RABBITMQ_CONFIG["uri"])
        break
    except:
        print("Unable to connect to RabbitMQ. Retrying..")
        time.sleep(3)
        continue

mainChannel_out = mainConnection.channel()
mainChannel_in = mainConnection.channel()


def gstLauncher(message):  #
    print("gstLauncher: ", message.body)
    message.ack()
    ## if message.body == start new stream, then do so.
    chkmsg = threading.Thread(target=gstreamer_process, args=(message.body,))
    chkmsg.start()


print("clearing image folder")
try:
    os.system("rm /var/www/html/images -r")
except:
    pass
os.mkdir("/var/www/html/images")

print("Starting to Listen: Main GST  thread")
mainChannel_in.queue.declare("gst-launcher", arguments={"x-message-ttl": 30000})
mainChannel_in.basic.consume(gstLauncher, "gst-launcher", no_ack=False)
mainChannel_in.start_consuming()

print("DECODER.PY STOPPED!!!!!!!!!!")
