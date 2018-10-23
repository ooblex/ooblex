# Ooblex
Deployable system for real-time streaming distributed processing, targeting mainly live media and neural network inference.

## Basic System Layout
![Flow](untitled_diagram.png)

## Server Requirements

Ooblex is comprised of several services, including those for media and inference. While these services can be run independently, they can also be run together on a single server enviroment.

Ooblex was developed for use on a server with a solid internet connection, an Intel 28-core CPU, and 16-GB of RAM. System requirements will vary greatly depending on application, however our goal is to allow for deployment onto even edge devices, such as a Raspberry Pi 3 with an attached Intel Movidous AI accelerator.

A domain name is required for a full installation as SSL is often required for WebRTC browser support. Point a valid domain (or subdomain) at your public server's IP address.
It is also recommended that all ports be made available and open to the server, as port tunneling and WebRTC TURN servers are out of scope of the provided support.

## Installing onto Ubuntu 18.04

We're are assuming a working and fresh deployment of a Ubuntu server. 
If a GRUB error occurs during installation, please see: https://askubuntu.com/questions/1040974/ubuntu-server-18-04-apt-get-fails
```
cd ~
sudo apt-get update
sudo apt-get install git
git clone https://github.com/ooblex-ai/ooblex/

cd ooblex
sudo chmod +x *.sh
sudo ./install_opencv.sh
```
[building OpenCV can take an hour or more]
```
sudo ./install_nginx.sh
```
[will request information about a domain name at this point to generate a free SSL certificate. If asked, redirecting from HTTP to HTTPS if asked is preferable]
```
sudo ./install_tensorflow.sh
sudo ./install_gstreamer.sh
sudo ./install_janus.sh
sudo ./install_redis.sh
```
You will need to setup a REDIS and RabbitMQ server of your own. We suggest using a seperate server, perhaps hosted by a cloud provider, rather deploying these services on this install environement.  This will enable many Ooblex deployments to access those systems as a shared resource.

```
sudo ./install_rtc.sh
```
You may need to configure another SSL and domain for the RTC server. It may already be setup though.

The dependencies for Ooblex are largely now installed. We can enter the code directory now and configure some final settings.

```
cd ~/ooblex/code
python3 api.py
```
Running the above code will likely show an error -- you will need to modify the api.py file with your domain name, as defined earlier for the SSL certificates. To do so, try the following command to open a file editor, going to the end of the document, and updating the file with the correct settings. 

```
sudo nano api.py
```
#### server = SimpleSSLWebSocketServer("", 8800, SimpleChat, "/etc/letsencrypt/live/api.ooblex.com/fullchain.pem", "/etc/letsencrypt/live/api.ooblex.com/privkey.pem", version=ssl.PROTOCOL_TLSv1)
to
#### server = SimpleSSLWebSocketServer("", 8800, SimpleChat, "/etc/letsencrypt/live/MYDOMAIN/fullchain.pem", "/etc/letsencrypt/live/MYDOMAIN/privkey.pem", version=ssl.PROTOCOL_TLSv1)

Running the api.py file with Python should now start the main API server, however there are several other services that we will need to run.  You can stop this server for now, or push it to the background. (control-Z & bg )

Next, Copy the files in the HTML folder to your /var/www/html folder.
You will need to MODIFY both the index.html file AND the Ooblex.v0.js file contained within the JS folder
You will need to update the domain names from api.ooblex.com to whatever your domain is.

At this point, going to your domain in your browser should show a yellow display. 

Next, in ~/ooblex/code, we can try running our next service:

```
cd ~/ooblex/code
python3 decoder.py
```

Any error about image folders can be ignored at this point. If things otherwise works, great! You will likely want to modify the file so that the REDIS and AMPQ connection strings point to your own servers, rather the Ooblex's demo servers.

Try the same thing with the pub.py file, to see if it runs without errors. You will need to configure this to point to your own REDIS server, rather than what might be set by default.

```
cd ~/ooblex/code
python3 pub.py
```

You will also need to test red.py.  This will also require the SSL certificate location to be properly configured. You can use nano or vim to do this. ie: sudo nano.py  

```
python3 red.py
```

Other than NGINX, with the HTML files, you will need to have the following servers running at the same time.

```
sudo /opt/janus/bin/janus -o & python3 api.py & python3 brain.py & python3 decoder.py & python3 red.py & python3 rtc.py &
```
rtc.py will fail if Janus is also not started and configured first.  Please see the Janus_config folder for instructions on how to configure Janus.  Start the Janus once configured using the following command, and then try starting rtc.py again.
```
sudo /opt/janus/bin/janus -o
```

or a one liner for everything
```
cd  ~/ooblex/code/
sudo /opt/janus/bin/janus -o & python3 api.py & python3 brain.py & python3 decoder.py & python3 red.py & python3 rtc.py &
```

Ensuring that Janus's socket server layer works is required to get rtc.py working also. 

Lastly, once the system is all configured, and each of the several servers are running all together, it is possible to modify the brain.py file, which contains the tensor threads.

The brain.py is configured to operate with a popular video-based facial recognition Tensorflow models, along with some face transformation models (Trump, Taylor), which can be downloaded as needed and implemnted. 

Due to the size of the models, they cannot be hosted directly on github, but they can be downloaded from here: https://api.ooblex.com/models/

More detailed explainations on how to program the Tensorthreads will be helpful, but for now using the brain.py as a template for your own AI scripts is intended. It is quite accessible if familiar with Python. Working with IBM's Watson Studio, exporting a Python-based version of a trained model can be directly imported into this brain.py file for rapid deployment of a high performing, low-latency. serialized model.

Information on the core server files at play:
```
red.py -- JPEG streaming server for low-latency output
api.py -- The main REST API server,  used for communicating with Ooblex and orchestrating manhy of the system components.
brain.py -- This contains the Tensor Thread code as a wrapper for a Python-based TensorThread model. It is pre-configured with example logic.
rtc.py -- This is the main API layer for the the WebRTC service
decoder.py -- This is the main live media deocder thread, configured for live webRTC video ingestion.
```
pixel_shuffler.py, npy files, and model.py files support the alread-configured AI models loaded in brain.py.  These can be modified or removed as needed, depending on changes to brain.py

