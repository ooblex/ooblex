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
[will request information about a domain name at this point to generate a free SSL certificate]
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
*
server = SimpleSSLWebSocketServer("", 8800, SimpleChat, "/etc/letsencrypt/live/api.ooblex.com/fullchain.pem", "/etc/letsencrypt/live/api.ooblex.com/privkey.pem", version=ssl.PROTOCOL_TLSv1)
* 
