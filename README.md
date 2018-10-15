# ooblex

#Server Requirements

Ooblex is comprised of several services, including those for media and inference. While these services can be run independently, they can also be run together on a single server enviroment.

Ooblex was developed for use on a server with a solid internet connection, an Intel 28-core CPU, and 16-GB of RAM. System requirements will vary greatly depending on application, however our goal is to allow for deployment onto even edge devices, such as a Raspberry Pi 3 with an attached Intel Movidous AI accelerator.

A domain name is required for a full installation as SSL is often required for WebRTC browser support. Point a valid domain (or subdomain) at your public server's IP address.
It is also recommended that all ports be made available and open to the server, as port tunneling and WebRTC TURN servers are out of scope of the provided support.

#Installing onto Ubuntu 18.04

We're are assuming a working and fresh deployment of a Ubuntu server. 
If a GRUB error occurs during installation, please see: https://askubuntu.com/questions/1040974/ubuntu-server-18-04-apt-get-fails

cd ~
sudo apt-get update
sudo apt-get install git
git clone https://github.com/ooblex-ai/ooblex/

cd ooblex
sudo chmod +x *.sh
sudo ./install_opencv.sh
[building OpenCV can take an hour or more]

sudo ./install_nginx.sh
[will request information about a domain name at this point to generate a free SSL certificate]

sudo ./install_tensorflow.sh
