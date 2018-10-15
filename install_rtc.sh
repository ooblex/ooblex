sudo apt-get install python3-pip
sudo pip3 install websockets
sudo pip3 install amqpstorm
sudo pip3 install git+https://github.com/dpallot/simple-websocket-server.git
sudo apt-get update
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:certbot/certbot
sudo apt-get update
sudo apt-get install certbot  -y
certbot certonly
