sudo apt-get update
apt-get install nginx -y

sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:certbot/certbot
sudo apt-get update
sudo apt-get install python-certbot-nginx -y

sudo certbot --nginx

