sudo apt update
sudo apt upgrade
sudo apt install rabbitmq-server -y
sudo rabbitmq-plugins enable rabbitmq_management
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
