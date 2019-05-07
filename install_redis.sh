
apt-get install redis -y && \
apt-get install redis-tools -y && \
pip3 install redis 
redis-cli -h localhost


// To test to see if your remotely hosted REDIS server is installed instead, if cloud hosted for example for use in a cluster...
// redis-cli -h portal1369-16.bmix-dal-yp-cbfabb84-69f1-472d-87d7-343fd70e44c6.steve-seguin-email.composedb.com -p 40299 -a admin  config set notify-keyspace-events KEA
