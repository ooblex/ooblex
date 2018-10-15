apt-get install redis -y && \
apt-get install redis-tools -y && \
redis-cli -h REDISSERVERURLHERE -p PASSWORDHERE -a admin  config set notify-keyspace-events KEA
