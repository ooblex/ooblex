These files can be used to configure the python files as launchable services.

copy services files to system service folder
ie:
```
cp webrtc.service /etc/systemd/system/webrtc.service
```
ensure service files have have chmod 644 permissions

also ensure log folder is created: /root/logs/

We then then want to reset things and ensure scripts boot at start
```
sudo systemctl daemon-reload
sudo systemctl enable myscript.service
```
scripts can be tested by means of: systemctl start webrtc

I'm using the default janus service script, so the logs should go to the default location I think.
