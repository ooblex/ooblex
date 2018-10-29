Put these files in your JANUS CONFIG folder, and configure them accordingly. 

```
cd ~/ooblex/janus_confs
sudo cp * /opt/janus/etc/janus
```

You will also need to modify the AMPQ / RabbitMQ configuration settings to point to the correct address. The config.py file settings are not compatible with Janus' config file, so you will have to update them manually.

You will also need to point the config files to the correct SSL certificate; we generated this SSL certificate with the CERTBOT step earlier in the Ooblex install process.


The following files + lines need updating
```
janus.cfg:67:cert_pem = /etc/letsencrypt/live/api.ooblex.com/fullchain.pem
janus.cfg:68:cert_key = /etc/letsencrypt/live/api.ooblex.com/privkey.pem
janus.transport.http.cfg:64:cert_pem = /etc/letsencrypt/live/api.ooblex.com/fullchain.pem
janus.transport.http.cfg:65:cert_key = /etc/letsencrypt/live/api.ooblex.com/privkey.pem
janus.transport.websockets.cfg:37:cert_pem = /etc/letsencrypt/live/api.ooblex.com/fullchain.pem
janus.transport.websockets.cfg:38:cert_key = /etc/letsencrypt/live/api.ooblex.com/privkey.pem
```

If feeling aggressive, you can try a one liner to do this:
```
sudo find /opt/janus/etc/janus -type f -exec sed -i 's/api.ooblex.com/YOU-DOMAIN-HERE/g' {} \;
```

Once configured, you can start janus with the following:

```
sudo /opt/janus/bin/janus -o
```
