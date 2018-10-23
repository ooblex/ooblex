Put these files in your JANUS CONFIG folder, and configure them accordingly. 

```
/opt/janus/etc/janus
```

You will also need to modify the AMPQ / RabbitMQ configuration settings to point to the correct address.

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
