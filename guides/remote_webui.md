# Remote Access WebUI

in order to be able to access the web ui from a remote machine, we need to set up either a reverse proxy or a tunnel.

## Using a tunnel

run the following command in a terminal:
```
ssh -L 8083:localhost:8083 -L 3336:localhost:3336 username@server-ip # webui, master
```
then navigate to `https://localhost:8083` in your browser while keeping the tunnel open.