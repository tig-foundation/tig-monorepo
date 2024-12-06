# in order to access pgadmin from a remote machine, we need to set up a tunnel.

run the following command:
```
ssh -L 5432:localhost:5432 username@server-ip # postgre
```

now open pgadmin while keeping the tunnel open and
    - Add new server
        - Connection:
            Host name/address: `localhost`
            Port: `5432`
            Username: `your_username_here`
            Password: `your_password_here`
    - Click on save
    - Under servers you should now see your server
    - Now you can connect to your database

alternatively, one can also use a SSH tunnel to access the pgadmin from a remote machine:

run the following command:
```
ssh -L 8888:localhost:8888 username@server-ip # pgadmin
```

now navigate to `http://localhost:8888` to access pgadmin.