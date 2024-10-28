#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating necessary directories..."

mkdir -p ./db_data
mkdir -p ./db_replica_data
mkdir -p ./backups
mkdir -p ./db_backup

echo "Setting permissions for data directories..."

# Adjust permissions to allow PostgreSQL container to access the directories. Bitnami Default UserId (1001)
sudo chown -R 1001:1001 ./db_data
sudo chown -R 1001:1001 ./db_replica_data

echo "Creating db_backup/Dockerfile..."

cat > ./db_backup/Dockerfile <<'EOL'
FROM postgres:latest

RUN apt-get update && apt-get install -y cron

COPY backup.sh /usr/local/bin/backup.sh
RUN chmod +x /usr/local/bin/backup.sh

COPY crontab /etc/cron.d/db_backup_cron
RUN chmod 0644 /etc/cron.d/db_backup_cron
RUN crontab /etc/cron.d/db_backup_cron

CMD ["cron", "-f"]
EOL

echo "Creating db_backup/backup.sh..."

cat > ./db_backup/backup.sh <<'EOL'
#!/bin/bash

export PGPASSWORD=${POSTGRES_PASSWORD}
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
pg_dump -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -Fc -f "/backups/db_backup_$TIMESTAMP.dump"
EOL

echo "Creating db_backup/crontab..."

cat > ./db_backup/crontab <<'EOL'
0 * * * * root /usr/local/bin/backup.sh >> /var/log/cron.log 2>&1
EOL

echo "Setting execute permissions for backup.sh..."

chmod +x ./db_backup/backup.sh

# -----------------------------------------------
# Additions for the failover.sh script
# -----------------------------------------------

echo "Creating failover.sh script..."

cat > failover.sh <<'EOL'
#!/bin/bash

failed_node=$1
new_primary_node=$2

echo "Failover triggered. Failed node: $failed_node, Promoting node: $new_primary_node"

if [ "$failed_node" = "0" ]; then
    echo "Promoting replica to primary..."
    PGPASSWORD=$POSTGRES_PASSWORD psql -h db_replica -U $POSTGRES_USER -d postgres -c "SELECT pg_promote();"
    if [ $? -eq 0 ]; then
        echo "Replica promoted to primary."
    else
        echo "Failed to promote replica."
        exit 1
    fi
else
    echo "No action needed for failed node $failed_node"
fi
EOL

echo "Setting execute permissions for failover.sh..."

chmod +x failover.sh

echo "Setup complete. You can now start the Docker services with 'docker-compose up -d'."
