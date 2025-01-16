#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update and install prerequisites
sudo apt update
sudo apt install -y wget curl gnupg unzip

# Check if MongoDB is already installed
if mongod --version &>/dev/null; then
    echo "MongoDB is already installed. Skipping installation."
else
    # Import MongoDB public GPG key
    curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
       sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
       --dearmor

    # Create the list file for MongoDB 8.0 on Ubuntu 24.04
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

    # Reload the package database
    sudo apt-get update

    # Install MongoDB 8.0
    sudo apt-get install -y mongodb-org

    # Start and enable MongoDB service
    sudo systemctl start mongod
    sudo systemctl enable mongod

    # Verify MongoDB is running
    if systemctl is-active --quiet mongod; then
        echo "MongoDB is running."
    else
        echo "MongoDB failed to start." >&2
        exit 1
    fi
fi

# Download the ZIP file
JIRA_ZIP_URL="https://zenodo.org/records/7182101/files/ThePublicJiraDataset.zip?download=1"
JIRA_ZIP_FILE="ThePublicJiraDataset.zip"
sudo wget -O "$JIRA_ZIP_FILE" "$JIRA_ZIP_URL"

# Extract the ZIP file
sudo unzip -o "$JIRA_ZIP_FILE" -d jira_dataset

# Run mongorestore
echo "Restoring MongoDB dump..."
mongorestore --gzip --archive="jira_dataset/ThePublicJiraDataset/3. DataDump/mongodump-JiraRepos.archive" --nsFrom "JiraRepos.*" --nsTo "JiraRepos.*"

echo "Done!"
