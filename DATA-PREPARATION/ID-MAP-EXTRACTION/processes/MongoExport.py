import csv
import logging
import os
import sys

from pymongo import MongoClient

# Set the global logging level to INFO
logging.getLogger().setLevel(logging.INFO)
# Handler for terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# Set up the connection to your MongoDB database
client = MongoClient('mongodb://localhost:27017')

# Replace 'JiraRepos' with your database name
db = client['JiraRepos']

# List of collections in the database
collections = db.list_collection_names()

# Base Path
base_path = 'CSV/'
# Destination folder for CSV files
output_folder = os.path.join(base_path,'ID-MAP/EXPORT/')
# Create the 'CSV' directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# For each collection, run the query and export the results to a CSV file
for collection_name in collections:
    collection = db[collection_name]
    
    filter={}
    
    project = {
        "_id": 0,
        "id": 1,
        "fields.issuetype.name": 1,
    }

    # Query the collection
    cursor = collection.find(
        filter=filter,
        projection=project
    )

    # Name of the destination CSV file
    output_file = f"{output_folder}{collection_name}.csv"
    
    # Function to delete non-printable control characters
    def remove_unprintable(text):
        if text is None:
            return ''
        return ''.join(char for char in text if char.isprintable())
    
    # Export query results to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["id","fields.issuetype.name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for document in cursor:
            fields_data = document.get('fields', {})
            
            # Apply the non-printing character removal function
            id = remove_unprintable(document.get('id', ''))
            fields_issuetype_name = remove_unprintable(fields_data.get('issuetype', {}).get('name', ''))

            writer.writerow({
                "id": id,
                "fields.issuetype.name": fields_issuetype_name,
            })
        
logging.info("Process completed.")
