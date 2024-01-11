import csv
from pymongo import MongoClient
import logging

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

# Destination folder for CSV files
output_folder = 'CSV/'

# For each collection, run the query and export the results to a CSV file
for collection_name in collections:
    collection = db[collection_name]
    
    filter={}
    
    project = {
        "_id": 0,
        "fields.summary": 1,
        "fields.description": 1,
        "fields.issuetype.name": 1,
        "fields.created": 1
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
        fieldnames = ["fields.summary", "fields.description", "fields.issuetype.name", "fields.created"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for document in cursor:
            fields_data = document.get('fields', {})
            
            # Apply the non-printing character removal function
            fields_summary = remove_unprintable(fields_data.get('summary', ''))
            fields_description = remove_unprintable(fields_data.get('description', ''))
            fields_issuetype_name = remove_unprintable(fields_data.get('issuetype', {}).get('name', ''))
            fields_created = remove_unprintable(fields_data.get('created', ''))

            writer.writerow({
                "fields.summary": fields_summary,
                "fields.description": fields_description,
                "fields.issuetype.name": fields_issuetype_name,
                "fields.created": fields_created
            })
        
logging.info("Process completed.")
