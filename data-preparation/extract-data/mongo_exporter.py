import csv
from pymongo import MongoClient
from config import Config
from utils import setup_logging, ensure_directories, remove_unprintable

class MongoExporter:
    def __init__(self):
        self.config = Config()
        setup_logging()
        ensure_directories([self.config.EXPORT_PATH])
        
    def connect_to_mongodb(self):
        """Establish MongoDB connection."""
        client = MongoClient(self.config.MONGO_CONFIG['host'])
        return client[self.config.MONGO_CONFIG['database']]
    
    def export_collection(self, collection):
        """Export a single collection to CSV."""
        projection = {
            "_id": 0,
            **{v: 1 for v in self.config.CSV_FIELD_MAPPINGS['input_fields'].values()}
        }
        
        cursor = collection.find(filter={}, projection=projection)
        output_file = self.config.EXPORT_PATH / f"{collection.name}.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=list(self.config.CSV_FIELD_MAPPINGS['input_fields'].values()),
                quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()
            
            for doc in cursor:
                cleaned_doc = {
                    field: remove_unprintable(doc.get('fields', {}).get(field.split('.')[-1], ''))
                    for field in self.config.CSV_FIELD_MAPPINGS['input_fields'].values()
                }
                writer.writerow(cleaned_doc)
    
    def run(self):
        """Run the export process for all collections."""
        db = self.connect_to_mongodb()
        for collection_name in db.list_collection_names():
            self.export_collection(db[collection_name])
            logging.info(f"Exported collection: {collection_name}")