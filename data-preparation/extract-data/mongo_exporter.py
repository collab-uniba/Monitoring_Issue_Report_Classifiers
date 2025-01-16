import csv
from pymongo import MongoClient
from config import Config
from utils import setup_logging, ensure_directories, remove_unprintable
import logging
from typing import Optional, List

class MongoExporter:
    def __init__(self, collections: Optional[List[str]] = None):
        """
        Initialize MongoExporter with optional collection filter.
        
        Args:
            collections: Optional list of collection names to export. If None, exports all collections.
        """
        self.config = Config()
        self.collections_filter = collections
        setup_logging()
        ensure_directories([self.config.EXPORT_PATH])
        
    def connect_to_mongodb(self):
        """Establish MongoDB connection."""
        client = MongoClient(self.config.MONGO_CONFIG['host'])
        return client[self.config.MONGO_CONFIG['database']]
    
    def get_collections_to_export(self, db) -> List[str]:
        """
        Get list of collections to export based on filter.
        
        Args:
            db: MongoDB database connection
            
        Returns:
            List of collection names to export
        """
        available_collections = db.list_collection_names()
        
        if not self.collections_filter:
            return available_collections
            
        # Validate that all requested collections exist
        invalid_collections = set(self.collections_filter) - set(available_collections)
        if invalid_collections:
            raise ValueError(
                f"The following collections were not found in the database: {invalid_collections}. "
                f"Available collections are: {available_collections}"
            )
            
        return self.collections_filter
    
    def export_collection(self, collection):
        """
        Export a single collection to CSV.
        
        Args:
            collection: MongoDB collection to export
        """
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
                
        logging.info(f"Exported collection: {collection.name}")
    
    def run(self):
        """Run the export process for specified collections."""
        db = self.connect_to_mongodb()
        collections_to_export = self.get_collections_to_export(db)
        
        if not collections_to_export:
            logging.warning("No collections found to export!")
            return
            
        logging.info(f"Starting export of collections: {collections_to_export}")
        
        for collection_name in collections_to_export:
            self.export_collection(db[collection_name])
            
        logging.info("Export process completed")