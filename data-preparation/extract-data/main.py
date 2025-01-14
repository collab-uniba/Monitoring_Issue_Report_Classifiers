from config import Config
from mongo_exporter import MongoExporter
from csv_cleaner import CSVCleaner
from label_mapper import LabelMapper
from utils import ensure_directories

def main():
    """Main execution flow."""
    config = Config()
    
    # Ensure all required directories exist
    ensure_directories([
        config.BASE_PATH,
        config.STANDARD_PATH,
        config.EXPORT_PATH,
        config.CLEANED_PATH,
        config.MAPPED_PATH
    ])
    
    # Run the pipeline
    MongoExporter().run()
    CSVCleaner().run()
    LabelMapper().run()

if __name__ == "__main__":
    main()