from config import Config
from mongo_exporter import MongoExporter
from csv_cleaner import CSVCleaner
from label_mapper import LabelMapper
from utils import ensure_directories
import argparse
from typing import List, Optional

class Pipeline:
    """Data processing pipeline that can run specified steps."""
    
    def __init__(self, cli_collections: Optional[List[str]] = None):
        """
        Initialize pipeline with optional CLI-specified collections.
        
        Args:
            cli_collections: Optional list of collection names from CLI arguments
        """
        self.config = Config()
        self.collections = Config.get_collections(cli_collections)
        self.steps = {
            'export': self._run_export,
            'clean': self._run_clean,
            'map': self._run_map
        }
        
    def _run_export(self):
        """Run MongoDB export step."""
        print("Running MongoDB export...")
        if self.collections:
            print(f"Processing collections: {', '.join(self.collections)}")
        else:
            print("Processing all collections")
        MongoExporter(collections=self.collections).run()
        
    def _run_clean(self):
        """Run CSV cleaning step."""
        print("Running CSV cleaning...")
        CSVCleaner(collections=self.collections).run()
        
    def _run_map(self):
        """Run label mapping step."""
        print("Running label mapping...")
        LabelMapper(collections=self.collections).run()
        
    def run_steps(self, selected_steps: List[str]):
        """Run specified pipeline steps in order."""
        # Ensure all required directories exist
        ensure_directories([
            self.config.BASE_PATH,
            self.config.STANDARD_PATH,
            self.config.EXPORT_PATH,
            self.config.CLEANED_PATH,
            self.config.MAPPED_PATH
        ])
        
        # Validate steps
        invalid_steps = [step for step in selected_steps if step not in self.steps]
        if invalid_steps:
            raise ValueError(f"Invalid steps specified: {invalid_steps}. "
                           f"Valid steps are: {list(self.steps.keys())}")
        
        # Run selected steps
        for step in selected_steps:
            self.steps[step]()

def main():
    """Main execution flow with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run specific steps of the data processing pipeline.')
    parser.add_argument('--steps', nargs='+', default=['export', 'clean', 'map'],
                       help='Steps to run in order. Valid steps: export, clean, map')
    parser.add_argument('--collections', nargs='+',
                       help='Optional list of collections to export. Overrides config.COLLECTIONS if specified.')
    
    args = parser.parse_args()
    
    try:
        pipeline = Pipeline(cli_collections=args.collections)
        pipeline.run_steps(args.steps)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise

if __name__ == "__main__":
    main()