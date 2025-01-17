from pathlib import Path
from typing import Optional, List

class Config:
    # Collection configuration
    COLLECTIONS = [
        'Apache',
        'Jira',
        'RedHat',
        'MongoDB'
    ]  # Default collections to process, None means process all
    
    # Base paths
    BASE_PATH = Path('data')
    
    EXPORT_PATH = BASE_PATH / 'export'
    CLEANED_PATH = BASE_PATH / 'cleaned'
    MAPPED_PATH = BASE_PATH / 'mapped'
    WINDOWS_PATH = BASE_PATH / 'windows'
    
    # MongoDB configuration
    MONGO_CONFIG = {
        'host': 'mongodb://localhost:27017',
        'database': 'JiraRepos'
    }
    
    # Field mappings for CSV processing
    CSV_FIELD_MAPPINGS = {
        'input_fields': {
            'summary': 'fields.summary',
            'description': 'fields.description',
            'issue_type': 'fields.issuetype.name',
            'created_date': 'fields.created'
        },
        'output_fields': ['title', 'body', 'label', 'date']
    }
    
    # Label mapping patterns
    bug_synonyms = ['bug', 'report']
    feature_synonyms = ['new feature', 'improvement', 'suggestion', 'feature request', 'enhancement']
    question_synonyms = ['support request', 'question']

    LABEL_PATTERNS = {
        rf"({'|'.join(bug_synonyms)})": 'bug',
        rf"({'|'.join(feature_synonyms)})": 'feature',
        rf"({'|'.join(question_synonyms)})": 'question'
    }

    @classmethod
    def get_collections(cls, cli_collections: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        Get collections to process based on CLI arguments and config.
        
        Args:
            cli_collections: Collections specified via CLI arguments
            
        Returns:
            List of collections to process or None for all collections
        """
        # CLI collections take precedence if specified
        if cli_collections:
            return cli_collections
            
        # Otherwise use config collections if specified
        if cls.COLLECTIONS:
            return cls.COLLECTIONS
            
        # If neither is specified, return None to process all collections
        return None