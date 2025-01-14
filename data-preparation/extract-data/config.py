from pathlib import Path

class Config:
    # Base paths
    BASE_PATH = Path('CSV')
    
    # Folder structure
    STANDARD_PATH = BASE_PATH / 'STANDARD'
    EXPORT_PATH = STANDARD_PATH / 'EXPORT'
    CLEANED_PATH = STANDARD_PATH / 'CLEANED'
    MAPPED_PATH = STANDARD_PATH / 'MAPPED'
    
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

