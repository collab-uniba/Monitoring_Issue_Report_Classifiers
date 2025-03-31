import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """
    Manage configuration loading, validation, and default settings.
    """
    def __init__(self, config_path: str):
        """
        Initialize ConfigManager with configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._set_defaults()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Loaded configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def _validate_config(self):
        """
        Validate core configuration parameters.
        Raise ValueError for missing or invalid configurations.
        """
        required_keys = [
            'project_name', 
            'start_year', 
            'end_year', 
            'split_type', 
            'range',
            'model_type'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

    def _set_defaults(self):
        """
        Set default values for optional configuration parameters.
        """
        defaults = {
            'model_type': 'roberta',
            'use_validation': True,
            'split_size': 0.3,
            'label_set': [],
            'results_root': 'results',
            'training_args': {
                'num_train_epochs': 3,
                'per_device_train_batch_size': 16,
                'per_device_eval_batch_size': 64,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'logging_dir': 'logs',
                'learning_rate': 5e-5,
                'adam_epsilon': 1e-8
            }
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value

    def get_config(self) -> Dict[str, Any]:
        """
        Get the processed configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config

    def save_config(self, save_path: str = None):
        """
        Save current configuration to a YAML file.
        
        Args:
            save_path: Optional path to save configuration
        """
        if save_path is None:
            save_path = self.config_path

        with open(save_path, 'w') as file:
            yaml.dump(self.config, file)