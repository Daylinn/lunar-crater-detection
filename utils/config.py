import yaml
from pathlib import Path
import os
from typing import Dict, Any

def load_config(config_file: str = 'config.yaml') -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / config_file
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Convert relative paths to absolute paths
        for key in ['raw_data_path', 'processed_data_path']:
            if key in config['data']:
                config['data'][key] = str(project_root / config['data'][key])
                
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {
            'data': {
                'raw_data_path': str(project_root / 'data/raw'),
                'processed_data_path': str(project_root / 'data/processed')
            }
        }

def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the project.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    directories = [
        config['data']['raw_data_path'],
        config['data']['processed_data_path'],
        config['data']['results_path'],
        config['logging']['log_dir']
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True) 