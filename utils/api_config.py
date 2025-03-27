import os
from pathlib import Path

def get_api_key() -> str:
    """
    Get the NASA API key from a configuration file.
    
    Returns:
        str: The API key
    """
    config_path = Path(__file__).parent.parent / 'config' / 'api_config.json'
    
    if not config_path.exists():
        # Create default config file if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write('{"nasa_api_key": "DEMO_KEY"}')
        print(f"Created default config file at {config_path}")
        print("Please edit this file with your actual NASA API key")
        return "DEMO_KEY"
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('nasa_api_key', 'DEMO_KEY')
    except Exception as e:
        print(f"Error reading API key from config file: {str(e)}")
        return "DEMO_KEY" 