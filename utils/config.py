import json
import os

class Config:
    """Loads configuration from setting JSON file."""
    def __init__(self, config_path="config.json"):
        self.config = None
        self.load_config(config_path)
    
    def load_config(self, config_path):
        """Loads and parses the JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_server_settings(self):
        """Returns server settings like host and port."""
        return self.config.get("server", {})
    
    def get_database_settings(self):
        """Returns database connection settings."""
        return self.config.get("database", {})
    
    def get_external_service_url(self):
        """Returns external service URL."""
        return self.config.get("external_service", {}).get("api_url")
