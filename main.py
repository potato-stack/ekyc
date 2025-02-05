# Entry point
import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse
from api.routes import app
from utils.config import Config

# Load the configuration from the JSON file
config = Config("config.json")

# Get server settings from the configuration
server_settings = config.get_server_settings()
host = server_settings.get("host", "127.0.0.1")  # Default to "127.0.0.1" if not set
port = server_settings.get("port", 8000)         # Default to 8000 if not set

if __name__ == "__main__":
    # Run the application, allow print screen for debug, allow reload and ignore test_script on change
    uvicorn.run("api.routes:app", host=host, port=port, reload=True, reload_excludes="./test_script/*", log_level="info")
