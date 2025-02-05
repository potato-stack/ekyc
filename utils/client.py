
import os
import lmdb
import httpx
import numpy as np
from .config import Config

# Load configuration
config = Config(config_path="config.json")

# POST request for fetching data
class ExternalServiceClient:
    """Client for handling fetching data from external APIs"""
    def __init__(self):
        self.api_url = config.get_external_service_url()
    
    async def fetch_data(self, endpoint, fetch_type="all", data = None):
        """
        Fetch data from an external API or sync up the local data

        :param endpoint: API endpoint from the server
        :param fetch_type: "all" for fetching all data, "change" for fetching only change data
        :param data: Post request custom data to send to the API
        """

        async with httpx.AsyncClient() as client:
            if fetch_type == "all":
                # Fetch all user data: 
                response = await client.get(f"{self.api_url}/{endpoint}/fetch_all")
            elif fetch_type == "change":
                # Fetch only changed data:
                response = await client.get(f"{self.api_url}/{endpoint}/fetch_change")
            else:
                # Fetch for custom data
                if data:
                    response = await client.post(f"{self.api_url}/{endpoint}", json=data)
                else:
                    response = await client.get(f"{self.api_url}/{endpoint}")
            
            # Resolve for the respone
            if response.status_code == 200:
                return response.json()
            else:
                return {"Error": f"Failed to fetch data from {fetch_type}"}

class LMDBCache:
    """Temp cached for storing embedding data"""
    def __init__(self, path="temp/lmdb_cache", map_size=10**9):
        if not os.path.exists(path):
            os.makedirs(path)
        self.env = lmdb.open(path, map_size=map_size)

    def store_embedding(self, user_id: str, orientation: str, embedding: np.ndarray):
        with self.env.begin(write=True) as store:
            store.put(f"{user_id}_{orientation}".encode(), np.array(embedding).tobytes())
    
    def load_embedding(self, user_id: str = None):
        embedding = {}
        with self.env.begin()as load:
            iterate = load.cursor()
            if user_id:
                prefix = f"{user_id}_".encode()
                for id, value in iterate:
                    if id.startwidth(prefix):
                        orientation = id.decode().split('_')[-1]  
                        if user_id not in embedding:
                            embedding[user_id] = {}
                        embedding[user_id][orientation] = np.frombuffer(value, dtype=np.float32)
            else:
                for id, value in iterate:
                    user_id, orientation = id.decode().split('_')[-1]  
                    embedding[user_id][orientation] = np.frombuffer(value, dtype=np.float32)
        
        return embedding
        
    def get_orientations(self, user_id: str):
        """Return a user valid orientation"""
        orientations = []
        with self.env.begin(write=True) as orients:
            iterate = orients.cursor()
            prefix = f"{user_id}_".encode()
            for key, _ in iterate:
                if key.startswith(prefix):
                    orientation = key.decode().split('_')[-1]  # Extract orientation from key
                    orientations.append(orientation)
        return orientations



    def delete_embedding(self, user_id: str = "all"):
        with self.env.begin(write=True) as delete:
            iterate = delete.cursor()
            if user_id != "all":
                # Delete all embeddings for the specific user
                prefix = f"{user_id}_".encode()
                for key, _ in iterate:
                    if key.startswith(prefix):
                        delete.delete(key)
            else:
                # Delete all embeddings for all users
                for key, _ in iterate:
                    delete.delete(key)

# Synchronize function to sync up with server's DB 
async def sync_embeddings_from_server():
    """Periodically fetch user embeddings from the external service and update LMDB cache."""
    external_service_client = ExternalServiceClient()
    lmdb_cache = LMDBCache()

    # Fetch new or modified face embeddings from the external service
    new_user_embeddings = await external_service_client.fetch_data("sync")

    # Store the embeddings in LMDB
    for user_id, embedding in new_user_embeddings.items():
        lmdb_cache.store_embedding(user_id, embedding)
        print(f"Updated embedding for {user_id}")