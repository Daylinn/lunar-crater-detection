import requests
import json
from pathlib import Path
import time
from typing import List, Dict
import random
from tqdm import tqdm
import os
import sys

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import load_config
from utils.api_config import get_api_key

class LROCDownloader:
    def __init__(self, api_key: str = None):
        """
        Initialize the LROC downloader.
        
        Args:
            api_key (str, optional): NASA API key
        """
        self.api_key = api_key
        self.base_url = "https://api.nasa.gov/planetary/earth/imagery"
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else None
        }
    
    def search_images(
        self,
        lat: float,
        lon: float,
        dim: float = 0.025,
        date: str = None
    ) -> List[Dict]:
        """
        Search for images at a specific location.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            dim (float): Dimension in degrees
            date (str, optional): Date in YYYY-MM-DD format
            
        Returns:
            List[Dict]: List of image metadata
        """
        params = {
            "lat": lat,
            "lon": lon,
            "dim": dim,
            "api_key": self.api_key
        }
        
        if date:
            params["date"] = date
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return [{"url": response.url}]
        except requests.exceptions.HTTPError as e:
            print(f"Error searching for images at coordinates {lat}, {lon}: {str(e)}")
            return []
    
    def download_image(self, url: str, save_path: Path) -> bool:
        """
        Download an image from a URL.
        
        Args:
            url (str): URL of the image
            save_path (Path): Path to save the image
            
        Returns:
            bool: True if download was successful
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

def get_lunar_coordinates() -> List[Dict[str, float]]:
    """
    Get a list of lunar coordinates where craters are known to exist.
    
    Returns:
        List[Dict[str, float]]: List of coordinates
    """
    # These are example coordinates of well-known lunar craters
    return [
        {"lat": 20.2, "lon": 30.8},  # Tycho
        {"lat": 29.4, "lon": -35.0},  # Copernicus
        {"lat": 43.4, "lon": -10.2},  # Plato
        {"lat": -8.9, "lon": 15.5},   # Archimedes
        {"lat": 25.3, "lon": 0.7},    # Ptolemaeus
        {"lat": -19.9, "lon": 175.6}, # Tsiolkovskiy
        {"lat": 36.4, "lon": 57.4},   # Aristoteles
        {"lat": -44.1, "lon": 31.5},  # Theophilus
        {"lat": 13.3, "lon": 5.2},    # Albategnius
        {"lat": -51.6, "lon": 169.3}, # Schrödinger
    ]

def main():
    """Main function to download lunar crater images."""
    # Load configuration
    config = load_config('config.yaml')
    
    # Get API key from config file
    api_key = get_api_key()
    print(f"\nUsing API key: {api_key[:8]}...")  # Only print first 8 characters for security
    
    if api_key == "DEMO_KEY":
        print("Warning: Using DEMO_KEY. Please edit config/api_config.json with your actual NASA API key")
    
    downloader = LROCDownloader(api_key)
    
    # Create directories
    raw_dir = Path(config['data']['raw_data_path'])
    images_dir = raw_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get lunar coordinates
    coordinates = get_lunar_coordinates()
    
    # Download images
    for i, coord in enumerate(tqdm(coordinates, desc="Downloading images")):
        # Search for images
        results = downloader.search_images(
            lat=coord["lat"],
            lon=coord["lon"],
            dim=0.025
        )
        
        if not results:
            print(f"No images found for coordinates {coord}")
            continue
        
        # Download the first image
        image_url = results[0]["url"]
        save_path = images_dir / f"crater_{i:04d}.jpg"
        
        if downloader.download_image(image_url, save_path):
            print(f"Successfully downloaded image {i}")
        else:
            print(f"Failed to download image {i}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)

if __name__ == '__main__':
    main() 