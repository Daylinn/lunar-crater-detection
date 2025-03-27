import requests
from pathlib import Path
import time
from typing import List, Dict
from tqdm import tqdm
import os
import sys
import json

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import load_config

class LunarImageDownloader:
    def __init__(self):
        """Initialize the lunar image downloader."""
        self.search_url = "https://wms.lroc.asu.edu/search/product-search.php"
        self.download_url = "https://wms.lroc.asu.edu/data"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def download_image(self, lat: float, lon: float, save_path: Path) -> bool:
        """
        Download a lunar image for given coordinates.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            save_path (Path): Path to save the image
            
        Returns:
            bool: True if download was successful
        """
        try:
            # Search for images near the coordinates
            params = {
                "lat": lat,
                "lon": lon,
                "rad": 0.1,  # 0.1 degree radius
                "limit": 1,  # Get only one image
                "format": "json",
                "pretty": "true",
                "ihid": "NAC",  # Narrow Angle Camera
                "res": "100",  # Resolution in meters/pixel
            }
            
            # Search for images
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if not data or "results" not in data or not data["results"]:
                print(f"No images found for coordinates {lat}, {lon}")
                return False
            
            # Get the first result
            image_data = data["results"][0]
            image_url = f"{self.download_url}/{image_data['path']}"
            
            # Download the image
            response = requests.get(image_url, headers=self.headers)
            response.raise_for_status()
            
            # Check if we got a valid image
            if len(response.content) < 100000:  # If response is too small, it might be an error
                print(f"Warning: Received small response ({len(response.content)} bytes) for coordinates {lat}, {lon}")
                return False
            
            # Save the image
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Verify the saved file
            if not save_path.exists() or save_path.stat().st_size < 100000:
                print(f"Warning: Saved file is too small for coordinates {lat}, {lon}")
                return False
                
            return True
        except Exception as e:
            print(f"Error downloading image for coordinates {lat}, {lon}: {str(e)}")
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
    
    # Create directories
    raw_dir = Path(config['data']['raw_data_path'])
    images_dir = raw_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader
    downloader = LunarImageDownloader()
    
    # Get lunar coordinates
    coordinates = get_lunar_coordinates()
    
    # Download images
    for i, coord in enumerate(tqdm(coordinates, desc="Downloading images")):
        save_path = images_dir / f"crater_{i:04d}.jpg"
        
        if downloader.download_image(coord["lat"], coord["lon"], save_path):
            print(f"Successfully downloaded image {i} ({coord['lat']}, {coord['lon']})")
        else:
            print(f"Failed to download image {i} ({coord['lat']}, {coord['lon']})")
        
        # Add delay to avoid rate limiting
        time.sleep(2)  # Increased delay

if __name__ == '__main__':
    main() 