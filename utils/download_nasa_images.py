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

class NASAImageDownloader:
    def __init__(self, api_key: str = "DEMO_KEY"):
        """Initialize the NASA image downloader."""
        self.api_key = api_key
        self.search_url = "https://images-api.nasa.gov/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def search_images(self, query: str, year_start: int = 2009, year_end: int = 2024) -> List[Dict]:
        """
        Search for images in NASA's image library.
        
        Args:
            query (str): Search query
            year_start (int): Start year for search
            year_end (int): End year for search
            
        Returns:
            List[Dict]: List of image metadata
        """
        try:
            params = {
                "q": query,
                "media_type": "image",
                "year_start": year_start,
                "year_end": year_end,
                "keywords": "moon,crater,lunar",
            }
            
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "collection" not in data or "items" not in data["collection"]:
                return []
                
            return data["collection"]["items"]
        except Exception as e:
            print(f"Error searching for images: {str(e)}")
            return []
    
    def download_image(self, item: Dict, save_path: Path) -> bool:
        """
        Download an image from NASA's image library.
        
        Args:
            item (Dict): Image metadata
            save_path (Path): Path to save the image
            
        Returns:
            bool: True if download was successful
        """
        try:
            # Get the image URL from the metadata
            if "links" not in item:
                return False
                
            image_url = None
            for link in item["links"]:
                if link["rel"] == "preview" and link["render"] == "image":
                    image_url = link["href"]
                    break
            
            if not image_url:
                return False
            
            # Download the image
            response = requests.get(image_url, headers=self.headers)
            response.raise_for_status()
            
            # Check if we got a valid image
            if len(response.content) < 10000:  # If response is too small, it might be an error
                print(f"Warning: Received small response ({len(response.content)} bytes)")
                return False
            
            # Save the image
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Verify the saved file
            if not save_path.exists() or save_path.stat().st_size < 10000:
                print(f"Warning: Saved file is too small")
                return False
                
            return True
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return False

def main():
    """Main function to download NASA crater images."""
    # Load configuration
    config = load_config('config.yaml')
    
    # Create directories
    raw_dir = Path(config['data']['raw_data_path'])
    images_dir = raw_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader with NASA API key from config
    api_key = config.get('api', {}).get('nasa_api_key', 'DEMO_KEY')
    downloader = NASAImageDownloader(api_key)
    
    # Search for crater images
    print("Searching for lunar crater images...")
    items = downloader.search_images("lunar crater", year_start=2009)
    
    if not items:
        print("No images found!")
        return
        
    print(f"Found {len(items)} images")
    
    # Download images
    for i, item in enumerate(tqdm(items[:10], desc="Downloading images")):
        save_path = images_dir / f"crater_{i:04d}.jpg"
        
        if downloader.download_image(item, save_path):
            print(f"Successfully downloaded image {i}")
        else:
            print(f"Failed to download image {i}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)

if __name__ == '__main__':
    main() 