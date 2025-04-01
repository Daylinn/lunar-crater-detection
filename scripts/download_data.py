"""
My lunar crater dataset download script.
This script helps us get our dataset from cloud storage and set it up for training.
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

def download_data():
    """
    Download our lunar crater dataset from cloud storage.
    This will get us all the images and labels we need for training.
    """
    # Create our data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # URL where our dataset is hosted
    # TODO: Replace this with the actual URL where you're hosting the dataset
    dataset_url = "YOUR_DATASET_URL_HERE"
    
    try:
        print("Downloading dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()  # Make sure we got a good response
        
        # Save the zip file
        zip_path = data_dir / 'dataset.zip'
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting dataset...")
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up the zip file
        zip_path.unlink()
        
        print("Dataset downloaded and extracted successfully!")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nIf you're having trouble downloading automatically, you can:")
        print("1. Download the dataset manually from:", dataset_url)
        print("2. Place the downloaded zip file in the 'data' directory")
        print("3. Extract the zip file")
        print("4. Run this script again")

if __name__ == '__main__':
    download_data() 