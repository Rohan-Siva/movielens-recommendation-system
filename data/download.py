import os
import zipfile
import requests
from io import BytesIO

def download_movielens(url, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    response = requests.get(url)
    response.raise_for_status()
    
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(save_dir)
    print(f"data in: {save_dir}")

if __name__ == "__main__":
    URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    SAVE_DIR = "data/raw"
    download_movielens(URL, SAVE_DIR)
