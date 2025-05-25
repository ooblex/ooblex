#!/usr/bin/env python3
"""
Download OpenCV Haar Cascades for face detection
"""
import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CASCADES = {
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml",
    "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
    "haarcascade_smile.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml",
    "haarcascade_upperbody.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_upperbody.xml",
    "haarcascade_fullbody.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml",
}


def download_cascades(output_dir="cascades"):
    """Download all cascade files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, url in CASCADES.items():
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            logger.info(f"Cascade {filename} already exists, skipping")
            continue
            
        logger.info(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"Successfully downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            
    logger.info("All cascades downloaded successfully")


if __name__ == "__main__":
    download_cascades()