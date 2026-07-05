import os
import zipfile
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("downloader")

def download_breakhis():
    dataset_slug = "forderation/breakhis-400x"
    download_dir = os.path.join("data", "raw")
    zip_path = os.path.join(download_dir, "breakhis-400x.zip")
    extract_dir = os.path.join(download_dir, "images")

    os.makedirs(download_dir, exist_ok=True)

    # 1. Initialize Kaggle API
    logger.info("Initializing Kaggle API and authenticating...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"Failed to authenticate Kaggle API. Check your kaggle.json: {str(e)}")
        return False

    # 2. Download dataset
    if not os.path.exists(zip_path):
        logger.info(f"Downloading dataset '{dataset_slug}' using Kaggle API...")
        try:
            api.dataset_download_files(dataset_slug, path=download_dir, unzip=False, quiet=False)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download dataset from Kaggle: {str(e)}")
            return False
    else:
        logger.info("Dataset zip file already exists. Skipping download.")

    # 3. Extract zip file
    logger.info(f"Extracting dataset files to {extract_dir}...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info("Extraction completed successfully.")
    except Exception as e:
        logger.error(f"Failed to extract zip file: {str(e)}")
        return False

    # 4. Clean up zip file to save space
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Cleaned up temporary zip download.")
    except Exception as e:
        logger.warning(f"Failed to delete zip file: {str(e)}")

    logger.info("BreakHis 400X Image Dataset Ingestion Complete!")
    return True

if __name__ == "__main__":
    download_breakhis()
