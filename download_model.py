from huggingface_hub import snapshot_download
import os
from app.config import settings

def download_model():
    print(f"Downloading model {settings.MODEL_NAME}...")
    
    # Get absolute path for cache directory
    if os.path.isabs(settings.HF_HOME):
        cache_dir = settings.HF_HOME
    else:
        cache_dir = os.path.join(os.path.dirname(__file__), settings.HF_HOME)
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download model files
    local_dir = snapshot_download(
        repo_id=settings.MODEL_NAME,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, settings.MODEL_NAME.split('/')[-1]),
        local_dir_use_symlinks=False  # Actual files instead of symlinks for Docker
    )
    
    print(f"Model downloaded successfully to: {local_dir}")
    print("\nModel cache is configured in .env:")
    print(f"HF_HOME={settings.HF_HOME}")
    print("\nAnd mounted in docker-compose.yml:")
    print("volumes:")
    print("  - ./model_cache:/app/model_cache")
    
    return local_dir

if __name__ == "__main__":
    download_model()
