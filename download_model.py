from huggingface_hub import snapshot_download
import os
from app.config import settings

def download_model():
    print(f"Downloading model {settings.MODEL_NAME}...")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download model files
    local_dir = snapshot_download(
        repo_id=settings.MODEL_NAME,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, settings.MODEL_NAME.split('/')[-1]),
        local_dir_use_symlinks=False  # Actual files instead of symlinks for Docker
    )
    
    print(f"Model downloaded successfully to: {local_dir}")
    print("\nTo use the cached model:")
    print("1. Update your .env file to include:")
    print(f"TRANSFORMERS_CACHE={cache_dir}")
    print("\n2. Or set the environment variable:")
    print(f"export TRANSFORMERS_CACHE={cache_dir}")
    
    return local_dir

if __name__ == "__main__":
    download_model()
