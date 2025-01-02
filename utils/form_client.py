import logging
import requests
from typing import Optional, Dict, Union
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormDataClient:
    """Client for interacting with ComputerAgent Pro Vision API using form-data format."""
    
    def __init__(self, api_key: str, base_url: str = "https://computeragent.pro"):
        """
        Initialize the form-data client.
        
        Args:
            api_key (str): Your API key for authentication
            base_url (str): Base URL of the API (default: https://computeragent.pro)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json"
        }

    def locate_element(self, 
                      image_path: Union[str, Path, Image.Image],
                      prompt: str,
                      return_annotated_image: bool = False) -> Dict:
        """
        Locate UI element using form-data format.
        
        Args:
            image_path: Path to image file or PIL Image object
            prompt: Description of the element to locate
            return_annotated_image: Whether to return the annotated image
            
        Returns:
            Dict containing the API response with coordinates and optionally annotated image
            
        Raises:
            ValueError: If image format is unsupported
            requests.exceptions.RequestException: If API request fails
        """
        url = f"{self.base_url}/api/chat"
        
        try:
            # Prepare the image file
            if isinstance(image_path, (str, Path)):
                files = {'file': ('image.png', open(image_path, 'rb'), 'image/png')}
            elif isinstance(image_path, Image.Image):
                buffered = BytesIO()
                image_path.save(buffered, format='PNG')
                buffered.seek(0)
                files = {'file': ('image.png', buffered, 'image/png')}
            else:
                raise ValueError("Unsupported image type")

            # Prepare form data
            data = {
                'prompt': prompt,
                'return_annotated_image': str(return_annotated_image).lower()
            }
            
            # Make request
            response = requests.post(
                url,
                headers=self.headers,
                files=files,
                data=data
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request: {e}")
            raise

def main():
    """Example usage of the form-data client."""
    # Initialize client
    client = FormDataClient(api_key="your-api-key")
    
    # Example 1: Just coordinates
    try:
        result = client.locate_element(
            image_path="screenshot.png",
            prompt="Find the submit button"
        )
        print("Form-data result (coordinates):", result)
    except Exception as e:
        print(f"Error in form-data request: {e}")
    
    # Example 2: With annotated image
    try:
        result_with_image = client.locate_element(
            image_path="screenshot.png",
            prompt="Find the submit button",
            return_annotated_image=True
        )
        print("Form-data result (with image):", result_with_image)
    except Exception as e:
        print(f"Error in form-data request with image: {e}")

if __name__ == "__main__":
    main() 