import logging
import requests
import base64
from typing import Optional, Dict, Union
from pathlib import Path
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONClient:
    """Client for interacting with ComputerAgent Pro Vision API using JSON format."""
    
    def __init__(self, api_key: str, base_url: str = "https://computeragent.pro"):
        """
        Initialize the JSON client.
        
        Args:
            api_key (str): Your API key for authentication
            base_url (str): Base URL of the API (default: https://computeragent.pro)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def _image_to_base64(self, image_path: Union[str, Path, Image.Image]) -> str:
        """Convert image to base64 string."""
        try:
            if isinstance(image_path, (str, Path)):
                with Image.open(image_path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format=img.format or "PNG")
                    return f"data:image/{img.format.lower() if img.format else 'png'};base64," + \
                           base64.b64encode(buffered.getvalue()).decode()
            elif isinstance(image_path, Image.Image):
                buffered = BytesIO()
                image_path.save(buffered, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
            else:
                raise ValueError("Unsupported image type")
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise

    def locate_element(self,
                      image_path: Union[str, Path, Image.Image],
                      prompt: str,
                      return_annotated_image: bool = False,
                      model: str = "OS-Copilot/OS-Atlas-Base-7B") -> Dict:
        """
        Locate UI element using JSON format (OpenAI-compatible).
        
        Args:
            image_path: Path to image file or PIL Image object
            prompt: Description of the element to locate
            return_annotated_image: Whether to return the annotated image
            model: Model to use for inference
            
        Returns:
            Dict containing the API response with coordinates and optionally annotated image
            
        Raises:
            ValueError: If image format is unsupported
            requests.exceptions.RequestException: If API request fails
        """
        url = f"{self.base_url}/api/chat"
        
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image_path)
            
            # Prepare JSON payload
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image_b64}
                    ]
                }],
                "return_annotated_image": return_annotated_image
            }
            
            # Make request
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request: {e}")
            raise

def main():
    """Example usage of the JSON client."""
    # Initialize client
    client = JSONClient(api_key="your-api-key")
    
    # Example 1: Just coordinates
    try:
        result = client.locate_element(
            image_path="screenshot.png",
            prompt="Find the submit button"
        )
        print("JSON result (coordinates):", result)
    except Exception as e:
        print(f"Error in JSON request: {e}")
    
    # Example 2: With annotated image
    try:
        result_with_image = client.locate_element(
            image_path="screenshot.png",
            prompt="Find the submit button",
            return_annotated_image=True
        )
        print("JSON result (with image):", result_with_image)
    except Exception as e:
        print(f"Error in JSON request with image: {e}")

if __name__ == "__main__":
    main() 