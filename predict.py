import requests
import json
import os
import base64
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

BASE_URL = "https://computeragent.pro/api/chat"

def is_url(string):
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_coordinates(image_input: str, prompt: str):
    """
    Make API request and handle response.
    
    Args:
        image_input: Either a local file path or a URL to an image
        prompt: The prompt text
    """
    try:
        # Verify API key
        api_key = os.getenv('AI_AGENT_KEY')
        if not api_key:
            print("Error: AI_AGENT_KEY not found in environment variables")
            return None

        print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")

        # Format the prompt exactly like the working example
        formatted_prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{prompt}\" (with bbox)?"

        # Request payload
        payload = {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_input,
                            "is_url": is_url(image_input)
                        },
                        {
                            "type": "text",
                            "text": formatted_prompt
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": os.getenv('AI_AGENT_KEY')
        }

        print(f"Making request to: {BASE_URL}")
        print(f"Headers: {headers}")
        print(f"Payload size: {len(str(payload))} bytes")
        
        response = requests.post(
            BASE_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            try:
                print(f"Raw response: {response.text}")
                if not response.text.strip():
                    print("Warning: Received empty response from server")
                    return None
                return response.json()['prediction']
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON response. Raw response: {response.text}")
                print(f"JSON decode error: {e}")
                return None
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
            return None

    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage with local file
    print("Making request with local file:")
    get_coordinates("./test_image_2.jpg", "Refresh Status")

    # Example usage with URL
    print("\nMaking request with URL:")
    get_coordinates("https://example.com/image.jpg", "Refresh Status")