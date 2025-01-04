import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "https://computeragent.pro/api/chat"

def get_coordinates(image_path: str, prompt: str, stream: bool = False):
    """Make API request and handle response."""
    try:
        # Read the image file
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()

        # Request payload
        payload = {
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": os.getenv('AI_AGENT_KEY')
        }

        response = requests.post(
            BASE_URL,
            headers=headers,
            json=payload,
            stream=stream
        )

        if response.status_code == 200:
            if stream:
                return response
            return response.json()['prediction']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    # For non-streaming
    print("Non-Streaming Request:")
    get_coordinates("./test_image_2.jpg", "Refresh Status", stream=False)

    # For streaming
    print("\nStreaming Request:")
    get_coordinates("./test_image_2.jpg", "Refresh Status", stream=True)