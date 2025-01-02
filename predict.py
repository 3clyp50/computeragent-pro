import requests
import json
import base64
import ast
import os

def api_key():
    return os.getenv('AI_AGENT_KEY', 'Missing API key')

def get_coordinates(image_path: str, prompt: str, stream: bool = False):
    url = os.getenv('BASE_URL')  # Updated endpoint

    try:
        # Read and encode the image in base64
        with open(image_path, 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        # Construct the JSON payload
        payload = {
            "stream": stream,
            "options": {},  # Add any specific options if needed
            "format": "",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        {
                            "content": encoded_image
                        }
                    ]
                }
            ],
            "tools": []  # Add any tools if needed
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

        if stream:
            # Handle streaming response
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                verify=True
            )

            if response.status_code == 200:
                # Assuming streaming returns plain text
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        print(chunk.decode('utf-8'))
            else:
                print(f"Error: {response.status_code} - {response.text}")
        else:
            # Handle non-streaming response
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                verify=True # Verify SSL
            )

            if response.status_code == 200:
                result = response.json()
                # Extract coordinates from prediction string
                # Assuming prediction is a string representation of a list: "[(x1, y1, x2, y2)]"
                try:
                    prediction = ast.literal_eval(result['prediction'])
                    coordinates = prediction[0] if prediction else []
                except Exception as e:
                    print(f"Error parsing prediction: {e}")
                    coordinates = []
                print(f"Coordinates: {coordinates}")
            else:
                print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage:
    # For non-streaming
    print("Non-Streaming Request:")
    get_coordinates("./test_image_2.jpg", "Refresh Status", stream=False)

    # For streaming
    print("\nStreaming Request:")
    get_coordinates("./test_image_2.jpg", "Refresh Status", stream=True)