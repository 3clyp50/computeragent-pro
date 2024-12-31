import requests
import urllib3
# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_prediction(image_path: str, prompt: str = "Refresh Status"):
    # Format prompt to ask for coordinates
    prompt = f'In this UI screenshot, what is the position of the element corresponding to the command "{prompt}" (with bbox)?'
    url = "https://computeragent.pro/predict"

    try:
        # Open and prepare the image file
        with open(image_path, 'rb') as img_file:
            files = {
                'file': (image_path, img_file, 'image/jpeg')
            }
            data = {
                'prompt': prompt
            }
            headers = {
                'X-API-Key': 'computeragent_prod_key_2024'
            }

            print(f"Sending request to: {url}")
            print(f"Prompt: {prompt}")
            response = requests.post(
                url,
                files=files,
                data=data,
                headers=headers,
                verify=False,
                allow_redirects=False
            )
        
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Raw response content: {response.content.decode('utf-8')}")

        if response.status_code != 200:
            print(f"Full response: {response.text}")
            if 'location' in response.headers:
                print(f"Redirect location: {response.headers['location']}")
        else:
            try:
                json_response = response.json()
                print("Response:", json_response)
                return json_response
            except Exception as json_error:
                print(f"JSON parsing error: {str(json_error)}")
                return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Test with an image
    test_prediction("./test_image_2.jpg", "Refresh Status")
