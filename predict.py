import re
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

        if response.status_code == 200:
            json_response = response.json()
            print("Response:", json_response)
            if 'prediction' in json_response:
                print("\nRaw prediction string:", json_response['prediction'])

                # Extract both coordinate formats
                box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"
                box_match = re.search(box_pattern, json_response['prediction'])
                if box_match:
                    print("\nOriginal box coordinates:", box_match.group(1))

                # Extract final coordinates
                coord_str = json_response['prediction'].split('[[')[1].split(']]')[0]
                coords = [float(x.strip()) for x in coord_str.split(',')]
                print("\nFinal scaled coordinates:")
                print(f"x1: {coords[0]}, y1: {coords[1]}")
                print(f"x2: {coords[2]}, y2: {coords[3]}")

                # Validate coordinates
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    print("✓ Coordinates are properly ordered (x2 > x1, y2 > y1)")
                else:
                    print("⚠ Warning: Coordinates are not properly ordered")
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
