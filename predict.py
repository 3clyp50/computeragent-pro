import requests
import urllib3
import json
# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_coordinates(image_path: str, prompt: str):
    url = "https://computeragent.pro/predict"
    
    try:
        with open(image_path, 'rb') as img_file:
            response = requests.post(
                url,
                files={'file': (image_path, img_file, 'image/jpeg')},
                data={'prompt': prompt},
                headers={'X-API-Key': 'computeragent_prod_key_2024'},
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract coordinates from prediction string
                prediction = eval(result['prediction'])[1]  # Get the second element (coordinates)
                return prediction if prediction else []
            else:
                print(f"Error: {response.status_code}")
                return []
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    coordinates = get_coordinates("./test_image_2.jpg", "Refresh Status")
    print(coordinates)
