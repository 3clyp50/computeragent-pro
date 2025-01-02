# test_predict.py

import pytest
import requests
import base64
import json
import ast
import os

@pytest.fixture
def api_key():
    return os.getenv('AI_AGENT_KEY', 'Missing API key')

@pytest.fixture
def api_url():
    return os.getenv('BASE_URL')

def encode_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_non_streaming(api_url, api_key):
    payload = {
        "stream": False,
        "options": {},
        "format": "",
        "messages": [
            {
                "role": "user",
                "content": "Refresh Status",
                "images": [
                    {
                        "content": encode_image("./test_image_2.jpg")
                    }
                ]
            }
        ],
        "tools": []
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    result = response.json()
    prediction = ast.literal_eval(result['prediction'])
    assert isinstance(prediction, list), "Prediction should be a list"
    print(f"Coordinates: {prediction}")

def test_streaming(api_url, api_key):
    payload = {
        "stream": True,
        "options": {},
        "format": "",
        "messages": [
            {
                "role": "user",
                "content": "Refresh Status",
                "images": [
                    {
                        "content": encode_image("./test_image_2.jpg")
                    }
                ]
            }
        ],
        "tools": []
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=True)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            print(chunk.decode('utf-8'))