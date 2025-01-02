# test_predict.py

import pytest
import requests
import base64
import json
import ast
import os
from pathlib import Path

@pytest.fixture
def api_key():
    key = os.getenv('AI_AGENT_KEY', 'Missing API key')
    if key == 'Missing API key':
        pytest.skip("API key not found in environment variables")
    return key

@pytest.fixture
def api_url():
    url = os.getenv('BASE_URL')
    if not url:
        pytest.skip("BASE_URL not found in environment variables")
    return url

def encode_image(image_path):
    try:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        pytest.fail(f"Failed to encode image: {str(e)}")

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

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {str(e)}")

    try:
        result = response.json()
        prediction = ast.literal_eval(result['prediction'])
        assert isinstance(prediction, list), "Prediction should be a list"
        print(f"Coordinates: {prediction}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        pytest.fail(f"Failed to parse response: {str(e)}")

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

    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Streaming request failed: {str(e)}")

    try:
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                print(chunk)
    except Exception as e:
        pytest.fail(f"Failed to process stream: {str(e)}")