# ComputerAgent Pro Vision API Documentation

## Overview

ComputerAgent Pro Vision API provides advanced image analysis capabilities for extracting precise pixel coordinates of UI elements based on textual descriptions. The API supports two request formats:

1. Form-data format (multipart/form-data)
2. JSON format (OpenAI-compatible)

## Authentication

All requests require API key authentication using the `X-API-Key` header.

```bash
X-API-Key: your-api-key
```

## API Endpoints

### Element Location (Form-Data Format)

#### Request

```bash
# Just coordinates
curl -X POST https://computeragent.pro/api/chat \
  -H "X-API-Key: your-api-key" \
  -F "prompt=Find the submit button" \
  -F "file=@screenshot.png"

# With annotated image
curl -X POST https://computeragent.pro/api/chat \
  -H "X-API-Key: your-api-key" \
  -F "prompt=Find the submit button" \
  -F "file=@screenshot.png" \
  -F "return_annotated_image=true"
```

#### Python Example (Form-Data)

```python
from form_client import FormDataClient

# Initialize client
client = FormDataClient(api_key="your-api-key")

# Locate element (just coordinates)
result = client.locate_element(
    image_path="screenshot.png",
    prompt="Find the submit button"
)

# Locate element (with annotated image)
result_with_image = client.locate_element(
    image_path="screenshot.png",
    prompt="Find the submit button",
    return_annotated_image=True
)
```

### Element Location (JSON Format)

#### Request

```bash
# Just coordinates
curl -X POST https://computeragent.pro/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Find the submit button"},
        {"type": "image_url", "image_url": "data:image/png;base64,..."}
      ]
    }]
  }'

# Just coordinates with inline base64 image encoding
curl -X POST https://computeragent.pro/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: computeragent_prod_key_2024" \
  -d "{
    \"model\": \"OS-Copilot/OS-Atlas-Base-7B\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {
          \"type\": \"text\",
          \"text\": \"Refresh Status\"
        },
        {
          \"type\": \"image_url\",
          \"image_url\": \"data:image/jpeg;base64,$(base64 -w 0 ./test_images/test_image_2.jpg)\"
        }
      ]
    }],
  }"

# If you are on macOS, the base64 command does not support the -w 0 option, so you can use this instead:
        {
          \"type\": \"image_url\",
          \"image_url\": \"data:image/jpeg;base64,$(base64 ./test_images/test_image_2.jpg | tr -d '\n')\"
        }

# With annotated image
curl -X POST https://computeragent.pro/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Find the submit button"},
        {"type": "image_url", "image_url": "data:image/png;base64,..."}
      ]
    }],
    "return_annotated_image": true
  }'
```

#### Python Example (JSON)

```python
from json_client import JSONClient

# Initialize client
client = JSONClient(api_key="your-api-key")

# Locate element (just coordinates)
result = client.locate_element(
    image_path="screenshot.png",
    prompt="Find the submit button"
)

# Locate element (with annotated image)
result_with_image = client.locate_element(
    image_path="screenshot.png",
    prompt="Find the submit button",
    return_annotated_image=True
)
```

## Response Format

### Success Response (Coordinates Only)
```json
{
    "prediction": "[[x1, y1, x2, y2]]"
}
```

### Success Response (With Annotated Image)
```json
{
    "prediction": "[[x1, y1, x2, y2]]",
    "annotated_image": "base64_encoded_image_data"
}
```

### Error Response
```json
{
    "detail": "Error message"
}
```

## Rate Limits and Constraints

- Maximum image size: 10MB
- Supported image formats: JPEG, PNG
- Maximum prompt length: 500 characters

## Best Practices

1. **Prompt Engineering**
   - Be specific about the UI element you want to locate
   - Use clear, concise language
   - Example prompts:
     - "Find the login button"
     - "Locate the refresh status icon"
     - "Where is the submit button?"

2. **Image Preparation**
   - Use clear, high-quality screenshots
   - Ensure the target element is visible
   - Optimize image size when possible

3. **Error Handling**
   - Implement proper error handling in your code
   - Validate input parameters before making requests

## Integration Examples

### UI Automation Framework
```python
from json_client import JSONClient
import pyautogui

class UIAutomation:
    def __init__(self, api_key: str):
        self.vision_client = JSONClient(api_key)
        
    def find_and_click(self, screenshot_path: str, element_description: str):
        result = self.vision_client.locate_element(
            image_path=screenshot_path,
            prompt=element_description
        )
        coordinates = eval(result["prediction"])[0]
        center_x = (coordinates[0] + coordinates[2]) / 2
        center_y = (coordinates[1] + coordinates[3]) / 2
        pyautogui.click(center_x, center_y)
```
