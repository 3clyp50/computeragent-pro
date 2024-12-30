# ComputerAgent Pro Vision API Documentation

## Overview
The ComputerAgent Pro Vision API provides advanced image analysis capabilities, specifically designed for extracting precise pixel coordinates of UI elements based on textual descriptions using [OS-Copilot's OS-Atlas-Base-7B Model](https://huggingface.co/OS-Copilot/OS-Atlas-Base-7B)

This API is capable of identifying and locating UI components such as buttons, text fields, and icons within images, making it a powerful tool for automating UI testing and enhancing user experience.

## Endpoint
- **URL**: `https://computeragent.pro/predict`
- **Method**: `POST`

## Authentication
- **Type**: API Key Authentication
- **Header**: `X-API-Key`
- **Required**: Yes (in production environment)

## Request Parameters

### Multipart Form Data
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `file` | File | Yes | Screenshot or UI image to analyze | - |
| `prompt` | String | No | Description of the element to locate | "Describe this image" |

### cURL Request
```bash
curl -X POST https://computeragent.pro/predict \
  -H "X-API-Key: your_computeragent_api_key" \
  -F "file=@screenshot.jpg" \
  -F "prompt=Find the login button"
```

### Python Request
```python
import requests

url = "https://computeragent.pro/predict"
headers = {
    "X-API-Key": "your_api_key_here"
}

with open("screenshot.jpg", "rb") as image_file:
    files = {"file": image_file}
    data = {"prompt": "Find the login button"}
    
    response = requests.post(url, files=files, data=data, headers=headers)
    result = response.json()
```

## Response Format
```json
{
  "status": "success",
  "prediction": "[x1, y1, x2, y2]"
}
```

### Response Fields
- `status`: Request processing status ("success" or "error")
- `prediction`: 
  - Array of 4 float values representing bounding box coordinates
  - Format: `[x_min, y_min, x_max, y_max]`
  - Coordinates are scaled to the original image dimensions

## Coordinate System
- `x_min, y_min`: Top-left corner of the bounding box
- `x_max, y_max`: Bottom-right corner of the bounding box
- Coordinates are in pixels, relative to the original image size

## Possible Responses
1. Successful Coordinate Detection
```json
{
  "status": "success", 
  "prediction": "[627.6, 692.17, 899.4, 757.25]"
}
```

2. No Coordinates Found
```json
{
  "status": "success", 
  "prediction": "[]"
}
```

3. Error Response
```json
{
  "status": "error",
  "message": "Internal server error",
  "detail": "Optional error details (in development mode)"
}
```

## Prompt Engineering Tips
- Be specific about the UI element you want to locate
- Use clear, concise language
- Examples:
  - "Find the login button"
  - "Locate the refresh status icon"
  - "Where is the submit button?"

## Error Handling
- 400: Invalid file or request parameters
- 401: Invalid API key
- 500: Server-side processing errors
- 503: Server overload

## Notes
- Maximum image size: 10MB
- Supported image formats: JPEG, PNG, etc.
- Coordinate precision may vary based on image/task complexity

## Practical Example: Automated UI Testing

### Scenario
Imagine you're building an automated testing framework for a web application. You want to programmatically interact with specific UI elements across different screen sizes and layouts.

### Sample Implementation in AI Framework
```python
import requests
import pyautogui

def find_and_click_element(screenshot_path, element_description):
    """
    Find UI element coordinates and perform an action
    
    Args:
        screenshot_path (str): Path to the screenshot
        element_description (str): Textual description of the element
    
    Returns:
        dict: Coordinate information or None if element not found
    """
    url = "https://computeragent.pro/predict"
    headers = {
        "X-API-Key": "your_computeragent_api_key"
    }
    
    with open(screenshot_path, "rb") as image_file:
        files = {"file": image_file}
        data = {"prompt": element_description}
        
        response = requests.post(url, files=files, data=data, headers=headers)
        result = response.json()
    
    if result['status'] == 'success' and result['prediction']:
        coords = result['prediction']
        # Calculate center of bounding box
        center_x = (coords[0] + coords[2]) / 2
        center_y = (coords[1] + coords[3]) / 2
        
        # Perform click action
        pyautogui.click(center_x, center_y)
        return {
            "coordinates": coords,
            "center": (center_x, center_y)
        }
    
    return None

# Example usage
screenshot = "dashboard_screenshot.png"
login_button_info = find_and_click_element(
    screenshot, 
    "Find the login button in this dashboard"
)

if login_button_info:
    print(f"Login button found at: {login_button_info['coordinates']}")
    print(f"Clicked at center point: {login_button_info['center']}")
else:
    print("Could not locate login button")
```

### Key Takeaways
- The API enables dynamic, prompt-based UI element location
- Works across different UI layouts and screen sizes
- Can be integrated into testing, automation, and accessibility tools
- Supports complex, natural language element descriptions
