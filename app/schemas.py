from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Union, Literal
from .config import settings
import requests
import base64
from io import BytesIO

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    image: str = Field(..., description="Path to the image file or URL")
    is_url: bool = Field(default=False, description="Whether the image field is a URL")

    @validator('image')
    def validate_and_process_image(cls, v, values):
        if values.get('is_url', False):
            try:
                response = requests.get(v)
                response.raise_for_status()
                # Convert to base64
                image_data = base64.b64encode(response.content).decode('utf-8')
                return f"data:image/jpeg;base64,{image_data}"
            except Exception as e:
                raise ValueError(f"Failed to fetch or process image URL: {str(e)}")
        return v

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant')")
    content: List[Union[TextContent, ImageContent]] = Field(..., description="List of content items (text and/or image)")

class ChatRequest(BaseModel):
    stream: bool = Field(..., description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Additional options for the chat")
    format: Optional[str] = Field(default="", description="Response format preference")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")

class InferenceResponse(BaseModel):
    prediction: str = Field(..., description="Prediction result as coordinates string")
    annotated_image: Optional[str] = Field(
        default=None, 
        description="Base64 encoded image with bounding boxes"
    )