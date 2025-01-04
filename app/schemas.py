from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union, Literal
from .config import settings

class ImageUrl(BaseModel):
    url: str = Field(..., description="Base64 image URL in format data:image/jpeg;base64,{base64_string}")

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")

class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl

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