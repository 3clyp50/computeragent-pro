from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
from .config import settings

class InferenceRequest(BaseModel):
    prompt: str
    return_annotated_image: bool = Field(default=False, description="Whether to return the image with drawn bounding boxes")

class InferenceResponse(BaseModel):
    prediction: str
    annotated_image: Optional[str] = None  # Base64 encoded image with bounding boxes

# OpenAI-compatible schemas
class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: str = Field(..., description="Base64 encoded image with data URI scheme")

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

MessageContent = List[Union[TextContent, ImageContent]]

class ChatMessage(BaseModel):
    role: str
    content: MessageContent

class ChatRequest(BaseModel):
    model: str = Field(default=settings.MODEL_NAME)
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=128)
    temperature: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default=False)
    return_annotated_image: bool = Field(default=False, description="Whether to return the image with drawn bounding boxes")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "OS-Copilot/OS-Atlas-Base-7B",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": "data:image/jpeg;base64,..."
                            }
                        ]
                    }
                ],
                "return_annotated_image": True
            }
        }

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage
    annotated_image: Optional[str] = None  # Base64 encoded image with bounding boxes
