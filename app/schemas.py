from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from .config import settings

class ChatImage(BaseModel):
    content: str = Field(..., description="Base64 encoded image string.")

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[ChatImage]] = []

class ChatRequest(BaseModel):
    # Required fields
    stream: bool
    options: Optional[Dict[str, Any]] = {}
    format: Optional[str] = ""
    messages: List[ChatMessage]
    tools: Optional[List[Any]] = []

class InferenceResponse(BaseModel):
    status: str # Status of the inference response
    prediction: str # Prediction result as a string
    annotated_image: Optional[str] = None # Base64 encoded image with bounding boxes