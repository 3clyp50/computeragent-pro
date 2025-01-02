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
    status: str = Field(..., description="Status of the inference response")
    prediction: str = Field(..., description="Prediction result as a string")
    annotated_image: Optional[str] = Field(None, description="Base64 encoded image with bounding boxes")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "prediction": "[[100, 200, 300, 400]]",
                "annotated_image": "base64_encoded_string"
            }
        }
        validate_assignment = True