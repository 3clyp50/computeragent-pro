from pydantic import BaseModel
from typing import Optional

class InferenceRequest(BaseModel):
    prompt: str

class InferenceResponse(BaseModel):
    status: str
    prediction: str
    annotated_image: Optional[str] = None  # Base64 encoded image with bounding boxes
