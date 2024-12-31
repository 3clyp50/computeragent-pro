import logging
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .config import settings
from .model import ModelInference
from .schemas import InferenceResponse
import base64
from PIL import Image
import io

# Initialize logger
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# API Key Security for production 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if settings.ENVIRONMENT == "production":
        if not settings.AI_AGENT_KEY:
            logger.error("AI_AGENT_KEY is not set in production environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error."
            )
        if api_key_header != settings.AI_AGENT_KEY:
            logger.warning(f"Unauthorized access attempt with key: {api_key_header}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key.",
                headers={"WWW-Authenticate": "API Key"},
            )
    return api_key_header

# Initialize FastAPI app
app = FastAPI(title="OS-Atlas Vision API")

# Configure CORS
if settings.ENVIRONMENT == "development":
    origins = ["*"]  # Allow all for local development
else:
    origins = [settings.DOMAIN_NAME]  # Restrict in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model inference
model_inference = ModelInference()

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "environment": settings.ENVIRONMENT}

@app.post("/predict", response_model=InferenceResponse)
async def predict(
    prompt: str = Form(...),
    file: UploadFile = File(...), 
    api_key: str = Depends(get_api_key)
):
    try:
        logger.info(f"Received a prediction request for file: {file.filename}")
        
        # Validate file size
        try:
            image_data = await file.read()
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=400,
                    detail="File size too large. Maximum size is 10MB."
                )
        except Exception as read_error:
            logger.error(f"Error reading file: {read_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(read_error)}"
            )

        # Open image
        try:
            image = Image.open(io.BytesIO(image_data))
            logger.debug(f"Image opened successfully")
        except Exception as img_error:
            logger.error(f"Image processing error: {img_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format or corrupted file: {str(img_error)}"
            )

        # Validate prompt length
        if len(prompt) > 500:
            logger.warning("Prompt too long, truncating")
            prompt = prompt[:500]
        
        # Run inference
        try:
            logger.info("Starting inference")
            coordinates = model_inference.infer(image, prompt)
            
            if not coordinates:
                logger.warning("No coordinates found")
                return InferenceResponse(
                    status="success",
                    prediction="[]"  # Return empty list when no coordinates found
                )
            
            logger.info("Inference completed successfully")
            return InferenceResponse(
                status="success",
                prediction=str(coordinates)  # Return just the coordinates
            )
            
        except Exception as inf_error:
            error_msg = str(inf_error)
            logger.error(f"Inference error: {error_msg}")
            
            if "CUDA out of memory" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="Server is currently overloaded. Please try again later."
                )
            elif "No valid tokens to decode" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Model generated invalid output. Please try again with a different prompt."
                )
            elif "Empty output after cleanup" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Model generated empty response. Please try again with a different prompt."
                )
            elif "No image inputs processed" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to process image input. Please ensure the image is valid."
                )
            elif "list index out of range" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Error processing model output. This is likely due to an unexpected model response format."
                )
            else:
                logger.error(f"Unhandled inference error: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail="An unexpected error occurred during inference."
                )
                
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error during prediction: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "detail": error_msg if settings.ENVIRONMENT == "development" else None
            }
        )


# --------------------------
# New endpoint for /api/chat
# --------------------------

# 1. Create Pydantic models to parse the request body.

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = []

class ChatRequest(BaseModel):
    model: str
    stream: bool
    options: Dict[str, Any]
    format: str
    messages: List[ChatMessage]
    tools: List[Any] = []

# 2. Create the /api/chat endpoint.
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    Example endpoint to handle a request like:
    
    curl -H 'Host: 127.0.0.1:11434' --compressed -H 'Connection: keep-alive' \
         -H 'content-type: application/json' -H 'accept: application/json' \
         -H 'user-agent: ollama-python/0.4.4 (x86_64 linux) Python/3.12.8' \
         -X POST http://127.0.0.1:11434/api/chat \
         -d '{"model": "moondream", "stream": true, "options": {}, "format": "", "messages": [{"role": "user", "content": "Describe the provided image in detail.", "images": ["<base64_encoded_image>"]}], "tools": []}'
    """
    try:
        # For demonstration, we’ll only use the FIRST user message if multiple are provided.
        if not chat_request.messages:
            raise HTTPException(
                status_code=400,
                detail="No messages provided in the request."
            )

        user_message = chat_request.messages[0]
        prompt = user_message.content.strip()

        # Check if there's at least one image in the user's message.
        if not user_message.images:
            raise HTTPException(
                status_code=400,
                detail="No images found in the request."
            )

        # Decode the first base64 image for inference
        base64_str = user_message.images[0]
        try:
            image_data = base64.b64decode(base64_str)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 encoding for image."
            )

        # Optional: check file size (e.g., 10MB limit)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image size too large. Maximum size is 10MB."
            )

        # Open the image from bytes
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Image open error: {e}")
            raise HTTPException(
                status_code=400,
                detail="Could not open the provided image. Possibly corrupted or invalid format."
            )

        # Truncate prompt if needed
        if len(prompt) > 500:
            logger.warning("Prompt too long, truncating")
            prompt = prompt[:500]

        # Run inference
        logger.info("Starting inference from /api/chat")
        try:
            inference_result = model_inference.infer(image, prompt)
        except Exception as inf_error:
            error_msg = str(inf_error)
            logger.error(f"Inference error: {error_msg}")
            # Handle known inference errors (similar to /predict)
            if "CUDA out of memory" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="Server is currently overloaded. Please try again later."
                )
            # ... handle other known errors if needed ...
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred during inference."
            )

        # Construct a response. For example, if your model output is textual:
        response_content = {
            "status": "success",
            "model": chat_request.model,
            "prompt": prompt,
            "inference_result": str(inference_result),
        }
        return JSONResponse(status_code=200, content=response_content)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error during /api/chat: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "detail": error_msg if settings.ENVIRONMENT == "development" else None
            }
        )

# Custom Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if settings.ENVIRONMENT == "development" else None
        }
    )
