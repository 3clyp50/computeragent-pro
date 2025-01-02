import logging
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from typing import List, Optional, Union
from .config import settings
from .model import ModelInference, image_to_base64
from .schemas import (
    InferenceResponse, ChatRequest, ChatResponse, ChatMessage, 
    ChatChoice, ChatUsage, TextContent, ImageContent
)
from PIL import Image
import base64
import io
import time
import uuid
import json

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
        if not settings.AI_AGENT_KEYS:
            logger.error("AI_AGENT_KEYS are not set in production environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error."
            )
        if api_key_header not in settings.ai_agent_keys_list:
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

@app.post("/api/chat", response_model=Union[InferenceResponse, ChatResponse])
async def chat_endpoint(
    request: Request,
    chat_request: Optional[ChatRequest] = None,
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    return_annotated_image: Optional[bool] = Form(False),
    api_key: str = Depends(get_api_key)
):
    try:
        # Determine if this is a form-data request or JSON request
        content_type = request.headers.get("content-type", "").lower()
        is_form = "multipart/form-data" in content_type
        is_json = "application/json" in content_type
        
        if is_form and prompt and file:
            # Handle existing form-data format
            return await handle_form_request(prompt, file, return_annotated_image)
        elif is_json:
            # For JSON requests, ensure we have a valid chat request
            if not chat_request:
                # Try to parse the raw request body
                try:
                    body = await request.json()
                    chat_request = ChatRequest(**body)
                except Exception as e:
                    logger.error(f"Failed to parse chat request: {e}")
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid chat request format"
                    )
            
            if not chat_request.messages:
                raise HTTPException(
                    status_code=400,
                    detail="No messages found in chat request"
                )
            
            # Handle OpenAI-compatible format
            return await handle_chat_request(chat_request)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Send either form-data with prompt and file, or JSON in OpenAI chat format."
            )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error during prediction: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "Internal server error",
                "detail": error_msg if settings.ENVIRONMENT == "development" else None
            }
        )

async def handle_form_request(prompt: str, file: UploadFile, return_annotated_image: bool = False) -> InferenceResponse:
    """Handle the existing form-data format request"""
    logger.info(f"Received a form-data prediction request for file: {file.filename}")
    
    # Validate file size
    image_data = await file.read()
    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum size is 10MB."
        )

    # Open and validate image
    try:
        image = Image.open(io.BytesIO(image_data))
        logger.debug("Image opened successfully")
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
        object_ref, coordinates = model_inference.infer(image, prompt)
        
        if not coordinates:
            logger.warning("No coordinates found")
            return InferenceResponse(
                prediction="[]"  # Return empty list when no coordinates found
            )
        
        response = InferenceResponse(
            prediction=str(coordinates)  # Return just the coordinates
        )

        # Add annotated image if requested
        if return_annotated_image:
            annotated_image = image.copy()
            annotated_image = model_inference.draw_bounding_boxes(annotated_image, coordinates)
            response.annotated_image = image_to_base64(annotated_image)
        
        logger.info("Inference completed successfully")
        return response
        
    except Exception as inf_error:
        handle_inference_error(inf_error)

async def handle_chat_request(chat_request: ChatRequest) -> ChatResponse:
    """Handle OpenAI-compatible chat format request"""
    logger.info("Received a chat format prediction request")
    
    # Validate model field
    if not chat_request.model:
        raise HTTPException(
            status_code=400,
            detail="Model field is required in chat format"
        )
    
    # Extract the last user message
    last_message = next((msg for msg in reversed(chat_request.messages) if msg.role == "user"), None)
    if not last_message:
        raise HTTPException(
            status_code=400,
            detail="No user message found in the conversation"
        )

    # Extract prompt and image from the message
    prompt = None
    image_b64 = None
    
    if isinstance(last_message.content, list):
        for content in last_message.content:
            if isinstance(content, TextContent):
                prompt = content.text
                logger.debug(f"Extracted prompt: {prompt}")
            elif isinstance(content, ImageContent):
                image_url = content.image_url
                if image_url.startswith("data:image"):
                    try:
                        image_b64 = image_url.split(",", 1)[1]
                        logger.debug(f"Extracted image_b64: {image_b64[:30]}...")
                    except IndexError:
                        logger.error("Invalid image data URI format")
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image data URI format"
                        )
    else:
        prompt = last_message.content  # Handle string content case
        logger.debug(f"Extracted prompt from string content: {prompt}")
    
    # Validate both prompt and image are present
    if not prompt:
        logger.warning("No text content found in message")
        raise HTTPException(
            status_code=400,
            detail="Message must contain text content"
        )
    
    if not image_b64:
        logger.warning("No image content found in message")
        raise HTTPException(
            status_code=400,
            detail="Message must contain image content"
        )

    # Decode and process image
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        logger.debug("Image decoded and opened successfully")
    except Exception as img_error:
        logger.error(f"Image processing error: {img_error}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format or corrupted file: {str(img_error)}"
        )

    # Run inference
    try:
        logger.info("Starting inference")
        object_ref, coordinates = model_inference.infer(image, prompt)
        
        # Create response with properly formatted message content
        response = ChatResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            created=int(time.time()),
            model=chat_request.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=[
                            TextContent(
                                type="text",
                                text=str(coordinates)
                            )
                        ]
                    )
                )
            ],
            usage=ChatUsage(
                prompt_tokens=len(prompt) // 4,  # Rough estimation
                completion_tokens=len(str(coordinates)) // 4,
                total_tokens=(len(prompt) + len(str(coordinates))) // 4
            )
        )

        # Add annotated image if requested
        if chat_request.return_annotated_image:
            annotated_image = image.copy()
            annotated_image = model_inference.draw_bounding_boxes(annotated_image, coordinates)
            response.annotated_image = image_to_base64(annotated_image)
        
        return response
        
    except Exception as inf_error:
        logger.error(f"Inference error: {str(inf_error)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during inference."
        )

def handle_inference_error(inf_error: Exception):
    """Common error handling for inference errors"""
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