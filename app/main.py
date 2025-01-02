import logging
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .config import settings
from .model import ModelInference, draw_bounding_boxes, image_to_base64
from .schemas import ChatRequest, InferenceResponse
import base64
from PIL import Image
import io
from io import BytesIO


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


@app.post("/api/chat", response_model=InferenceResponse)
async def chat_endpoint(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    Handles chat requests by replacing the /predict endpoint.
    Accepts JSON payload with base64-encoded images.
    """
    try:
        if not chat_request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided in the request."
            )

        # Assuming the first message contains the user prompt and image
        user_message = chat_request.messages[0]
        prompt = user_message.content.strip()

        if not user_message.images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No images found in the request."
            )

        # Decode the first base64 image
        base64_str = user_message.images[0].content
        try:
            image_data = base64.b64decode(base64_str)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 encoding for image."
            )

        # Check file size (e.g., 10MB limit)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image size too large. Maximum size is 10MB."
            )

        # Open the image
        try:
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            logger.error(f"Image open error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not open the provided image. Possibly corrupted or invalid format."
            )

        # Truncate prompt if necessary
        if len(prompt) > 500:
            logger.warning("Prompt too long, truncating")
            prompt = prompt[:500]

        if chat_request.stream:
            logger.info("Starting streaming inference from /api/chat")

            token_generator = model_inference.stream_infer(image, prompt)
            return StreamingResponse(token_generator, media_type="text/plain")
        else:
            logger.info("Starting normal (non-streaming) inference from /api/chat")
            object_ref, coordinates = model_inference.infer(image, prompt)

            if not coordinates:
                logger.warning("No coordinates found")
                return InferenceResponse(
                    status="success",
                    prediction="[]",  # Return empty list when no coordinates found
                    annotated_image=None
                )

            # Optionally, you can return an annotated image
            annotated_image = image_to_base64(draw_bounding_boxes(image.copy(), coordinates))

            return InferenceResponse(
                status="success",
                prediction=str(coordinates),
                annotated_image=annotated_image
            )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error during /api/chat: {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
