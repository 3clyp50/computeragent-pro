import logging
from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .model import ModelInference, draw_bounding_boxes, image_to_base64
from .schemas import ChatRequest, InferenceResponse
from PIL import Image
from io import BytesIO
import base64

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
    origins = ["*"]
else:
    origins = [settings.DOMAIN_NAME]

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
    Handles chat requests with OpenAI-style message format.
    Accepts JSON payload with text and base64-encoded images.
    """
    try:
        if not chat_request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided in the request."
            )

        # Get the last user message
        user_message = next(
            (msg for msg in reversed(chat_request.messages) if msg.role == "user"),
            None
        )
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in the request."
            )

        # Extract text and image from the message content
        text_content = next(
            (content for content in user_message.content if content.type == "text"),
            None
        )
        image_content = next(
            (content for content in user_message.content if content.type == "image"),
            None
        )

        if not text_content or not image_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message must contain both text and image content."
            )

        prompt = text_content.text.strip()
        image_path = image_content.image

        # Handle image input
        try:
            if image_content.is_url:
                # For URLs, the image is already base64 encoded by the schema validator
                image_data = base64.b64decode(image_path.split('base64,')[1])
                image = Image.open(BytesIO(image_data))
            else:
                # For local files
                image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not process the provided image. Either the file is invalid/missing or the URL is inaccessible."
            )

        # Truncate prompt if necessary
        if len(prompt) > 500:
            logger.warning("Prompt too long, truncating")
            prompt = prompt[:500]

        logger.info("Starting inference from /api/chat")
        try:
            object_ref, coordinates = model_inference.infer(image, prompt)
            logger.info(f"Inference completed. Object ref: {object_ref}, Coordinates: {coordinates}")

            if not coordinates:
                logger.warning("No coordinates found")
                response_dict = {
                    "prediction": "[]",
                    "annotated_image": None
                }
            else:
                # Create annotated image
                annotated_image = image_to_base64(draw_bounding_boxes(image.copy(), coordinates))
                response_dict = {
                    "prediction": str(coordinates),
                    "annotated_image": annotated_image
                }
                
            logger.info(f"Preparing response with data: {response_dict}")
            return InferenceResponse(**response_dict)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error during /api/chat: {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
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
            "message": "Internal server error",
            "detail": str(exc) if settings.ENVIRONMENT == "development" else None
        }
    )
