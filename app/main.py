import logging
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.throttling import ThrottlingMiddleware
from starlette.requests import Request
from .config import settings
from .model import ModelInference
from .schemas import InferenceResponse
from .utils import process_vision_info
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

def get_api_key(api_key_header: str = Depends(api_key_header)):
    if settings.ENVIRONMENT == "production":
        if not settings.AI_AGENT_KEY:
            logger.error("AI_AGENT_KEY is not set in production environment.")
            # AI_AGENT_KEY is set in the .env file
            # It will be used internally by us for security purposes
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

# Configure CORS (dev environment)
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

app.add_middleware(
    ThrottlingMiddleware, 
    rate_limit=100,  # requests
    time_window=60   # seconds
)

# Initialize model inference
model_inference = ModelInference()

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "environment": settings.ENVIRONMENT}

from functools import wraps

def require_api_key(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        api_key = await get_api_key()
        return await func(*args, **kwargs)
    return wrapper

@app.post("/predict", response_model=InferenceResponse, dependencies=[Depends(get_api_key)])
@require_api_key
async def predict(file: UploadFile = File(...), prompt: str = "Describe this image"):
    try:
        logger.info("Received a prediction request.")

        # Read and validate the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        logger.debug(f"Image opened: {file.filename}")

        # Run inference
        result = model_inference.infer(image, prompt)
        logger.info("Inference completed successfully.")

        return InferenceResponse(
            status="success",
            prediction=result
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal Server Error"}
        )

# Custom Exception Handler (optional, for more detailed error handling)
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