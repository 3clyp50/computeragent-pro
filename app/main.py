import logging
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
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
    file: UploadFile = File(...), 
    prompt: str = "Describe this image",
    api_key: str = Depends(get_api_key)
):
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
            content={"status": "error", "message": str(e)}
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