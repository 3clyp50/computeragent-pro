from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw
import torch
from qwen_vl_utils import process_vision_info
from .config import settings
import logging
import base64
from io import BytesIO
import os

logger = logging.getLogger("uvicorn.error")

def get_cache_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def draw_bounding_boxes(image, bounding_boxes, outline_color="red", line_width=2):
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
    return image

def rescale_bounding_boxes(bounding_boxes, original_width, original_height, scaled_width=1000, scaled_height=1000):
    """Rescale bounding boxes from model coordinates to image dimensions"""
    # Calculate scaling factors
    x_scale = original_width / scaled_width
    y_scale = original_height / scaled_height
    
    rescaled_boxes = []
    for box in bounding_boxes:
        if len(box) != 4:
            continue
            
        xmin, ymin, xmax, ymax = box
        
        # Ensure coordinates are within bounds
        xmin = max(0, min(scaled_width, xmin))
        ymin = max(0, min(scaled_height, ymin))
        xmax = max(0, min(scaled_width, xmax))
        ymax = max(0, min(scaled_height, ymax))
        
        # Apply scaling
        rescaled_box = [
            round(xmin * x_scale, 2),
            round(ymin * y_scale, 2),
            round(xmax * x_scale, 2),
            round(ymax * y_scale, 2)
        ]
        rescaled_boxes.append(rescaled_box)
    
    return rescaled_boxes

class ModelInference:
    def __init__(self):
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")
            # Try to load from cache first
            cache_dir = get_cache_dir()
            local_model_path = os.path.join(cache_dir, settings.MODEL_NAME.split('/')[-1])
            
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from cache: {local_model_path}")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype=getattr(torch, settings.TORCH_DTYPE.upper(), torch.float32) if settings.TORCH_DTYPE != "auto" else "auto",
                    device_map=settings.DEVICE_MAP,
                    local_files_only=True
                )
            else:
                logger.info(f"Cache not found, downloading model: {settings.MODEL_NAME}")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    settings.MODEL_NAME,
                    torch_dtype=getattr(torch, settings.TORCH_DTYPE.upper(), torch.float32) if settings.TORCH_DTYPE != "auto" else "auto",
                    device_map=settings.DEVICE_MAP,
                    cache_dir=cache_dir
                )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        try:
            logger.info(f"Loading processor for model: {settings.MODEL_NAME}")
            # Try to load processor from cache
            if os.path.exists(local_model_path):
                logger.info("Loading processor from cache")
                self.processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
            else:
                logger.info("Downloading processor")
                self.processor = AutoProcessor.from_pretrained(settings.MODEL_NAME, cache_dir=cache_dir)
            logger.info("Processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise e

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def warmup(self):
        """Perform model warmup inference"""
        try:
            dummy_image = Image.new('RGB', (224, 224))
            self.infer(dummy_image, "test prompt")
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")

    def infer(self, image: Image.Image, prompt: str) -> tuple[str, list]:
        try:
            # Format messages
            prompt = f"Find the exact pixel coordinates of \"{prompt}\" in this screenshot."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Process text input
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.error(f"Error applying chat template: {e}")
                raise

            # Process inputs and generate output
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                
                # Process output and extract coordinates exactly like HuggingFace Space
                text = output_text[0]
                
                # Extract coordinates using regex patterns
                import re
                object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

                object_ref = re.search(object_ref_pattern, text)
                if not object_ref:
                    logger.warning("No object reference found in model output")
                    return text, []
                    
                box_content = re.search(box_pattern, text)
                if not box_content:
                    logger.warning("No box coordinates found in model output")
                    return text, []

                # Parse coordinates - expect format [x1,y1,x2,y2]
                try:
                    # Clean up the box content and parse as a list of floats
                    coords_str = box_content.group(1).strip('[]')
                    coords = [float(x.strip()) for x in coords_str.split(',')]
                    
                    if len(coords) != 4:
                        logger.warning(f"Invalid coordinate format: {coords}")
                        return []
                    
                    # Create box in [x1,y1,x2,y2] format
                    box = [[coords[0], coords[1], coords[2], coords[3]]]
                    
                    # Scale boxes to image dimensions
                    scaled_boxes = rescale_bounding_boxes(box, image.width, image.height)
                    
                    # Draw boxes internally to help model determine coordinates
                    draw_bounding_boxes(image.copy(), scaled_boxes)
                    
                    # Return only the scaled coordinates
                    return scaled_boxes[0] if scaled_boxes else []
                except ValueError as e:
                    logger.error(f"Error parsing coordinates: {e}")
                    return []
                
            except Exception as e:
                logger.error(f"Error processing output: {e}")
                raise

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e
