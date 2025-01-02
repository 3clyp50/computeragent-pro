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
    x_scale = original_width / scaled_width
    y_scale = original_height / scaled_height
    rescaled_boxes = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        rescaled_box = [
            xmin * x_scale,
            ymin * y_scale,
            xmax * x_scale,
            ymax * y_scale
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
            # Format prompt and messages similar to the example
            prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{prompt}\" (with bbox)?"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            logger.debug("Messages formatted")

            # Process text input
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                logger.debug(f"Chat template applied: {text[:100]}...")
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

                # Process output text
                text = output_text[0]
                logger.debug(f"Raw model output: {text}")

                # Extract coordinates using regex patterns
                import re
                object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

                try:
                    object_ref = re.search(object_ref_pattern, text).group(1)
                    box_content = re.search(box_pattern, text).group(1)

                    # Parse coordinates in the same format as the example
                    boxes = [tuple(map(int, pair.strip("()").split(','))) 
                            for pair in box_content.split("),(")]
                    boxes = [[boxes[0][0], boxes[0][1], boxes[1][0], boxes[1][1]]]

                    # Scale boxes to image dimensions
                    scaled_boxes = rescale_bounding_boxes(boxes, image.width, image.height)

                    # Draw boxes for visualization (optional)
                    draw_bounding_boxes(image.copy(), scaled_boxes)

                    # Return object reference and scaled coordinates
                    return object_ref, scaled_boxes

                except (AttributeError, IndexError, ValueError) as e:
                    logger.error(f"Error parsing model output: {e}")
                    return text, []

            except Exception as e:
                logger.error(f"Error processing output: {e}")
                raise

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e