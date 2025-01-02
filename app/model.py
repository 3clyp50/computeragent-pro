from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw
import torch
from qwen_vl_utils import process_vision_info
from .config import settings
import logging
import base64
from io import BytesIO
import os
import re
from typing import Union, Tuple, List

logger = logging.getLogger("uvicorn.error")

def get_cache_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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

    def draw_bounding_boxes(self, image, bounding_boxes, outline_color="red", line_width=2):
        draw = ImageDraw.Draw(image)
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
        return image

    def warmup(self):
        """Perform model warmup inference"""
        try:
            dummy_image = Image.new('RGB', (224, 224))
            self.infer(dummy_image, "test prompt")
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")

    def process_image_input(self, image_data: Union[str, bytes, Image.Image]) -> Image.Image:
        """Process different types of image inputs into PIL Image"""
        try:
            if isinstance(image_data, Image.Image):
                return image_data
            
            if isinstance(image_data, str):
                # Handle base64 string
                if image_data.startswith('data:image'):
                    # Extract base64 data after the comma
                    base64_data = re.sub('^data:image/.+;base64,', '', image_data)
                else:
                    base64_data = image_data
                
                image_bytes = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_bytes))
            
            if isinstance(image_data, bytes):
                return Image.open(BytesIO(image_data))
            
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
        except Exception as e:
            logger.error(f"Error processing image input: {e}")
            raise

    def infer(self, image: Union[str, bytes, Image.Image], prompt: str) -> Tuple[str, List]:
        """Updated infer method to handle multiple image input types"""
        try:
            # Process the image input
            processed_image = self.process_image_input(image)
            
            # Continue with existing inference logic
            prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{prompt}\" (with bbox)?"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{image_to_base64(processed_image)}"},
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
                object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

                try:
                    object_ref = re.search(object_ref_pattern, text)
                    box_content = re.search(box_pattern, text)
                    
                    if not object_ref or not box_content:
                        logger.warning("Could not find object reference or box coordinates in model output")
                        return "", [[]]  # Return empty string and properly formatted empty box list

                    object_ref = object_ref.group(1)
                    box_content = box_content.group(1)
                    logger.debug(f"Raw box content before cleanup: {box_content}")

                    # Clean up the box content string and remove any unexpected characters
                    box_content = re.sub(r'[\[\]]', '', box_content)
                    logger.debug(f"Box content after cleanup: {box_content}")
                    
                    # First try parsing as a single array of 4 coordinates
                    try:
                        coords = [int(x.strip()) for x in box_content.split(',')]
                        if len(coords) == 4:
                            logger.debug(f"Successfully parsed single array format: {coords}")
                            boxes = [coords]
                            scaled_boxes = rescale_bounding_boxes(boxes, processed_image.width, processed_image.height)
                            return object_ref, scaled_boxes
                    except ValueError:
                        logger.debug("Failed to parse as single array, trying pair format")
                    
                    # If that fails, try the pair format
                    coord_pairs = box_content.split('),(')
                    logger.debug(f"Split coordinate pairs: {coord_pairs}")
                    coord_pairs = [pair.strip('()') for pair in coord_pairs]
                    logger.debug(f"Cleaned coordinate pairs: {coord_pairs}")
                    
                    if len(coord_pairs) != 2:
                        logger.warning(f"Failed to parse coordinates in either format. Got pairs: {coord_pairs}")
                        return "", [[]]  # Return empty string and properly formatted empty box list

                    try:
                        # Parse each coordinate pair
                        boxes = []
                        for pair in coord_pairs:
                            coords = [int(x.strip()) for x in pair.split(',')]
                            if len(coords) != 2:
                                raise ValueError(f"Invalid coordinate pair: {pair}")
                            boxes.extend(coords)

                        if len(boxes) != 4:
                            raise ValueError(f"Expected 4 coordinates, got {len(boxes)}")

                        boxes = [boxes]  # Wrap in list to maintain expected format

                        # Scale boxes to image dimensions
                        scaled_boxes = rescale_bounding_boxes(boxes, processed_image.width, processed_image.height)

                        # Draw boxes for visualization (optional)
                        self.draw_bounding_boxes(processed_image.copy(), scaled_boxes)

                        # Return object reference and scaled coordinates
                        return object_ref, scaled_boxes

                    except (AttributeError, IndexError, ValueError) as e:
                        logger.error(f"Error parsing model output: {e}")
                        return "", [[]]  # Return empty string and properly formatted empty box list

                except (AttributeError, IndexError, ValueError) as e:
                    logger.error(f"Error parsing model output: {e}")
                    return "", [[]]  # Return empty string and properly formatted empty box list

            except Exception as e:
                logger.error(f"Error processing output: {e}")
                raise

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e