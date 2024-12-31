import logging
import os
import re
import base64
from io import BytesIO

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from .config import settings

logger = logging.getLogger("uvicorn.error")


def get_cache_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 PNG string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def draw_bounding_boxes(
    image: Image.Image, bounding_boxes: list, outline_color="red", line_width=2
) -> Image.Image:
    """Draw bounding boxes on the given image (for debug/visual checks)."""
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
    return image


def rescale_bounding_boxes(
    bounding_boxes: list,
    original_width: int,
    original_height: int,
    scaled_width=1000,
    scaled_height=1000
) -> list:
    """
    Rescale bounding boxes from model coordinate space to actual image dimensions.
    """
    rescaled_boxes = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        # First normalize to 0-1 range
        norm_box = [
            xmin / scaled_width,
            ymin / scaled_height,
            xmax / scaled_width,
            ymax / scaled_height
        ]
        # Then scale to actual dimensions
        scaled_box = [
            round(norm_box[0] * original_width, 2),
            round(norm_box[1] * original_height, 2),
            round(norm_box[2] * original_width, 2),
            round(norm_box[3] * original_height, 2)
        ]
        rescaled_boxes.append(scaled_box)
    return rescaled_boxes

class ModelInference:
    def __init__(self):
        """
        Initialize model and processor from local cache or download if not found.
        Sets up the correct device for inference.
        """
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")
            cache_dir = get_cache_dir()
            local_model_path = os.path.join(cache_dir, settings.MODEL_NAME.split('/')[-1])

            # Load from local cache if available
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from cache: {local_model_path}")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype=(
                        getattr(torch, settings.TORCH_DTYPE.upper(), torch.float32)
                        if settings.TORCH_DTYPE != "auto" else
                        "auto"
                    ),
                    device_map=settings.DEVICE_MAP,
                    local_files_only=True
                )
            else:
                logger.info(f"Cache not found, downloading model: {settings.MODEL_NAME}")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    settings.MODEL_NAME,
                    torch_dtype=(
                        getattr(torch, settings.TORCH_DTYPE.upper(), torch.float32)
                        if settings.TORCH_DTYPE != "auto" else
                        "auto"
                    ),
                    device_map=settings.DEVICE_MAP,
                    cache_dir=cache_dir
                )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        try:
            logger.info(f"Loading processor for model: {settings.MODEL_NAME}")
            if os.path.exists(local_model_path):
                logger.info("Loading processor from cache")
                self.processor = AutoProcessor.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
            else:
                logger.info("Downloading processor")
                self.processor = AutoProcessor.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=cache_dir
                )
            logger.info("Processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def warmup(self):
        """Perform a trivial inference to ensure the model and processor are ready."""
        try:
            dummy_image = Image.new('RGB', (224, 224))
            self.infer(dummy_image, "test prompt")
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")

    def infer(self, image: Image.Image, command: str):
        """
        Perform inference to find bounding-box coordinates of the UI element
        matching the given command. Returns (raw_output_text, [scaled_box]).
        """
        try:
            # Prompt match your HF example
            prompt_text = (
                f"In this UI screenshot, what is the position of the element "
                f'corresponding to the command "{command}" (with bbox)?'
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            logger.debug("Messages formatted.")

            # Convert to Qwen’s chat format
            text_for_model = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"Chat template applied. Truncated: {text_for_model[:150]}...")

            # Build model inputs
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_for_model],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)

            # Remove tokens from the original prompt
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            raw_output = outputs[0]
            logger.debug(f"Raw model output: {raw_output}")

            # Extract box coordinates
            object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
            box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

            box_content_match = re.search(box_pattern, raw_output)
            if not box_content_match:
                logger.warning("No box coordinates found in model output.")
                return raw_output, []

            box_content = box_content_match.group(1)

            # Parse the (x1,y1),(x2,y2) format from box_content
            coords = [tuple(map(float, pair.strip("()").split(','))) 
                    for pair in box_content.split("),(")]

            if len(coords) == 2:
                # Convert to [xmin, ymin, xmax, ymax] format
                boxes = [[coords[0][0], coords[0][1], coords[1][0], coords[1][1]]]

                # First, normalize coordinates to 0-1 range
                normalized_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    # Normalize assuming model's internal coordinate space is 1000x1000
                    norm_box = [
                        xmin / 1000.0,
                        ymin / 1000.0,
                        xmax / 1000.0,
                        ymax / 1000.0
                    ]
                    normalized_boxes.append(norm_box)

                # Then scale to actual image dimensions
                scaled_boxes = []
                for norm_box in normalized_boxes:
                    xmin, ymin, xmax, ymax = norm_box
                    scaled_box = [
                        round(xmin * image.width, 2),
                        round(ymin * image.height, 2),
                        round(xmax * image.width, 2),
                        round(ymax * image.height, 2)
                    ]
                    scaled_boxes.append(scaled_box)

                # Validate coordinates
                for box in scaled_boxes:
                    xmin, ymin, xmax, ymax = box
                    if xmin >= xmax or ymin >= ymax:
                        logger.warning(f"Invalid box coordinates: {box}")
                        continue
                    if xmin < 0 or ymin < 0 or xmax > image.width or ymax > image.height:
                        logger.warning(f"Coordinates out of bounds: {box}")
                        # Clamp to image boundaries
                        box[0] = max(0, min(xmin, image.width))
                        box[1] = max(0, min(ymin, image.height))
                        box[2] = max(0, min(xmax, image.width))
                        box[3] = max(0, min(ymax, image.height))

                return raw_output, scaled_boxes
            else:
                logger.warning("Invalid coordinate format")
                return raw_output, []

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e