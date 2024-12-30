import logging
import os
import re
import base64
from io import BytesIO
from typing import Tuple, List, Optional

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

class CoordinateError(Exception):
    """Custom exception for coordinate processing errors."""
    pass

def get_cache_dir() -> str:
    """Return the path to the model cache directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 PNG string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def draw_bounding_boxes(
    image: Image.Image, 
    bounding_boxes: List[List[float]], 
    outline_color: str = "red", 
    line_width: int = 2
) -> Image.Image:
    """
    Draw bounding boxes on the given image for visualization.

    Args:
        image: PIL Image to draw on
        bounding_boxes: List of [xmin, ymin, xmax, ymax] coordinates
        outline_color: Color of the bounding box
        line_width: Width of the box lines

    Returns:
        PIL Image with drawn bounding boxes
    """
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline=outline_color,
            width=line_width
        )
    return image

def validate_coordinates(
    box: List[float],
    image_width: int,
    image_height: int,
    min_box_size: int = 5
) -> Tuple[bool, List[str]]:
    """
    Validate coordinate values and box dimensions.

    Args:
        box: [xmin, ymin, xmax, ymax] coordinates
        image_width: Width of the original image
        image_height: Height of the original image
        min_box_size: Minimum allowed box dimension

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    xmin, ymin, xmax, ymax = box
    issues = []

    # Check ordering
    if xmin >= xmax or ymin >= ymax:
        issues.append(f"Invalid coordinate ordering: ({xmin},{ymin}),({xmax},{ymax})")

    # Check boundaries
    if xmin < 0 or ymin < 0:
        issues.append(f"Negative coordinates: ({xmin},{ymin})")
    if xmax > image_width or ymax > image_height:
        issues.append(f"Coordinates exceed image dimensions: ({xmax},{ymax})")

    # Check box size
    if xmax - xmin < min_box_size or ymax - ymin < min_box_size:
        issues.append(f"Bounding box too small: {xmax-xmin}x{ymax-ymin}")

    return len(issues) == 0, issues

def rescale_coordinates(
    box: List[float],
    image_width: int,
    image_height: int,
    model_space: int = 1000
) -> List[float]:
    """
    Rescale coordinates from model space to image dimensions.

    Args:
        box: [xmin, ymin, xmax, ymax] in model space
        image_width: Target image width
        image_height: Target image height
        model_space: Size of model's coordinate space (default 1000x1000)

    Returns:
        Scaled coordinates [xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = box

    # Normalize to 0-1 range
    norm_box = [
        x1 / model_space,
        y1 / model_space,
        x2 / model_space,
        y2 / model_space
    ]

    # Scale to image dimensions
    scaled_box = [
        round(norm_box[0] * image_width, 2),
        round(norm_box[1] * image_height, 2),
        round(norm_box[2] * image_width, 2),
        round(norm_box[3] * image_height, 2)
    ]

    return scaled_box

class ModelInference:
    def __init__(self):
        """Initialize model and processor with proper error handling and logging."""
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            self._load_model()
            self._load_processor()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _load_model(self):
        """Load the model from cache or download if needed."""
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        cache_dir = get_cache_dir()
        local_model_path = os.path.join(cache_dir, settings.MODEL_NAME.split('/')[-1])

        try:
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
                logger.info(f"Downloading model: {settings.MODEL_NAME}")
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
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _load_processor(self):
        """Load the processor from cache or download if needed."""
        try:
            cache_dir = get_cache_dir()
            local_model_path = os.path.join(cache_dir, settings.MODEL_NAME.split('/')[-1])

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
        except Exception as e:
            logger.error(f"Processor loading failed: {str(e)}")
            raise

    def warmup(self):
        """Perform a warmup inference pass."""
        try:
            dummy_image = Image.new('RGB', (224, 224))
            self.infer(dummy_image, "test prompt")
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {str(e)}")
            raise

    def parse_coordinates(self, raw_output: str) -> Optional[List[float]]:
        """
        Parse coordinate strings from model output.

        Args:
            raw_output: Model's raw output string

        Returns:
            List of coordinates [xmin, ymin, xmax, ymax] or None if parsing fails
        """
        try:
            box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"
            box_match = re.search(box_pattern, raw_output)

            if not box_match:
                logger.warning("No box coordinates found in output")
                return None

            box_content = box_match.group(1)
            coords = [tuple(map(float, pair.strip("()").split(','))) 
                     for pair in box_content.split("),(")]

            if len(coords) != 2:
                logger.warning(f"Invalid coordinate format: {box_content}")
                return None

            return [coords[0][0], coords[0][1], coords[1][0], coords[1][1]]

        except Exception as e:
            logger.error(f"Coordinate parsing failed: {str(e)}")
            return None

    def infer(
        self,
        image: Image.Image,
        command: str
    ) -> Tuple[str, List[List[float]]]:
        """
        Perform inference to locate UI elements.

        Args:
            image: Input PIL Image
            command: User command/prompt

        Returns:
            Tuple of (raw_output_text, list_of_scaled_boxes)
        """
        try:
            # Prepare prompt
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

            # Process inputs
            text_for_model = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_for_model],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
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

            # Parse and process coordinates
            raw_box = self.parse_coordinates(raw_output)
            if not raw_box:
                return raw_output, []

            # Scale coordinates
            scaled_box = rescale_coordinates(raw_box, image.width, image.height)

            # Validate results
            is_valid, issues = validate_coordinates(scaled_box, image.width, image.height)
            if not is_valid:
                logger.warning(f"Coordinate validation issues: {issues}")
                # Attempt to fix coordinates by clamping to image boundaries
                scaled_box = [
                    max(0, min(scaled_box[0], image.width)),
                    max(0, min(scaled_box[1], image.height)),
                    max(0, min(scaled_box[2], image.width)),
                    max(0, min(scaled_box[3], image.height))
                ]

            return raw_output, [scaled_box]

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise