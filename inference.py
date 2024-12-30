import os
import torch
import logging
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from dataclasses import dataclass
from typing import Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the model"""
    model_name: str = "OS-Copilot/OS-Atlas-Base-7B"
    device_map: str = "auto"  # or specific like "cuda:0"
    torch_dtype: str = "auto"
    cache_dir: str = "model_cache"

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def draw_bounding_boxes(image: Image.Image, bounding_boxes: List[List[float]], 
                       outline_color: str = "red", line_width: int = 2) -> Image.Image:
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
    return image

def rescale_bounding_boxes(bounding_boxes: List[List[float]], 
                          original_width: int, original_height: int,
                          scaled_width: int = 1000, scaled_height: int = 1000) -> List[List[float]]:
    """Rescale bounding boxes from model coordinates to image dimensions"""
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

class UIElementLocator:
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model with given configuration"""
        self.config = config or ModelConfig()
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model and processor"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=getattr(torch, self.config.torch_dtype.upper(), torch.float32) 
                           if self.config.torch_dtype != "auto" else "auto",
                device_map=self.config.device_map,
                cache_dir=self.config.cache_dir
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Model loaded successfully. Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def warmup(self):
        """Perform model warmup inference"""
        try:
            dummy_image = Image.new('RGB', (224, 224))
            self.locate_element(dummy_image, "test button")
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")

    def locate_element(self, image: Image.Image, element_description: str) -> Union[List[float], List]:
        """
        Locate a UI element in the image based on the description
        
        Args:
            image: PIL Image object containing the UI screenshot
            element_description: Description of the element to locate (e.g., "login button")
            
        Returns:
            List of coordinates [x1, y1, x2, y2] or empty list if element not found
        """
        try:
            # Format prompt for precise coordinate detection
            prompt = f'Find the exact pixel coordinates of the element labeled "{element_description}" in this UI screenshot. Return a tight bounding box that includes ONLY the element itself.'
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128
                )

            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )[0]

            # Extract coordinates
            box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"
            box_match = re.search(box_pattern, output_text)
            
            if not box_match:
                logger.warning("No coordinates found in model output")
                return []

            # Parse coordinates
            try:
                coords_str = box_match.group(1).strip('[]')
                coords = [float(x.strip()) for x in coords_str.split(',')]
                
                if len(coords) != 4:
                    logger.warning(f"Invalid coordinate format: {coords}")
                    return []
                
                # Scale coordinates to image dimensions
                scaled_boxes = rescale_bounding_boxes([[coords[0], coords[1], coords[2], coords[3]]], 
                                                    image.width, image.height)
                
                return scaled_boxes[0] if scaled_boxes else []
                
            except ValueError as e:
                logger.error(f"Error parsing coordinates: {e}")
                return []

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []

def main():
    """Example usage"""
    # Initialize model
    locator = UIElementLocator()
    
    # Load an image
    image_path = "screenshot.png"  # Replace with your image path
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    # Locate element
    element_description = "login button"  # Replace with your element description
    coordinates = locator.locate_element(image, element_description)
    
    if coordinates:
        print(f"Element found at coordinates: {coordinates}")
        
        # Optionally draw bounding box
        annotated_image = draw_bounding_boxes(image.copy(), [coordinates])
        annotated_image.save("annotated_screenshot.png")
    else:
        print("Element not found")

if __name__ == "__main__":
    main()
