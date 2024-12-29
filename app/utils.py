# app/utils.py
from typing import List, Tuple, Dict, Any
from PIL import Image

import logging
from typing import List, Tuple, Dict, Any
from PIL import Image

logger = logging.getLogger("uvicorn.error")

def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[Any]]:
    """
    Process vision information from messages.
    
    Args:
        messages: List of message dictionaries containing image and text content
        
    Returns:
        Tuple of (image_inputs, video_inputs) where:
        - image_inputs is a list of PIL Image objects
        - video_inputs is a list (empty in this case as we don't handle videos)
    """
    image_inputs = []
    video_inputs = []
    
    try:
        # Validate messages structure
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Invalid messages format")
            
        message = messages[0]  # We expect only one message
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("Invalid message format")
            
        # Find image in content
        for content in message["content"]:
            if content.get("type") == "image":
                image = content.get("image")
                logger.debug(f"Processing image of type: {type(image)}")
                
                if isinstance(image, Image.Image):
                    # Ensure image is in RGB mode
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image_inputs.append(image)
                    logger.debug(f"Added PIL Image: size={image.size}, mode={image.mode}")
                elif isinstance(image, str):
                    # Load and convert image from path
                    img = Image.open(image)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    image_inputs.append(img)
                    logger.debug(f"Added image from path: size={img.size}, mode={img.mode}")
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                    
        if not image_inputs:
            raise ValueError("No valid images found in message")
            
        return image_inputs, video_inputs
        
    except Exception as e:
        logger.error(f"Error processing vision info: {str(e)}")
        raise ValueError(f"Failed to process vision info: {str(e)}")
