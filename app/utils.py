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
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content.get("type") == "image":
                        image = content.get("image")
                        if isinstance(image, Image.Image):
                            image_inputs.append(image)
                        elif isinstance(image, str):
                            image_inputs.append(Image.open(image))
                        else:
                            logger.warning(f"Skipping unsupported image type: {type(image)}")
                            
        return image_inputs, video_inputs
        
    except Exception as e:
        logger.error(f"Error processing vision info: {str(e)}")
        raise ValueError(f"Failed to process vision info: {str(e)}")
