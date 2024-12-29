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
        if not messages:
            raise ValueError("No messages provided")
            
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Invalid message format: {type(message)}")
                
            if "content" not in message:
                logger.warning(f"No content in message: {message}")
                continue
                
            if not isinstance(message["content"], list):
                raise ValueError(f"Content must be a list, got: {type(message['content'])}")
                
            for content in message["content"]:
                if not isinstance(content, dict):
                    logger.warning(f"Skipping invalid content type: {type(content)}")
                    continue
                    
                if content.get("type") == "image":
                    image = content.get("image")
                    if image is None:
                        raise ValueError("Image content is None")
                        
                    if isinstance(image, str):
                        try:
                            image = Image.open(image)
                            logger.debug(f"Opened image from path: {image}")
                        except Exception as e:
                            raise ValueError(f"Failed to open image at path {image}: {str(e)}")
                    elif isinstance(image, Image.Image):
                        logger.debug("Using provided PIL Image directly")
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                    
                    # Ensure image is in RGB mode
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                        logger.debug(f"Converted image to RGB mode from {image.mode}")
                    
                    image_inputs.append(image)
                    logger.debug(f"Added image to inputs. Total images: {len(image_inputs)}")
        
        if not image_inputs:
            raise ValueError("No valid images found in messages")
            
        return image_inputs, video_inputs
        
    except Exception as e:
        logger.error(f"Error processing vision info: {str(e)}")
        raise
