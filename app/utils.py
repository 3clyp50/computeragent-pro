# app/utils.py
from typing import List, Tuple, Dict, Any
from PIL import Image

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
    
    for message in messages:
        if "content" in message:
            # Handle content which can be a list of different types
            for content in message["content"]:
                if isinstance(content, dict):
                    if content.get("type") == "image":
                        # The image could either be a PIL Image object or a path
                        image = content.get("image")
                        if isinstance(image, str):
                            # If it's a path, open the image
                            try:
                                image = Image.open(image)
                            except Exception as e:
                                raise ValueError(f"Failed to open image at path {image}: {str(e)}")
                        elif isinstance(image, Image.Image):
                            # If it's already a PIL Image, use it directly
                            pass
                        else:
                            raise ValueError(f"Unsupported image type: {type(image)}")
                        
                        image_inputs.append(image)
    
    return image_inputs, video_inputs