from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from .config import settings
import logging

logger = logging.getLogger("uvicorn.error")

class ModelInference:
    def __init__(self):
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                settings.MODEL_NAME,
                torch_dtype=getattr(torch, settings.TORCH_DTYPE.upper(), torch.float32) if settings.TORCH_DTYPE != "auto" else "auto",
                device_map=settings.DEVICE_MAP
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        try:
            logger.info(f"Loading processor for model: {settings.MODEL_NAME}")
            self.processor = AutoProcessor.from_pretrained(settings.MODEL_NAME)
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

    def extract_coordinates(self, text: str) -> str:
        """Extract coordinates and object reference from model output text using regex"""
        try:
            import re
            # Extract object reference and box coordinates using regex patterns
            object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
            box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"
            
            object_ref_match = re.search(object_ref_pattern, text)
            box_match = re.search(box_pattern, text)
            
            if box_match:
                box_content = box_match.group(1)
                # Convert coordinates to list format
                boxes = [tuple(map(int, pair.strip("()").split(','))) 
                        for pair in box_content.split("),(")]
                # Format as [[x1, y1, x2, y2]]
                coords = [[boxes[0][0], boxes[0][1], boxes[1][0], boxes[1][1]]]
                
                # Include object reference if found
                if object_ref_match:
                    object_ref = object_ref_match.group(1)
                    return f"{object_ref}: {coords}"
                return str(coords)
            
            # If no coordinates found, return original text
            logger.warning(f"No coordinates found in text: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            return text

    def infer(self, image: Image.Image, prompt: str) -> str:
        try:
            # Format messages directly with PIL Image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,  # Pass PIL Image directly
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
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

            # Process image input
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                if not image_inputs:
                    raise ValueError("No image inputs processed")
                logger.debug(f"Processed vision info: {len(image_inputs)} images")
                
                # Log image details for debugging
                for idx, img in enumerate(image_inputs):
                    logger.debug(f"Image {idx} - Size: {img.size}, Mode: {img.mode}")
            except Exception as e:
                logger.error(f"Error processing vision info: {e}")
                raise

            # Prepare model inputs - modified to match working example exactly
            try:
                logger.debug("Preparing inputs with processor...")
                
                # Ensure image is in expected format
                if len(image_inputs) == 0:
                    raise ValueError("No valid images to process")
                    
                # Create inputs exactly as in working example
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                logger.debug(f"Input keys: {inputs.keys()}")
                logger.debug(f"Input shapes - ids: {inputs.input_ids.shape}, attention: {inputs.attention_mask.shape}")
                
                inputs = inputs.to(self.device)
                logger.debug(f"Inputs moved to device {self.device}")
            except Exception as e:
                logger.error(f"Error preparing inputs: {e}")
                raise

            # Generate output
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128
                    )
                logger.debug(f"Generation completed. Shape: {generated_ids.shape}")
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                raise

            # Process output exactly as in working example
            try:
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                
                if not output_text:
                    raise ValueError("No output generated")
                
                # Extract coordinates from the output
                result = output_text[0]
                return self.extract_coordinates(result)
                
            except Exception as e:
                logger.error(f"Error processing output: {e}")
                raise

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e
