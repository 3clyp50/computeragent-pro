from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from .utils import process_vision_info
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

    def infer(self, image: Image.Image, prompt: str) -> str:
        try:
            # Save image to temporary file to match working implementation
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                # Save image in JPEG format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(temp_file.name, 'JPEG')
                logger.debug(f"Saved image to temporary file: {temp_file.name}")
                
                try:
                    # Format messages as in working implementation
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": temp_file.name,  # Use file path instead of PIL Image
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
                        logger.debug(f"Chat template applied: {text[:100]}...")  # Log first 100 chars
                    except Exception as e:
                        logger.error(f"Error applying chat template: {e}")
                        raise

                    # Process image input
                    try:
                        image_inputs, video_inputs = process_vision_info(messages)
                        if not image_inputs:
                            raise ValueError("No image inputs processed")
                        logger.debug(f"Processed vision info: {len(image_inputs)} images")
                    except Exception as e:
                        logger.error(f"Error processing vision info: {e}")
                        raise

                    # Prepare model inputs
                    try:
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(self.device)
                        logger.debug(f"Inputs prepared and moved to device. Shape: {inputs.input_ids.shape}")
                    except Exception as e:
                        logger.error(f"Error preparing inputs: {e}")
                        raise

                    # Generate output
                    try:
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=128,
                                pad_token_id=self.processor.tokenizer.pad_token_id,
                                eos_token_id=self.processor.tokenizer.eos_token_id
                            )
                        logger.debug(f"Generation completed. Shape: {generated_ids.shape}")
                    except Exception as e:
                        logger.error(f"Error during generation: {e}")
                        raise

                    # Process output
                    try:
                        if len(inputs.input_ids) == 0 or len(generated_ids) == 0:
                            raise ValueError("Empty input or generated IDs")
                            
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] 
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        logger.debug(f"Trimmed IDs. Length: {len(generated_ids_trimmed)}")

                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False
                        )
                        logger.debug(f"Output decoded. Length: {len(output_text)}")

                        if not output_text:
                            raise ValueError("No output generated")

                        return output_text[0]
                    except Exception as e:
                        logger.error(f"Error processing output: {e}")
                        raise
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file.name)
                        logger.debug("Temporary file cleaned up")
                    except Exception as e:
                        logger.warning(f"Error cleaning up temporary file: {e}")
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e
