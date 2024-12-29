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
            # Create a temporary file path for the image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image  # Pass PIL Image directly
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Process text input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.debug("Chat template applied")

            # Process image input
            image_inputs, video_inputs = process_vision_info(messages)
            if not image_inputs:
                raise ValueError("No image inputs processed")
            logger.debug(f"Processed vision info: {len(image_inputs)} images")

            # Prepare model inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            logger.debug("Inputs prepared and moved to device")

            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            logger.debug("Generation completed")

            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            logger.debug("Output decoded")

            if not output_text:
                raise ValueError("No output generated")

            return output_text[0]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e
