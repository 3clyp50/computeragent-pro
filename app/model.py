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
                torch_dtype=getattr(torch, settings.TORCH_DTYPE.upper(), torch.float32),
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

    def infer(self, image: Image.Image, prompt: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug("Model inputs prepared and moved to device.")

            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            logger.debug("Inference generation completed.")

            return output_text[0]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e