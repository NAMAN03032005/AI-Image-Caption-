# build_model_cache.py
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

print("--- DOWNLOADING MODEL FOR CACHE SPEEDUPS ---")
try:
    import torch
    # Pre-cache vit-gpt2 for Render execution speed
    VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", torch_dtype=torch.float16)
    ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    print("--- CACHING SUCCESSFUL ---")
except Exception as e:
    print(f"Caching skipped/failed: {e}")
