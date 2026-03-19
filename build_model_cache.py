# build_model_cache.py
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

print("--- DOWNLOADING MODEL FOR CACHE SPEEDUPS ---")
try:
    # Pre-cache vit-gpt2 for Render execution speed
    VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    print("--- CACHING SUCCESSFUL ---")
except Exception as e:
    print(f"Caching skipped/failed: {e}")
