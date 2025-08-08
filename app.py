import streamlit as st
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
from PIL import Image  # Required for image resizing
import altair.vegalite.v4 as alt



class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)

    # Image generation settings
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

    # Prompt generation settings (if used)
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token='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'  # Replace with your real token
)

image_gen_model = image_gen_model.to(CFG.device)


def generate_image(prompt, model):
    image = model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

st.title("Image Generation with Stable Diffusion")

# Input field with a default prompt
prompt = st.text_input("Enter a prompt for the image:", "A man playing chess")

# Button to trigger image generation
generate_button = st.button("Generate Image")

if generate_button:
    with st.spinner("Generating image..."):
        image = generate_image(prompt, image_gen_model)  # Calls your earlier function
        st.image(
            image,
            caption=f"Generated Image for Prompt: {prompt}",
            use_column_width=True
        )



