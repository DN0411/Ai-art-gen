from diffusers import StableDiffusionPipeline
import torch
import os
import time
import gc
from huggingface_hub import snapshot_download
import streamlit as st
from PIL import Image

# Paths
MODEL_PATH = "/home/diyen/sd-turbo"
OUTPUT_DIR = "/home/diyen/AI art gen app/images"

# Streamlit Page Setup
st.set_page_config(page_title="AI Art Generator", layout="wide")
st.title("🎨 AI Art Generator")
prompt = st.text_input("Enter a prompt to generate an image:")

# Download model if not already
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        snapshot_download(repo_id="stabilityai/sd-turbo", local_dir=MODEL_PATH, local_dir_use_symlinks=False)

# Function to generate image
def generate_image(prompt, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"image_{int(time.time())}.png")

    # Load model
    torch.backends.cudnn.benchmark = True
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
        low_cpu_mem_usage=True
    ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    # Generate image
    image = pipe(prompt, num_inference_steps=10, guidance_scale=3.0, height=512, width=512).images[0]
    image.save(output_path)

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return image, output_path


# UI Logic
if prompt:
    with st.spinner("Generating image..."):
        image, path = generate_image(prompt)
        st.image(image, caption="Generated Image", use_container_width=True)

        # Download button
        with open(path, "rb") as img_file:
            btn = st.download_button(
            label="Download Image",data=img_file,file_name=os.path.basename(path),mime="image/png")