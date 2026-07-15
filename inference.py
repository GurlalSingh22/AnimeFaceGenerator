import os
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import Generator

# =====================================================
# Configuration
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 100
CHANNELS = 3
FEATURES_G = 64

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "generator_final.pth"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Load Generator
# =====================================================

generator = Generator(
    LATENT_DIM,
    CHANNELS,
    FEATURES_G,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

generator.load_state_dict(state_dict)
generator.eval()

print("✅ Generator Loaded Successfully")

# =====================================================
# Generate Faces
# =====================================================

def generate_faces(num_images=16, output_name=None):
    if num_images not in {1, 4, 16, 64}:
        raise ValueError("num_images must be one of 1, 4, 16, or 64")

    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
        fake = generator(noise)
        fake = (fake + 1) / 2

        if output_name is None:
            output_name = f"anime_faces_{num_images}.png"

        output_path = OUTPUT_DIR / output_name
        nrow = int(num_images ** 0.5)
        save_image(fake, output_path, nrow=nrow)
        return str(output_path)
