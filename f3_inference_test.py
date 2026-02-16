import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import our specific model
from geophysics.model.pipeline_net import GeophysicsPipelineNet
from geophysics.config import ProcessingConfig

def test_inference():
    # --- CONFIG ---
    CHECKPOINT = "checkpoint.pt"
    # Pick a specific line to test (e.g., inline 200)
    TEST_SEIS = "/Users/kirilboiko/Downloads/netherlands_f3_dataset/inlines/inline_200.tiff"
    TEST_MASK = "/Users/kirilboiko/Downloads/netherlands_f3_dataset/masks/inline_mask/inline_200_mask.png"
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Running inference on: {device}")

    # Load Model
    model = GeophysicsPipelineNet(config=ProcessingConfig()).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False), strict=False)
    model.eval()

    # Prep Data (Match training preprocessing)
    seis_img = Image.open(TEST_SEIS).convert('L').resize((512, 128))
    mask_img = Image.open(TEST_MASK).convert('L').resize((512, 128))
    
    seis_np = np.array(seis_img).astype(np.float32) / 255.0
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    
    seis_t = torch.from_numpy(seis_np).unsqueeze(0).unsqueeze(0).to(device) # (B, C, H, W)
    
    # Missing modalities
    mag_t = torch.zeros_like(seis_t).squeeze(1).to(device)
    grav_t = torch.zeros_like(seis_t).squeeze(1).to(device)
    avail_mask = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        # Adjusting for model forward signature: (seismic, magnetics, gravity, mask)
        out = model(seis_t.squeeze(1), mag_t, grav_t, avail_mask)
        prediction = out["section"].squeeze().cpu().numpy()

    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Raw Seismic (Input)")
    plt.imshow(seis_np, cmap='gray', aspect='auto')
    
    plt.subplot(1, 3, 2)
    plt.title("Human Interpretation (Target)")
    plt.imshow(mask_np, cmap='jet', aspect='auto')
    
    plt.subplot(1, 3, 3)
    plt.title("AI Interpretation (Output)")
    plt.imshow(prediction, cmap='jet', aspect='auto')
    
    plt.tight_layout()
    plt.savefig("f3_test_comparison.png")
    print("Comparison saved to 'f3_test_comparison.png'")

if __name__ == "__main__":
    test_inference()
