import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our specific modelpython3 f3_train_m3.py --resume checkpoint.pt
from geophysics.model.pipeline_net import GeophysicsPipelineNet
from geophysics.config import ProcessingConfig

os.makedirs("results", exist_ok=True)
class F3Dataset(Dataset):
    def __init__(self, seismic_dir, mask_dir, target_shape=(128, 512)):
        self.seismic_dir = seismic_dir
        self.mask_dir = mask_dir
        self.target_shape = target_shape
        
        # Match files (e.g., inline_100.tiff -> inline_100_mask.png)
        self.seismic_files = sorted([f for f in os.listdir(seismic_dir) if f.endswith('.tiff')])
        self.pairs = []
        for f in self.seismic_files:
            mask_name = f.replace('.tiff', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.pairs.append((os.path.join(seismic_dir, f), mask_path))
        
        print(f"Found {len(self.pairs)} training pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seis_path, mask_path = self.pairs[idx]
        
        # Load and resize
        seis = Image.open(seis_path).convert('L').resize((self.target_shape[1], self.target_shape[0]))
        mask = Image.open(mask_path).convert('L').resize((self.target_shape[1], self.target_shape[0]))
        
        # Normalize to 0-1
        seis_np = np.array(seis).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        
        # Remove extra channel dimension: (H, W)
        seis_t = torch.from_numpy(seis_np)
        # mask_np is likely 0-255 or 0-N. For F3, we assume pixel values map to classes.
        # If it's a PNG mask, PIL opens it. We need discrete integers.
        mask_array = np.array(mask)
        # Verify if values are 0-5. If 0-255 scaling happened, revert or just take raw.
        # Here we assume mask strings are class indices. 
        mask_t = torch.from_numpy(mask_array).long()
        
        # Missing modalities (magnetics/gravity) set to zero (H, W)
        mag_t = torch.zeros_like(seis_t)
        grav_t = torch.zeros_like(seis_t)
        
        # Availability mask: [Seismic=1, Mag=0, Grav=0]
        avail_mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        
        return seis_t, mag_t, grav_t, avail_mask, mask_t

def train():
    # --- CONFIG ---
    DATA_DIR = "/Users/kirilboiko/Downloads/netherlands_f3_dataset"
    SEISMIC_DIR = os.path.join(DATA_DIR, "inlines")
    MASK_DIR = os.path.join(DATA_DIR, "masks/inline_mask")
    EPOCHS = 50
    BATCH_SIZE = 1 # Reducing batch size for M3 Pro stability
    LR = 1e-4
    
    # Device: M3 Pro GPU (MPS)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training on: {device}")

    # Data
    dataset = F3Dataset(SEISMIC_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = GeophysicsPipelineNet(config=ProcessingConfig(), out_channels=6).to(device)
    
    # Load previous weights if they exist
    if os.path.exists("checkpoint.pt"):
        print("Resuming from existing checkpoint...")
        # PyTorch 2.6 security fix: adding weights_only=False
        # Custom loading to handle size mismatch in the final layer
        checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=False)
        model_state = model.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_state and v.size() == model_state[k].size()}
        # Overwrite entries in the existing state dict
        model_state.update(pretrained_dict)
        # Load the new state dict
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_state)} layers from checkpoint.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- TRAINING LOOP ---
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(loader, leave=True)
        epoch_loss = 0
        for i, (seis, mag, grav, avail, target) in enumerate(loop):
            seis, mag, grav, avail, target = seis.to(device), mag.to(device), grav.to(device), avail.to(device), target.to(device)
            
            # Forward
            optimizer.zero_grad()
            out = model(seis, mag, grav, avail)
            
            # Loss: CrossEntropy expects (B, C, H, W) logits and (B, H, W) long targets
            loss = criterion(out["section"], target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss/len(loader):.6f}")
        torch.save(model.state_dict(), "checkpoint.pt")
        # --- ADD AUTOMATED VISUALIZATION HERE ---
        model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient tracking to save memory
            # 1. Get a test batch (ensure 'loader' or your test_loader is accessible)
            test_iter = iter(loader) # or use a dedicated validation loader
            inputs, mag, grav, avail, targets = next(test_iter)
            inputs, mag, grav, avail, targets = inputs.to(device), mag.to(device), grav.to(device), avail.to(device), targets.to(device)
            
            # 2. Run the model to get the AI interpretation
            outputs = model(inputs, mag, grav, avail)
            
            # 3. Save a unique comparison image for this epoch
            # Use f-string to include the epoch number so it doesn't overwrite
            save_path = f"results/epoch_{epoch+1}_comparison.png"
            
           # Example logic to match your previous results:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(inputs[0].cpu(), cmap='gray')
            ax1.set_title("Raw Seismic")
            # Target is already class indices (H, W)
            ax2.imshow(targets[0].cpu(), cmap='tab10') 
            ax2.set_title("Human Interpretation")
            # Output is (B, 6, H, W). Argmax to get class indices (H, W).
            prediction = torch.argmax(outputs["section"], dim=1)
            ax3.imshow(prediction[0].cpu(), cmap='tab10')
            ax3.set_title("AI Prediction")
            plt.savefig(save_path)
            plt.close(fig) # Essential to prevent memory leaks

            print(f"Comparison saved to {save_path}")

        model.train() # Switch back to training mode

if __name__ == "__main__":
    train()
