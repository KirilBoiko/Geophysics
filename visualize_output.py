import segyio
import matplotlib.pyplot as plt
import numpy as np
import sys

def visualize_segy(path, out_img):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        data = np.stack([f.trace[i] for i in range(len(f.trace))], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data.T, cmap='seismic', aspect='auto', interpolation='none')
    plt.colorbar(label='Amplitude')
    plt.title(f"SEG-Y Visualization: {path}")
    plt.xlabel("Trace Number")
    plt.ylabel("Sample Index")
    plt.savefig(out_img)
    print(f"Saved visualization to {out_img}")

if __name__ == "__main__":
    visualize_segy(sys.argv[1], sys.argv[2])
