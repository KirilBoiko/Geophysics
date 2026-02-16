import numpy as np
import segyio
from scipy.signal import ricker
import matplotlib.pyplot as plt
from geophysics.io_segy import write_segy_2d
from scipy.ndimage import gaussian_filter

def create_velocity_model(shape=(128, 512)):
    """Creates a synthetic velocity model with layers, salt, gas chimney, and reservoirs."""
    nz, nx = shape
    velocity = np.zeros(shape)
    
    # Layer 1: Water
    velocity[0:30, :] = 1500
    
    # Layer 2: Sediments (Gradient)
    for i in range(30, 80):
        velocity[i, :] = 2000 + (i - 30) * 10
        
    # Layer 3: Deeper Sediments
    for i in range(80, nz):
        velocity[i, :] = 2500 + (i - 80) * 15

    # --- Feature 1: Salt Body (High Velocity) ---
    center_x = nx // 2
    for i in range(80, nz):
        for j in range(nx):
            if (j - center_x)**2 + (i - 120)**2 < 40**2:
                 velocity[i, j] = 4500

    # --- Feature 2: Gas Chimney (Low Velocity, Chaotic) ---
    # Vertical disturbance typically caused by escaping gas
    chimney_x = nx // 4
    chimney_width = 40
    for i in range(40, nz):
        # Add random wobble to the chimney path
        wobble = int(10 * np.sin(i / 5) + np.random.randint(-5, 5))
        center = chimney_x + wobble
        
        start = max(0, center - chimney_width // 2)
        end = min(nx, center + chimney_width // 2)
        
        # Gas makes velocity drop significantly and become irregular
        velocity[i, start:end] *= 0.85 
        # Add some random noise inside the chimney to simulate chaotic scattering
        noise = np.random.randn(end - start) * 100
        velocity[i, start:end] += noise

    # --- Feature 3: Gas Deposits (Bright Spots) ---
    # High contrast low-velocity lenses trapped in sediments
    
    # Deposit A: Flat spot
    gas_z = 55
    gas_x_start = nx // 2 + 60
    gas_x_end = nx // 2 + 140
    # Create a lens shape
    for j in range(gas_x_start, gas_x_end):
        thickness = int(5 * np.sin(np.pi * (j - gas_x_start) / (gas_x_end - gas_x_start)))
        velocity[gas_z:gas_z+thickness, j] = 1600 # Very low velocity relative to surround (2250)

    # --- Feature 4: Fault ---
    fault_x = nx // 3 * 2
    shift = 20
    # Apply fault shift to the right side of the model
    velocity[:, fault_x:] = np.roll(velocity[:, fault_x:], shift, axis=0)
    
    # Smooth boundaries slightly to be realistic
    velocity = gaussian_filter(velocity, sigma=0.8)
    
    return velocity

def generate_seismic(velocity):
    """Generates synthetic seismic data using convolution."""
    # 1. Calculate Reflectivity (Impedance contrast)
    # Simple approximation: Density is constant, so just use velocity changes
    # Reflectivity R = (V2 - V1) / (V2 + V1)
    reflectivity = (velocity[1:] - velocity[:-1]) / (velocity[1:] + velocity[:-1])
    
    # Gas chimneys scatter energy, so we boost reflectivity noise in low-velocity zones
    mask_low_vel = velocity[1:] < 1800
    reflectivity[mask_low_vel] *= 1.5 # Bright spots are "bright"
    
    # Pad to match original shape
    reflectivity = np.pad(reflectivity, ((1, 0), (0, 0)), mode='constant')
    
    # 2. Create Source Wavelet (Ricker)
    points = 100
    a = 4.0
    vec2 = np.arange(0, points) - (points - 1.0) / 2
    ts = vec2 / 1000 # Time axis
    wavelet = (1.0 - 2.0 * (np.pi**2) * (a**2) * (ts**2)) * np.exp(-(np.pi**2) * (a**2) * (ts**2))
    
    # 3. Convolve Trace-by-Trace
    seismic = np.zeros_like(velocity)
    for i in range(velocity.shape[1]):
        trace_ref = reflectivity[:, i]
        trace_seis = np.convolve(trace_ref, wavelet, mode='same')
        seismic[:, i] = trace_seis
        
    # Add random noise (ocean noise / sensor noise)
    noise = np.random.randn(*seismic.shape) * 0.02 * np.max(np.abs(seismic))
    seismic += noise
    
    return seismic

if __name__ == "__main__":
    print("Generating Synthetic Earth Model with Gas Chimney & Deposits...")
    vel = create_velocity_model()
    print(f"Velocity Model Shape: {vel.shape}")
    
    print("Simulating Seismic Response...")
    seis = generate_seismic(vel)
    print(f"Seismic Data Shape: {seis.shape}")
    
    # Save to SEG-Y
    write_segy_2d("synthetic_velocity_gas.sgy", vel, dt_ms=4.0)
    write_segy_2d("synthetic_seismic_gas.sgy", seis, dt_ms=4.0)
    
    # Also generate a preview image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Velocity Model (Truth)")
    plt.imshow(vel, aspect='auto', cmap='jet')
    plt.colorbar(label='m/s')
    
    plt.subplot(1, 2, 2)
    plt.title("Seismic Response (Input)")
    plt.imshow(seis, aspect='auto', cmap='gray')
    plt.colorbar()
    
    plt.savefig("synthetic_preview.png")
    print("Saved preview to 'synthetic_preview.png'")
    
    print("Success! Saved .sgy files.")
