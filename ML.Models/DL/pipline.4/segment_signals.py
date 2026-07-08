import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import glob
import os

def segment_dataset(input_folder):
    print("Beginning 1-Second Peak-Centered Segmentation with NeuroKit2...")
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    X_signals = [] # Will hold the raw 100-point wave segments
    X_bmi = []     # Will hold the matching BMI
    y_glucose = [] # Will hold the target glucose
    
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            
            bmi = df['bmi'].iloc[0]
            glucose = df['glucose'].iloc[0]
            
            # 1. Clean both channels using NeuroKit2
            # The 'sampling_rate=100' tells the algorithm to expect your ESP32 speed
            clean_ir = nk.ppg_clean(df['ir'].values, sampling_rate=100)
            clean_red = nk.ppg_clean(df['red'].values, sampling_rate=100)
            
            # 2. Find peaks based on the IR channel (IR penetrates deeper, usually cleaner)
            # distance=60 means peaks must be at least 0.6 seconds apart
            peaks, _ = find_peaks(clean_ir, distance=60)
            
            # 3. Extract 1-second Windows
            for peak in peaks:
                # Make sure the peak isn't too close to the very beginning or end of the file
                if peak - 50 >= 0 and peak + 50 < len(clean_ir):
                    
                    # Extract 50 points before the peak, and 50 points after (100 points total)
                    slice_ir = clean_ir[peak-50 : peak+50]
                    slice_red = clean_red[peak-50 : peak+50]
                    
                    # Normalize the waves between 0 and 1 so brightness doesn't confuse the AI
                    slice_ir = (slice_ir - np.min(slice_ir)) / (np.max(slice_ir) - np.min(slice_ir) + 1e-6)
                    slice_red = (slice_red - np.min(slice_red)) / (np.max(slice_red) - np.min(slice_red) + 1e-6)
                    
                    # Stack them together into a single 2-Channel array
                    combined_signal = np.stack([slice_ir, slice_red], axis=-1)
                    
                    X_signals.append(combined_signal)
                    X_bmi.append(bmi)
                    y_glucose.append(glucose)
                    
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
                
    # Convert to pure Numpy Arrays for Deep Learning
    X_signals = np.array(X_signals)
    X_bmi = np.array(X_bmi)
    y_glucose = np.array(y_glucose)
    
    print(f"Generated {X_signals.shape[0]} unique heartbeat segments!")
    return X_signals, X_bmi, y_glucose

# ==========================================
# VISUALIZATION FUNCTION
# ==========================================
def visualize_segments(X_signals, X_bmi, y_glucose, num_samples=3):
    """
    This function draws 3 random heartbeat segments so you can see the 
    matrix shape and ensure the Red/IR channels look correct.
    """
    print("\nOpening visualization window... (Close the window to end the script)")
    
    plt.figure(figsize=(15, 5))
    
    # Pick 3 random segments from our massive dataset
    random_indices = np.random.choice(len(X_signals), num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_samples, i+1)
        
        # Plot IR (channel 0) and Red (channel 1)
        plt.plot(X_signals[idx, :, 0], label="IR Channel", color="purple", linewidth=2)
        plt.plot(X_signals[idx, :, 1], label="Red Channel", color="red", linewidth=2, linestyle="--")
        
        plt.title(f"Segment ID: {idx}\nBMI: {X_bmi[idx]} | Glucose: {y_glucose[idx]}")
        plt.xlabel("Time Steps (100 = 1 Second)")
        plt.ylabel("Normalized Amplitude (0 to 1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

# ==========================================
# RUN THE PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Extract the Data
    X_sig, X_b, y = segment_dataset("Raw_Data")
    
    # 2. Save the Deep Learning Matrices to your hard drive
    np.save("X_signals.npy", X_sig)
    np.save("X_bmi.npy", X_b)
    np.save("y_glucose.npy", y)
    print("Saved matrix segments safely to disk as .npy files.")
    
    # 3. Visualize the results
    if len(X_sig) > 0:
        visualize_segments(X_sig, X_b, y, num_samples=3)