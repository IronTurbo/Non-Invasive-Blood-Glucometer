import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import os
import glob
from scipy.stats import skew

def apply_bandpass_filter(data_array, fs, lowcut=0.5, highcut=4.0):
    nyquist = 0.5 * fs
    b, a = butter(3, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
    return filtfilt(b, a, data_array)

def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    
    # Forward-fill metadata (like age, bmi, gender) which only exists on row 0
    df = df.ffill()
    
    # Handle files with less than 2 rows gracefully
    if len(df) < 2:
        print(f"Skipping {file_path} - not enough data")
        return
        
    # Calculate true sampling frequency
    time_diffs = df['time'].diff().dropna()
    if len(time_diffs) == 0 or time_diffs.mean() == 0:
        print(f"Skipping {file_path} - invalid time column")
        return
        
    fs = 1.0 / time_diffs.mean()
    # print(f"Processing {os.path.basename(file_path)} with fs = {fs:.2f} Hz")
    
    # Filter only internally for evaluating quality
    raw_red = df['red'].values
    try:
        clean_red = apply_bandpass_filter(raw_red, fs=fs)
    except Exception as e:
        print(f"Skipping {file_path} - filtering failed: {e}")
        return
    
    window_length = int(60 * fs) # 1 minute
    step = int(10 * fs) # 10 seconds step
    
    # If the file is less than 1 minute, save the whole file
    if len(df) <= window_length:
        print(f"File {os.path.basename(file_path)} is shorter than 1 minute. Saving entirely.")
        df.to_csv(os.path.join(output_dir, os.path.basename(file_path)), index=False)
        return
        
    best_score = -np.inf
    best_start = 0
    
    # Distance between peaks should correspond to roughly max 180 BPM = 3 Hz -> 1/3 sec distance
    peak_distance = max(1, int(fs / 3.0)) 
    
    for i in range(0, len(clean_red) - window_length + 1, step):
        window = clean_red[i:i+window_length]
        
        # Calculate skewness (good PPG signals have sharp systolic peaks -> positive skewness)
        s = skew(window)
        
        # Check Heart Rate plausibility
        peaks, _ = find_peaks(window, distance=peak_distance)
        heart_rate = len(peaks) # Since window is exactly 1 minute, len(peaks) is the BPM
        
        if 40 <= heart_rate <= 180:
            score = s
            if score > best_score:
                best_score = score
                best_start = i

    # If no window had a valid heart rate, just pick the one with max skewness
    if best_score == -np.inf:
        print(f"Warning: No window with valid HR found for {os.path.basename(file_path)}. Picking highest skewness.")
        for i in range(0, len(clean_red) - window_length + 1, step):
            window = clean_red[i:i+window_length]
            s = skew(window)
            if s > best_score:
                best_score = s
                best_start = i
                
    # Crop the ORIGINAL RAW DATAFRAME
    best_df = df.iloc[best_start : best_start + window_length]
    
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    best_df.to_csv(output_path, index=False)
    # print(f"Saved best 1-min window for {os.path.basename(file_path)}")

def main():
    input_dir = "Raw_Data"
    output_dir = "Processed_Data_1Min"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(all_files)} files. Starting extraction...")
    
    for f in all_files:
        process_file(f, output_dir)
        
    print(f"Extraction complete! Best 1-minute files saved to '{output_dir}'.")

if __name__ == '__main__':
    main()
