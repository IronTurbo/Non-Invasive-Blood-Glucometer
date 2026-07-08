import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
import glob
import os

def apply_bandpass_filter(data_array, fs, lowcut=0.5, highcut=8.0):
    nyquist = 0.5 * fs
    safe_highcut = min(highcut, nyquist - 0.1)
    b, a = butter(3, [lowcut / nyquist, safe_highcut / nyquist], btype='bandpass')
    return filtfilt(b, a, data_array)

def build_master_dataset(input_folder, output_filename):
    print(f"Scanning folder: {input_folder}...")
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    master_rows = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        subject_id = os.path.basename(file_path)
        
        bmi = df['bmi'].iloc[0]
        glucose = df['glucose'].iloc[0]
        age = df['age'].iloc[0]
        gender_raw = str(df['gender'].iloc[0]).strip().lower()
        gender_num = 1 if gender_raw in ['m', 'male'] else 0

        raw_red = df['red'].values
        raw_ir = df['ir'].values
        
        # Calculate true sampling frequency
        time_diffs = df['time'].diff().dropna()
        fs = 1.0 / time_diffs.mean() if len(time_diffs) > 0 and time_diffs.mean() != 0 else 100.0
        
        clean_red = apply_bandpass_filter(raw_red, fs=fs)
        clean_ir = apply_bandpass_filter(raw_ir, fs=fs)

        # 1. AC / DC & Ratio of Ratios
        red_dc, ir_dc = np.mean(raw_red), np.mean(raw_ir)
        red_ac, ir_ac = np.std(clean_red), np.std(clean_ir)
        ratio_r = (red_ac / red_dc) / (ir_ac / ir_dc)

        # 2. Slopes
        red_slope = np.mean(np.diff(clean_red)[np.diff(clean_red) > 0])
        ir_slope = np.mean(np.diff(clean_ir)[np.diff(clean_ir) > 0])

        # 3. NEW FEATURE: Heart Rate (BPM)
        # distance=0.5 * fs means peaks must be at least 0.5s apart (Max 120 BPM)
        peak_dist = max(1, int(0.5 * fs))
        peaks, _ = find_peaks(clean_red, distance=peak_dist)
        total_seconds = len(raw_red) / fs
        heart_rate = (len(peaks) / total_seconds) * 60 if total_seconds > 0 else 0

        # 4. NEW FEATURES: Waveform Morphology (Shape)
        red_skew = skew(clean_red)
        ir_skew = skew(clean_ir)
        
        red_kurt = kurtosis(clean_red)
        ir_kurt = kurtosis(clean_ir)

        # 5. NEW FEATURE: Signal Power (Proxy for Pulse Area)
        red_power = np.mean(clean_red ** 2)
        ir_power = np.mean(clean_ir ** 2)

        person_summary = {
            'Subject_ID': subject_id,
            'Age': age,
            'Gender': gender_num,
            'BMI': bmi,
            'Heart_Rate': heart_rate,
            'Ratio_R': ratio_r,
            'Red_Variability': red_ac,
            'IR_Variability': ir_ac,
            'Red_Slope': red_slope,
            'IR_Slope': ir_slope,
            'Red_Skew': red_skew,
            'IR_Skew': ir_skew,
            'Red_Kurtosis': red_kurt,
            'IR_Kurtosis': ir_kurt,
            'Red_Power': red_power,
            'IR_Power': ir_power,
            'Glucose': glucose 
        }
        
        master_rows.append(person_summary)

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS! Master dataset saved as '{output_filename}' with {len(master_df)} rows and {len(master_df.columns)-1} features.")

if __name__ == "__main__":
    build_master_dataset("Raw_Data", "master_features_advanced.csv")