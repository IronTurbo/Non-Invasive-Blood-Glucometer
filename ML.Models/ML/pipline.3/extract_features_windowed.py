import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
import glob
import os

def apply_bandpass_filter(data_array, fs, lowcut=0.5, highcut=8.0):
    nyquist = 0.5 * fs
    safe_highcut = min(highcut, nyquist - 0.1)
    if lowcut >= safe_highcut:
        safe_highcut = lowcut + 0.1
    b, a = butter(3, [lowcut / nyquist, safe_highcut / nyquist], btype='bandpass')
    return filtfilt(b, a, data_array)

def extract_window_features(clean_red, clean_ir, raw_red, raw_ir, fs):
    red_dc, ir_dc = np.mean(raw_red), np.mean(raw_ir)
    red_ac, ir_ac = np.std(clean_red), np.std(clean_ir)
    ratio_r = (red_ac / red_dc) / (ir_ac / ir_dc) if (red_dc!=0 and ir_dc!=0 and ir_ac!=0) else 0

    red_slope = np.mean(np.diff(clean_red)[np.diff(clean_red) > 0])
    ir_slope = np.mean(np.diff(clean_ir)[np.diff(clean_ir) > 0])

    peak_dist = max(1, int(0.5 * fs))
    peaks, _ = find_peaks(clean_red, distance=peak_dist)
    total_seconds = len(raw_red) / fs
    heart_rate = (len(peaks) / total_seconds) * 60 if total_seconds > 0 else 0

    red_skew = skew(clean_red)
    ir_skew = skew(clean_ir)
    red_kurt = kurtosis(clean_red)
    ir_kurt = kurtosis(clean_ir)
    red_power = np.mean(clean_red ** 2)
    ir_power = np.mean(clean_ir ** 2)

    return {
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
        'IR_Power': ir_power
    }

def process_subject(file_path):
    df = pd.read_csv(file_path)
    
    # Handle too short files
    if len(df) < 2:
        return None
        
    df = df.ffill()
    
    subject_id = os.path.basename(file_path)
    bmi = df['bmi'].iloc[0] if 'bmi' in df.columns else np.nan
    glucose = df['glucose'].iloc[0] if 'glucose' in df.columns else np.nan
    age = df['age'].iloc[0] if 'age' in df.columns else np.nan
    
    gender_num = 0
    if 'gender' in df.columns:
        gender_raw = str(df['gender'].iloc[0]).strip().lower()
        gender_num = 1 if gender_raw in ['m', 'male'] else 0

    time_diffs = df['time'].diff().dropna()
    if len(time_diffs) == 0 or time_diffs.mean() == 0:
        return None
        
    fs = 1.0 / time_diffs.mean()
    
    raw_red = df['red'].values
    raw_ir = df['ir'].values
    
    try:
        clean_red = apply_bandpass_filter(raw_red, fs=fs)
        clean_ir = apply_bandpass_filter(raw_ir, fs=fs)
    except Exception as e:
        print(f"Skipping {subject_id} - filtering failed: {e}")
        return None
        
    window_samples = int(60 * fs)
    windows = []
    idx = 0
    
    while idx < len(clean_red):
        end_idx = idx + window_samples
        if end_idx <= len(clean_red):
            windows.append((idx, end_idx))
        else:
            rem_sec = (len(clean_red) - idx) / fs
            if rem_sec >= 30:
                windows.append((len(clean_red) - window_samples, len(clean_red)))
            break
        idx += window_samples
        
    if not windows:
        # If the file is entirely shorter than 1 minute but > 30s
        if len(clean_red) / fs >= 30:
            windows.append((0, len(clean_red)))
        else:
            return None
            
    features_list = []
    for w in windows:
        f = extract_window_features(
            clean_red[w[0]:w[1]], clean_ir[w[0]:w[1]],
            raw_red[w[0]:w[1]], raw_ir[w[0]:w[1]], fs
        )
        features_list.append(f)
        
    df_feat = pd.DataFrame(features_list)
    
    # MAD Outlier Rejection
    if len(df_feat) >= 3:
        mad = (df_feat - df_feat.median()).abs().median()
        threshold = 3 * mad
        threshold = threshold.replace(0, 1e-5) # Prevent dividing by zero or too strict threshold
        
        outliers = (df_feat - df_feat.median()).abs() > threshold
        invalid_windows = outliers.sum(axis=1) >= 2
        
        valid_feats = df_feat[~invalid_windows]
        
        # fallback if everything is rejected
        if len(valid_feats) == 0:
            valid_feats = df_feat
    else:
        valid_feats = df_feat
        
    # Average the valid windows
    avg_features = valid_feats.mean().to_dict()
    
    avg_features['Subject_ID'] = subject_id
    avg_features['Age'] = age
    avg_features['Gender'] = gender_num
    avg_features['BMI'] = bmi
    avg_features['Glucose'] = glucose
    
    return avg_features

def build_windowed_dataset(input_folder, output_filename):
    print(f"Scanning folder: {input_folder}...")
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    master_rows = []

    for file_path in all_files:
        res = process_subject(file_path)
        if res:
            master_rows.append(res)

    master_df = pd.DataFrame(master_rows)
    # Reorder columns to have metadata first
    cols = ['Subject_ID', 'Age', 'Gender', 'BMI', 'Glucose'] + [c for c in master_df.columns if c not in ['Subject_ID', 'Age', 'Gender', 'BMI', 'Glucose']]
    master_df = master_df[cols]
    
    master_df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS! Master dataset saved as '{output_filename}' with {len(master_df)} rows and {len(master_df.columns)-1} features.")

if __name__ == "__main__":
    build_windowed_dataset("Raw_Data", "master_features_windowed.csv")
