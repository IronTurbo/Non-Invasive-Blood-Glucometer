import os
import sys
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RAW_DATA_DIR   = Path("Raw_Data")
OUTPUT_DIR     = Path("Processed_Data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Signal parameters
FS              = 13.0        # Nominal sampling frequency (Hz)
BANDPASS_LOW    = 0.5         # Hz
BANDPASS_HIGH   = 4.0         # Hz
BANDPASS_ORDER  = 4           

# Windowing
WINDOW_SEC      = 10.0        # Window length in seconds
OVERLAP_RATIO   = 0.5         # 50% overlap
MIN_WINDOW_FILL = 0.8         # Require 80% non-artifact samples

# Artifact detection
SQI_AMPLITUDE_SIGMA = 4.0     
SQI_FLATNESS_THRESH = 10      

N_JOBS = -1

print("=" * 65)
print("  PPG-to-Glucose Dataset Preparation Pipeline (DL Only)")
print("=" * 65)

# =============================================================================
# STAGE 1: DATA LOADING
# =============================================================================
def load_subject(csv_path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    meta_row = df.iloc[0]
    metadata = {
        'subject_id': csv_path.stem,
        'glucose'   : float(meta_row['glucose']) if pd.notna(meta_row.get('glucose', np.nan)) else np.nan,
    }

    signal_df = df[['time', 'red', 'ir']].dropna(subset=['time']).copy()
    signal_df = signal_df.reset_index(drop=True)
    signal_df['time'] = pd.to_numeric(signal_df['time'], errors='coerce')
    signal_df['red']  = pd.to_numeric(signal_df['red'],  errors='coerce')
    signal_df['ir']   = pd.to_numeric(signal_df['ir'],   errors='coerce')
    signal_df.dropna(inplace=True)

    return metadata, signal_df

def estimate_fs(time_array: np.ndarray) -> float:
    diffs = np.diff(time_array)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return FS
    median_dt = np.median(diffs)
    return float(1.0 / median_dt)

# =============================================================================
# STAGE 2: SIGNAL PREPROCESSING (Filter Only)
# =============================================================================
def bandpass_filter(sig: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    low_n  = max(BANDPASS_LOW  / nyq, 1e-3)
    high_n = min(BANDPASS_HIGH / nyq, 0.99)
    if low_n >= high_n:
        return sig.copy()
    b, a = butter(BANDPASS_ORDER, [low_n, high_n], btype='band')
    return filtfilt(b, a, sig)

def preprocess_signal(sig: np.ndarray, fs: float) -> np.ndarray:
    """Only applies bandpass filtering as requested."""
    sig = np.array(sig, dtype=float)
    return bandpass_filter(sig, fs)

# =============================================================================
# STAGE 3: ARTIFACT DETECTION
# =============================================================================
def compute_sqi_mask(sig: np.ndarray) -> np.ndarray:
    mask = np.ones(len(sig), dtype=bool)
    if len(sig) == 0:
        return mask

    med  = np.median(sig)
    std_ = np.std(sig)

    if std_ < 1e-9:
        return np.zeros(len(sig), dtype=bool)

    high = med + SQI_AMPLITUDE_SIGMA * std_
    low  = med - SQI_AMPLITUDE_SIGMA * std_
    mask &= (sig >= low) & (sig <= high)
    return mask

def sqi_ratio(mask: np.ndarray) -> float:
    if len(mask) == 0: return 0.0
    return float(mask.sum()) / len(mask)

# =============================================================================
# STAGE 4: WINDOWING
# =============================================================================
def create_windows(sig_ir: np.ndarray, sig_red: np.ndarray, sqi_mask: np.ndarray, fs: float) -> list[dict]:
    win_len = int(WINDOW_SEC * fs)
    hop_len = int(win_len * (1 - OVERLAP_RATIO))
    if hop_len < 1: hop_len = 1
    
    if win_len > len(sig_ir):
        if sqi_ratio(sqi_mask) >= MIN_WINDOW_FILL:
            return [{'ir': sig_ir, 'red': sig_red}]
        return []

    windows = []
    start = 0
    while start + win_len <= len(sig_ir):
        w_ir   = sig_ir[start:start + win_len]
        w_red  = sig_red[start:start + win_len]
        w_mask = sqi_mask[start:start + win_len]
        if sqi_ratio(w_mask) >= MIN_WINDOW_FILL:
            windows.append({'ir': w_ir, 'red': w_red})
        start += hop_len
    return windows

# =============================================================================
# PER-SUBJECT PROCESSING
# =============================================================================
def process_subject(csv_path: Path) -> tuple[list[np.ndarray], float, str]:
    try:
        metadata, signal_df = load_subject(csv_path)
        subject_id = metadata['subject_id']

        if len(signal_df) < 50:
            return [], np.nan, subject_id

        fs = estimate_fs(signal_df['time'].values)
        raw_ir  = signal_df['ir'].values.astype(float)
        raw_red = signal_df['red'].values.astype(float)

        filt_ir  = preprocess_signal(raw_ir,  fs)
        filt_red = preprocess_signal(raw_red, fs)

        sqi_ir   = compute_sqi_mask(filt_ir)
        sqi_red  = compute_sqi_mask(filt_red)
        sqi_mask = sqi_ir & sqi_red

        windows = create_windows(filt_ir, filt_red, sqi_mask, fs)
        if not windows:
            return [], np.nan, subject_id

        sequences = []
        for w in windows:
            seq_ir  = w['ir'].copy()
            seq_red = w['red'].copy()
            
            ir_range  = seq_ir.max()  - seq_ir.min()
            red_range = seq_red.max() - seq_red.min()
            if ir_range  > 1e-9: seq_ir  = (seq_ir  - seq_ir.min())  / ir_range
            if red_range > 1e-9: seq_red = (seq_red - seq_red.min()) / red_range

            seq = np.stack([seq_ir, seq_red], axis=-1)
            
            tgt_len = int(WINDOW_SEC * round(fs))
            if seq.shape[0] < tgt_len:
                seq = np.pad(seq, ((0, tgt_len - seq.shape[0]), (0, 0)), 'edge')
            else:
                seq = seq[:tgt_len]
                
            sequences.append(seq)

        return sequences, metadata.get('glucose', np.nan), subject_id

    except Exception as e:
        sid = csv_path.stem
        print(f"  [ERROR] {sid}: {e}")
        return [], np.nan, sid

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    csv_files = sorted(RAW_DATA_DIR.glob("S*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {RAW_DATA_DIR.resolve()}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} subject files. Starting processing...\n")

    results = Parallel(n_jobs=N_JOBS, prefer='threads')(
        delayed(process_subject)(csv_path)
        for csv_path in tqdm(csv_files, desc="Processing subjects", unit="subj")
    )

    all_sequences = []
    all_seq_labels = []
    all_subject_ids_seq = []

    for sequences, glucose, subject_id in results:
        if sequences and pd.notna(glucose):
            all_sequences.extend(sequences)
            all_seq_labels.extend([glucose] * len(sequences))
            all_subject_ids_seq.extend([subject_id] * len(sequences))

    if not all_sequences:
        print("[ERROR] No valid sequences generated. Check your data.")
        sys.exit(1)

    print("\n[SAVE] Building DL sequences NPZ...")
    max_len = max(s.shape[0] for s in all_sequences)
    n_ch    = all_sequences[0].shape[1]
    padded  = np.zeros((len(all_sequences), max_len, n_ch), dtype=np.float32)
    for i, s in enumerate(all_sequences):
        padded[i, :s.shape[0], :] = s

    labels   = np.array(all_seq_labels, dtype=np.float32)
    subj_ids = np.array(all_subject_ids_seq, dtype=str)

    npz_path = OUTPUT_DIR / "dataset_sequences.npz"
    np.savez_compressed(
        npz_path,
        X           = padded,    # [n_windows, timesteps, 2]
        y           = labels,    # [n_windows]  glucose values
        subject_ids = subj_ids   # [n_windows]  subject IDs
    )
    print(f"[SAVED] {npz_path}")
    print(f"        X shape: {padded.shape}  (windows × timesteps × channels)")
    print(f"        y shape: {labels.shape}  (glucose labels)")
    print("\n[OK] Deep Learning Dataset preparation complete!")

if __name__ == "__main__":
    main()
