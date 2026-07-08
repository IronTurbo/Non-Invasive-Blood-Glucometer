"""
Convert XGBoost model + StandardScaler to a self-contained glucose_model.js using m2cgen.
Handles the base_score=NaN issue in newer XGBoost versions with m2cgen.

Features (in order): ['BMI', 'Ratio_R', 'Red_Slope', 'IR_Slope', 'Red_Skew', 'IR_Skew']
"""

import joblib
import json
import re
import numpy as np
from scipy.signal import butter
import m2cgen as m2c

# ── Load artefacts ──────────────────────────────────────────────────────────
MODEL_PATH  = "XGBoost/XGBoost.joblib"
SCALER_PATH = "XGBoost/scaler_xgb.joblib"
OUTPUT_JS   = "glucose-app-0/glucose_model.js"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"Model type  : {type(model)}")
print(f"Scaler type : {type(scaler)}")
print(f"Scaler mean : {scaler.mean_.tolist()}")
print(f"Scaler scale: {scaler.scale_.tolist()}")

# ── Get the real base score from the booster config ─────────────────────────
booster = model.get_booster()
cfg = json.loads(booster.save_config())
try:
    raw_base = cfg["learner"]["learner_model_param"]["base_score"]
    # XGBoost 2.x stores it as '[1.1304348E2]' — strip brackets and parse
    clean = str(raw_base).strip().strip('[]')
    base_score = float(clean)
    if np.isnan(base_score):
        raise ValueError("NaN base score")
    print(f"Base score from config: {base_score}")
except Exception as e:
    print(f"Warning: could not get base score ({e}), using 0.5 fallback")
    base_score = 0.5


# ── Generate JavaScript for the XGBoost model alone ─────────────────────────
raw_js = m2c.export_to_javascript(model)

# Rename the entry function to _xgbScore (internal)
raw_js = raw_js.replace("function score(", "function _xgbScore(")

# Replace the 'nan' literal with the actual base score
# m2cgen outputs: return nan + (var0 + var1 + ...)
raw_js = re.sub(r'\bnan\b', str(base_score), raw_js)

print(f"Replaced 'nan' with {base_score} in generated JS.")

# ── Bandpass filter coefficients ─────────────────────────────────────────────
b, a = butter(3, [0.5 / 50.0, 8.0 / 50.0], btype='bandpass')

# ── Scaler parameters ────────────────────────────────────────────────────────
scaler_mean  = scaler.mean_.tolist()
scaler_scale = scaler.scale_.tolist()

feature_names = ['BMI', 'Ratio_R', 'Red_Slope', 'IR_Slope', 'Red_Skew', 'IR_Skew']

# ── Assemble final JS ─────────────────────────────────────────────────────────
wrapper = f"""// ============================================================
// AUTO-GENERATED via m2cgen — XGBoost Regressor -> JavaScript
// Feature order: {feature_names}
// ============================================================

// Bandpass filter coefficients (Butterworth order-3, 0.5-8 Hz at 100 Hz)
const FILTER_B = {json.dumps(b.tolist())};
const FILTER_A = {json.dumps(a.tolist())};

// StandardScaler parameters
const SCALER_MEAN  = {json.dumps(scaler_mean)};
const SCALER_SCALE = {json.dumps(scaler_scale)};

{raw_js}

/**
 * Public entry point — takes RAW (unscaled) features, scales them,
 * then runs the XGBoost model.
 * @param {{number[]}} rawFeatures - [{", ".join(feature_names)}]
 * @returns {{number}} Predicted glucose in mg/dL
 */
function calculateGlucose(rawFeatures) {{
    const scaled = rawFeatures.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);
    return _xgbScore(scaled);
}}
"""

with open(OUTPUT_JS, "w") as f:
    f.write(wrapper.strip() + "\n")

print(f"\nSUCCESS! Wrote {OUTPUT_JS}  ({len(wrapper):,} bytes)")

# ── Sanity check ────────────────────────────────────────────────────────────
sample = np.array([[25.0, 0.95, 12.0, 15.0, 0.1, -0.2]])
scaled = scaler.transform(sample)
py_pred = model.predict(scaled)[0]
print(f"Python prediction for sample {sample.tolist()}: {py_pred:.4f} mg/dL")
print("Verify the browser JS output matches this value.")
