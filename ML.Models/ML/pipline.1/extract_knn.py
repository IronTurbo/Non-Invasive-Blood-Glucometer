import joblib
import json

try:
    knn = joblib.load('KNN/KNN.joblib')
    scaler = joblib.load('KNN/scaler_knn.joblib')
    
    # KNN parameters
    k = knn.n_neighbors
    X = knn._fit_X.tolist()
    y = knn._y.tolist()
    
    # Scaler parameters
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()

    js_code = f"""// Auto-generated KNN model data
const KNN_K = {k};
const SCALER_MEAN = {json.dumps(mean)};
const SCALER_SCALE = {json.dumps(scale)};
const KNN_X = {json.dumps(X)};
const KNN_Y = {json.dumps(y)};

function calculateGlucose(featuresRaw) {{
    // 1. StandardScaler normalization
    const scaled = featuresRaw.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);

    // 2. KNN inference
    // Calculate Euclidean distances
    const distances = [];
    for (let i = 0; i < KNN_X.length; i++) {{
        let sumSq = 0;
        for (let j = 0; j < scaled.length; j++) {{
            const diff = scaled[j] - KNN_X[i][j];
            sumSq += diff * diff;
        }}
        distances.push({{ dist: sumSq, y: KNN_Y[i] }});
    }}

    // Sort by distance (ascending)
    distances.sort((a, b) => a.dist - b.dist);

    // Get top K neighbors
    let sum = 0;
    for (let i = 0; i < KNN_K; i++) {{
        sum += distances[i].y;
    }}

    // Return average of K neighbors
    return sum / KNN_K;
}}
"""
    with open('glucose-app-4th/glucose_model.js', 'w') as f:
        f.write(js_code)
    print("Successfully generated glucose_model.js")
except Exception as e:
    print(f"Error: {e}")
