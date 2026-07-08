/* ============================================================
   GlucoSense — Main Application Logic
   BLE connection, signal processing, ML inference, UI
   ============================================================ */

// ============================================================
// Configuration
// ============================================================
const BLE_SERVICE_UUID    = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'; // Nordic UART Service
const BLE_TX_UUID         = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'; // Nordic UART TX (ESP32 → Phone)
const SAMPLE_RATE         = 100;   // Hz — must match your ESP32 sampling rate
const MIN_SAMPLES         = 500;   // Minimum samples before Stop is useful (~5 seconds)

// ============================================================
// State
// ============================================================
let bleDevice        = null;
let bleCharacteristic = null;
let isConnected      = false;
let isCollecting     = false;

let redSamples       = [];
let irSamples        = [];
let bleBuffer        = '';   // Buffer for incomplete BLE messages

let readingHistory   = [];

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    loadHistory();
    checkBLESupport();
});

function checkBLESupport() {
    if (!navigator.bluetooth) {
        const btn = document.getElementById('btn-connect');
        btn.textContent = '❌ Bluetooth Not Supported';
        btn.disabled = true;
        btn.title = 'Web Bluetooth API requires Chrome on Android';
    }
}

// ============================================================
// BLE Connection
// ============================================================
async function handleConnect() {
    if (isConnected) {
        disconnectBLE();
        return;
    }

    try {
        setBLEStatus('connecting');

        bleDevice = await navigator.bluetooth.requestDevice({
            filters: [{ name: 'GlucoSense_ESP32' }],
            optionalServices: [BLE_SERVICE_UUID]
        });

        bleDevice.addEventListener('gattserverdisconnected', onDisconnected);

        const server = await bleDevice.gatt.connect();
        const service = await server.getPrimaryService(BLE_SERVICE_UUID);
        bleCharacteristic = await service.getCharacteristic(BLE_TX_UUID);

        await bleCharacteristic.startNotifications();
        bleCharacteristic.addEventListener('characteristicvaluechanged', onBLEData);

        isConnected = true;
        setBLEStatus('connected');

        // Update buttons
        const btnConnect = document.getElementById('btn-connect');
        btnConnect.innerHTML = '<span class="btn-icon">🔌</span> Disconnect';
        document.getElementById('reading-controls').classList.remove('hidden');

        updateGlucoseLabel('Ready — tap Start Reading');

    } catch (error) {
        console.error('BLE connection failed:', error);
        setBLEStatus('disconnected');

        if (error.name !== 'NotFoundError') {
            updateGlucoseLabel('Connection failed. Try again.');
        }
    }
}

function disconnectBLE() {
    if (bleDevice && bleDevice.gatt.connected) {
        bleDevice.gatt.disconnect();
    }
    onDisconnected();
}

function onDisconnected() {
    isConnected = false;
    isCollecting = false;
    bleCharacteristic = null;
    setBLEStatus('disconnected');

    const btnConnect = document.getElementById('btn-connect');
    btnConnect.innerHTML = '<span class="btn-icon">📶</span> Connect Sensor';
    document.getElementById('reading-controls').classList.add('hidden');
    document.getElementById('progress-section').classList.add('hidden');

    showStartButton();
    updateGlucoseLabel('Connect sensor to begin');
}

function onBLEData(event) {
    if (!isCollecting) return;

    const value = event.target.value;
    const decoder = new TextDecoder('utf-8');
    bleBuffer += decoder.decode(value, { stream: true });

    // Parse complete lines (format: "red_value,ir_value\n")
    const lines = bleBuffer.split('\n');
    bleBuffer = lines.pop(); // Keep the incomplete last chunk

    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        const parts = trimmed.split(',');
        if (parts.length >= 2) {
            const red = parseFloat(parts[0]);
            const ir  = parseFloat(parts[1]);

            if (!isNaN(red) && !isNaN(ir)) {
                redSamples.push(red);
                irSamples.push(ir);
                updateProgress(redSamples.length);
            }
        }
    }
}

function setBLEStatus(status) {
    const el = document.getElementById('ble-status');
    el.className = 'ble-status ' + status;
    const text = el.querySelector('.ble-text');

    switch (status) {
        case 'connected':
            text.textContent = 'Connected';
            break;
        case 'connecting':
            text.textContent = 'Connecting...';
            break;
        default:
            text.textContent = 'Disconnected';
    }
}

// ============================================================
// Data Collection — Start / Stop
// ============================================================
function startReading() {
    const bmi = getCurrentBMI();
    if (bmi === null) {
        updateGlucoseLabel('⚠️ Enter your weight & height first');
        highlightProfile();
        return;
    }

    // Reset sample buffers
    redSamples = [];
    irSamples = [];
    bleBuffer = '';
    isCollecting = true;
    recordingTimer = null;
    recordingSeconds = 0;

    // Show progress, hide start, show stop
    document.getElementById('progress-section').classList.remove('hidden');
    showStopButton();
    updateProgress(0);
    
    recordingSeconds = 0;
    document.getElementById('timer-count').textContent = '0s';
    if (recordingTimer) clearInterval(recordingTimer);
    recordingTimer = setInterval(() => {
        recordingSeconds++;
        document.getElementById('timer-count').textContent = recordingSeconds + 's';
    }, 1000);
    updateGlucoseLabel('Recording... hold sensor still');

    // Reset glucose display
    document.getElementById('glucose-value').textContent = '---';
    document.getElementById('glucose-section').removeAttribute('data-range');
    document.getElementById('glucose-range').classList.add('hidden');
    setRingProgress(0);
}

function stopReading() {
    isCollecting = false;

    if (redSamples.length < MIN_SAMPLES) {
        updateGlucoseLabel(`Need at least ${MIN_SAMPLES} samples (have ${redSamples.length})`);
        // Let user continue collecting
        isCollecting = true;
        return;
    }

    // Hide progress and stop button
    showStartButton();
    updateGlucoseLabel('Processing...');
    
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }

    // Show processing animation
    document.getElementById('glucose-ring').classList.add('processing');

    // Process data asynchronously to keep UI responsive
    setTimeout(() => {
        processData();
    }, 100);
}

function showStartButton() {
    document.getElementById('btn-start').classList.remove('hidden');
    document.getElementById('btn-stop').classList.add('hidden');
}

function showStopButton() {
    document.getElementById('btn-start').classList.add('hidden');
    document.getElementById('btn-stop').classList.remove('hidden');
}

// ============================================================
// Signal Processing — Port of extract_features.py
// ============================================================

/**
 * IIR filter (direct form II transposed) — equivalent to scipy.signal.lfilter
 */
function lfilter(b, a, x) {
    const n = x.length;
    const nb = b.length;
    const na = a.length;
    const y = new Float64Array(n);

    // Normalize by a[0]
    const a0 = a[0];

    for (let i = 0; i < n; i++) {
        let val = 0;
        for (let j = 0; j < nb; j++) {
            if (i - j >= 0) val += (b[j] / a0) * x[i - j];
        }
        for (let j = 1; j < na; j++) {
            if (i - j >= 0) val -= (a[j] / a0) * y[i - j];
        }
        y[i] = val;
    }
    return y;
}

/**
 * Zero-phase digital filtering — equivalent to scipy.signal.filtfilt (simplified)
 * Applies the filter forward, then backward, to eliminate phase distortion.
 */
function filtfilt(b, a, x) {
    const n = x.length;

    // Edge padding to reduce transient effects (3 * max filter order)
    const padLen = Math.min(3 * Math.max(b.length, a.length), n - 1);

    // Pad signal by reflecting at edges
    const padded = new Float64Array(n + 2 * padLen);
    for (let i = 0; i < padLen; i++) {
        padded[i] = 2 * x[0] - x[padLen - i];
    }
    for (let i = 0; i < n; i++) {
        padded[padLen + i] = x[i];
    }
    for (let i = 0; i < padLen; i++) {
        padded[padLen + n + i] = 2 * x[n - 1] - x[n - 2 - i];
    }

    // Forward pass
    let forward = lfilter(b, a, padded);

    // Reverse
    const reversed = new Float64Array(forward.length);
    for (let i = 0; i < forward.length; i++) {
        reversed[i] = forward[forward.length - 1 - i];
    }

    // Backward pass
    let backward = lfilter(b, a, reversed);

    // Reverse again
    const result = new Float64Array(backward.length);
    for (let i = 0; i < backward.length; i++) {
        result[i] = backward[backward.length - 1 - i];
    }

    // Remove padding
    const output = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        output[i] = result[padLen + i];
    }
    return output;
}

/**
 * Compute the mean of an array
 */
function mean(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    return sum / arr.length;
}

/**
 * Compute the standard deviation (population, ddof=0) — matches numpy.std()
 */
function std(arr) {
    const m = mean(arr);
    let sumSq = 0;
    for (let i = 0; i < arr.length; i++) {
        const d = arr[i] - m;
        sumSq += d * d;
    }
    return Math.sqrt(sumSq / arr.length);
}

/**
 * Compute skewness (biased) — matches scipy.stats.skew(bias=True)
 * skew = m3 / m2^(3/2)
 * where m2 and m3 are the 2nd and 3rd central moments (biased, i.e. dividing by N)
 */
function skewness(arr) {
    const n = arr.length;
    const m = mean(arr);

    let m2 = 0, m3 = 0;
    for (let i = 0; i < n; i++) {
        const d = arr[i] - m;
        const d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    m2 /= n;
    m3 /= n;

    if (m2 === 0) return 0;
    return m3 / Math.pow(m2, 1.5);
}

/**
 * Compute first differences of an array — equivalent to numpy.diff()
 */
function diff(arr) {
    const result = new Float64Array(arr.length - 1);
    for (let i = 0; i < result.length; i++) {
        result[i] = arr[i + 1] - arr[i];
    }
    return result;
}

/**
 * Mean of only the positive values — equivalent to np.mean(d[d > 0])
 */
function meanPositive(arr) {
    let sum = 0, count = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > 0) {
            sum += arr[i];
            count++;
        }
    }
    return count > 0 ? sum / count : 0;
}

/**
 * Compute all features from raw PPG data.
 * Matches extract_features.py logic, returning only the 6 features
 * used by the XGBoost model.
 *
 * @param {Float64Array|number[]} rawRed - Raw red PPG values
 * @param {Float64Array|number[]} rawIr  - Raw IR PPG values
 * @param {number} bmi - User's BMI
 * @returns {number[]} Feature array in model order:
 *   [BMI, Ratio_R, Red_Slope, IR_Slope, Red_Skew, IR_Skew]
 */
function computeFeatures(rawRed, rawIr, bmi) {
    // Apply bandpass filter (coefficients from glucose_model.js)
    const cleanRed = filtfilt(FILTER_B, FILTER_A, rawRed);
    const cleanIr  = filtfilt(FILTER_B, FILTER_A, rawIr);

    // 1. AC/DC & Ratio of Ratios
    const redDC  = mean(rawRed);
    const irDC   = mean(rawIr);
    const redAC  = std(cleanRed);
    const irAC   = std(cleanIr);
    const ratioR = (redAC / redDC) / (irAC / irDC);

    // 2. Slopes (mean of positive differences)
    const redDiff  = diff(cleanRed);
    const irDiff   = diff(cleanIr);
    const redSlope = meanPositive(redDiff);
    const irSlope  = meanPositive(irDiff);

    // 3. Skewness
    const redSkew = skewness(Array.from(cleanRed));
    const irSkew  = skewness(Array.from(cleanIr));

    // Return the 6 features in the exact XGBoost training order
    // Scaling is handled inside calculateGlucose (glucose_model.js)
    return [bmi, ratioR, redSlope, irSlope, redSkew, irSkew];
}

// ============================================================
// Data Processing & Prediction
// ============================================================
function processData() {
    try {
        const bmi = getCurrentBMI();
        const rawRed = new Float64Array(redSamples);
        const rawIr  = new Float64Array(irSamples);

        console.log(`Processing ${rawRed.length} samples...`);

        // Compute features
        const features = computeFeatures(rawRed, rawIr, bmi);
        console.log('Features:', features);

        // Run model inference
        const glucose = calculateGlucose(features);
        console.log('Predicted glucose:', glucose);

        // Clamp to reasonable range
        const clampedGlucose = Math.max(30, Math.min(400, glucose));

        // Update UI
        displayGlucose(clampedGlucose);
        addToHistory(clampedGlucose, redSamples.length);

    } catch (error) {
        console.error('Processing error:', error);
        updateGlucoseLabel('⚠️ Error: ' + error.message);
    } finally {
        document.getElementById('glucose-ring').classList.remove('processing');
        document.getElementById('progress-section').classList.add('hidden');
    }
}

// ============================================================
// UI — Glucose Display
// ============================================================
function displayGlucose(value) {
    const rounded = Math.round(value);
    const el = document.getElementById('glucose-value');
    const card = document.getElementById('glucose-section');
    const rangeEl = document.getElementById('glucose-range');
    const rangeText = document.getElementById('range-text');

    // Set range data
    // Removed diagnosis label
    rangeEl.classList.add('hidden');

    // Animate the value
    el.textContent = rounded;
    el.classList.remove('pop');
    void el.offsetWidth; // Force reflow
    el.classList.add('pop');

    // Animate ring to full
    setRingProgress(1);

    updateGlucoseLabel(`Measured from ${redSamples.length} samples`);
}

function setRingProgress(fraction) {
    const circumference = 2 * Math.PI * 90; // r=90
    const offset = circumference * (1 - fraction);
    document.getElementById('ring-progress').style.strokeDashoffset = offset;
}

function updateGlucoseLabel(text) {
    document.getElementById('glucose-label').textContent = text;
}

function updateProgress(sampleCount) {
    document.getElementById('sample-count').textContent = `${sampleCount} samples`;

    // Use a soft target for progress bar visualization (e.g., ~3000 samples = 30s at 100Hz)
    const target = 3000;
    const pct = Math.min(100, (sampleCount / target) * 100);
    document.getElementById('progress-bar').style.width = pct + '%';

    // Enable stop button only after minimum samples
    const btnStop = document.getElementById('btn-stop');
    if (sampleCount >= MIN_SAMPLES) {
        btnStop.disabled = false;
    } else {
        btnStop.disabled = true;
    }
}

// ============================================================
// UI — Profile (Weight / Height / BMI)
// ============================================================
function updateBMI() {
    const weight = parseFloat(document.getElementById('input-weight').value);
    const height = parseFloat(document.getElementById('input-height').value);
    const bmiEl = document.getElementById('bmi-value');

    if (weight > 0 && height > 0) {
        const heightM = height / 100;
        const bmi = weight / (heightM * heightM);
        bmiEl.textContent = bmi.toFixed(1);
        saveProfile(weight, height);
    } else {
        bmiEl.textContent = '—';
    }
}

function getCurrentBMI() {
    const weight = parseFloat(document.getElementById('input-weight').value);
    const height = parseFloat(document.getElementById('input-height').value);

    if (weight > 0 && height > 0) {
        const heightM = height / 100;
        return weight / (heightM * heightM);
    }
    return null;
}

function toggleProfile() {
    const body = document.getElementById('profile-body');
    const arrow = document.getElementById('profile-toggle');
    body.classList.toggle('collapsed');
    arrow.classList.toggle('collapsed');
}

function highlightProfile() {
    const section = document.getElementById('profile-section');
    const body = document.getElementById('profile-body');
    const arrow = document.getElementById('profile-toggle');

    // Make sure it's expanded
    body.classList.remove('collapsed');
    arrow.classList.remove('collapsed');

    // Flash highlight
    section.style.borderColor = 'var(--glucose-pre)';
    section.style.boxShadow = '0 0 20px rgba(255, 209, 102, 0.2)';
    setTimeout(() => {
        section.style.borderColor = '';
        section.style.boxShadow = '';
    }, 2000);

    // Scroll to it
    section.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function saveProfile(weight, height) {
    localStorage.setItem('glucosense_weight', weight);
    localStorage.setItem('glucosense_height', height);
}

function loadProfile() {
    const weight = localStorage.getItem('glucosense_weight');
    const height = localStorage.getItem('glucosense_height');

    if (weight) document.getElementById('input-weight').value = weight;
    if (height) document.getElementById('input-height').value = height;

    if (weight && height) {
        updateBMI();
    }
}

// ============================================================
// UI — History
// ============================================================
function addToHistory(glucose, sampleCount) {
    const entry = {
        glucose: Math.round(glucose),
        samples: sampleCount,
        timestamp: Date.now()
    };

    readingHistory.unshift(entry);

    // Keep last 50 readings
    if (readingHistory.length > 50) readingHistory.pop();

    localStorage.setItem('glucosense_history', JSON.stringify(readingHistory));
    renderHistory();
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('glucosense_history');
        if (saved) {
            readingHistory = JSON.parse(saved);
            renderHistory();
        }
    } catch (e) {
        readingHistory = [];
    }
}

function clearHistory() {
    readingHistory = [];
    localStorage.removeItem('glucosense_history');
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('history-list');

    if (readingHistory.length === 0) {
        list.innerHTML = '<p class="history-empty">No readings yet. Connect your sensor to start!</p>';
        return;
    }

    list.innerHTML = readingHistory.map(entry => {
        const color = getGlucoseColor(entry.glucose);
        const time = formatTime(entry.timestamp);
        return `
            <div class="history-item">
                <div class="history-item-left">
                    <span class="history-dot" style="background: ${color}"></span>
                    <span class="history-glucose" style="color: ${color}">
                        ${entry.glucose}<span class="history-glucose-unit">mg/dL</span>
                    </span>
                </div>
                <div>
                    <span class="history-time">${time}</span>
                    <span class="history-samples">${entry.samples} samples</span>
                </div>
            </div>
        `;
    }).join('');
}

function getGlucoseColor(value) {
    if (value < 70) return 'var(--glucose-low)';
    if (value <= 99) return 'var(--glucose-normal)';
    if (value <= 125) return 'var(--glucose-pre)';
    if (value <= 180) return 'var(--glucose-high)';
    return 'var(--glucose-very-high)';
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();

    const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (isToday) return time;
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + time;
}

// ============================================================
// Service Worker Registration
// ============================================================
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('sw.js').catch(err => {
            console.log('Service Worker registration skipped:', err);
        });
    });
}
