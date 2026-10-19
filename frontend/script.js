/**
 * Belt RUL Predictor — Main Application Script
 * ==============================================
 * Handles all UI interactions, API communication, chart rendering
 * and the inspection loop with progressive confidence scoring.
 */

const API = (window.location.origin && (window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')))
  ? window.location.origin
  : 'http://localhost:8001';

// ── State ──────────────────────────────────────────────────────────────────
let selectedDatasetPath = null;
let modelLoaded         = false;
let beltProperties      = {};
let inspections         = [];      // accumulated inspection records
let rulChart            = null;    // Chart.js instance

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadDatasets();
  checkModelStatus();
  setupDragDrop();
  setupInputListeners();
});

// ── Step navigation ────────────────────────────────────────────────────────
function goToStep(n) {
  for (let i = 1; i <= 4; i++) {
    document.getElementById(`panel-${i}`)?.classList.add('hidden');
    const nav = document.getElementById(`step-nav-${i}`);
    if (nav) {
      nav.classList.remove('active', 'done');
      if (i < n) nav.classList.add('done');
    }
  }
  const panel = document.getElementById(`panel-${n}`);
  if (panel) panel.classList.remove('hidden');
  const nav = document.getElementById(`step-nav-${n}`);
  if (nav) nav.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Model Status ───────────────────────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res = await fetch(`${API}/metadata`);
    const data = await res.json();
    if (data.metrics) {
      setModelStatus('ready', `Model ready — R² ${data.metrics.ensemble?.r2_score ?? '—'}`);
      modelLoaded = true;
    }
  } catch {
    setModelStatus('idle', 'No model loaded');
  }
}

function setModelStatus(type, text) {
  const badge = document.getElementById('model-status');
  const label = document.getElementById('model-status-text');
  badge.className = `status-badge status-${type}`;
  label.textContent = text;
}

// ── Datasets ───────────────────────────────────────────────────────────────
async function loadDatasets() {
  const list = document.getElementById('dataset-list');
  list.innerHTML = '<div class="empty-state">Loading…</div>';
  try {
    const res  = await fetch(`${API}/datasets`);
    const data = await res.json();
    if (!data.datasets || data.datasets.length === 0) {
      list.innerHTML = '<div class="empty-state">No CSV datasets found. Generate or upload one above.</div>';
      return;
    }
    list.innerHTML = '';
    data.datasets.forEach(ds => {
      const item = document.createElement('div');
      item.className = 'dataset-item';
      item.dataset.path = ds.path;
      // Auto-select the canonical training file
      if (ds.filename === 'belt_rul_training_data.csv') {
        item.classList.add('selected');
        selectedDatasetPath = ds.path;
        updateSelectedDisplay(ds.filename);
        document.getElementById('btn-train').disabled = false;
      }
      item.innerHTML = `
        <div class="dataset-icon">📊</div>
        <div>
          <div class="dataset-name">${ds.filename}</div>
          <div class="dataset-meta">${ds.size_mb} MB — ${ds.columns.length} columns</div>
        </div>
        <div class="dataset-check">✓</div>
      `;
      item.addEventListener('click', () => selectDataset(item, ds));
      list.appendChild(item);
    });
  } catch (e) {
    list.innerHTML = `<div class="empty-state" style="color:var(--red)">Could not reach API. Is the server running?</div>`;
  }
}

function selectDataset(item, ds) {
  document.querySelectorAll('.dataset-item').forEach(el => el.classList.remove('selected'));
  item.classList.add('selected');
  selectedDatasetPath = ds.path;
  updateSelectedDisplay(ds.filename);
  document.getElementById('btn-train').disabled = false;
}

function updateSelectedDisplay(name) {
  document.getElementById('selected-dataset-display').textContent = `Selected: ${name}`;
}

// ── Generate Data ──────────────────────────────────────────────────────────
async function generateData() {
  const belts    = parseInt(document.getElementById('gen-belts').value)    || 300;
  const maxHours = parseInt(document.getElementById('gen-max-hours').value) || 20000;
  const step     = parseInt(document.getElementById('gen-step').value)      || 100;
  const seed     = parseInt(document.getElementById('gen-seed').value)      || 42;
  const logBox   = document.getElementById('gen-log');
  const btn      = document.getElementById('btn-generate');

  btn.disabled = true;
  btn.textContent = '⟳ Generating…';
  logBox.classList.remove('hidden');
  logBox.textContent = 'Starting data generation…\n';

  try {
    const res = await fetch(`${API}/generate-data`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ belts, max_hours: maxHours, step, seed }),
    });
    const data = await res.json();
    if (data.error) {
      logBox.textContent += `ERROR: ${data.error}\n${data.stderr || ''}`;
      toast('Data generation failed.', 'error');
    } else {
      logBox.textContent += data.log + '\n\n✓ Saved to: ' + data.output_path;
      toast('Dataset generated!', 'success');
      await loadDatasets();
    }
  } catch (e) {
    logBox.textContent += `Network error: ${e.message}`;
    toast('Could not reach API.', 'error');
  }

  btn.disabled = false;
  btn.innerHTML = '<span class="btn-icon">⚡</span> Generate Dataset';
}

// ── Upload CSV ─────────────────────────────────────────────────────────────
async function uploadDataset() {
  const file = document.getElementById('upload-file').files[0];
  if (!file) return;
  const status = document.getElementById('upload-status');
  status.classList.remove('hidden');
  status.textContent = `Uploading ${file.name}…`;

  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch(`${API}/upload-dataset`, { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      status.textContent = `Error: ${data.error}`;
      toast('Upload failed.', 'error');
    } else {
      status.textContent = `✓ Uploaded: ${data.filename} (${data.rows.toLocaleString()} rows)`;
      toast(`Uploaded ${data.filename}`, 'success');
      await loadDatasets();
    }
  } catch (e) {
    status.textContent = `Network error: ${e.message}`;
    toast('Upload failed.', 'error');
  }
}

// ── Train Model ────────────────────────────────────────────────────────────
async function trainModel() {
  if (!selectedDatasetPath) {
    toast('Please select a dataset first.', 'error');
    return;
  }

  showOverlay('Training XGBoost + Random Forest model…', 'This may take 1–2 minutes depending on dataset size.');
  setModelStatus('training', 'Training…');

  try {
    const res  = await fetch(`${API}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_path: selectedDatasetPath }),
    });
    const data = await res.json();
    hideOverlay();

    if (data.error) {
      toast(`Training failed: ${data.error}`, 'error');
      setModelStatus('error', 'Training failed');
      return;
    }

    modelLoaded = true;
    setModelStatus('ready', `Model ready — R² ${data.metrics?.ensemble?.r2_score ?? '—'}`);
    showTrainResult(data);
    toast('Model trained successfully!', 'success');
  } catch (e) {
    hideOverlay();
    toast(`Error: ${e.message}`, 'error');
    setModelStatus('error', 'Error');
  }
}

function showTrainResult(data) {
  const box = document.getElementById('train-result');
  const ens = data.metrics?.ensemble || {};
  box.classList.remove('hidden');
  box.innerHTML = `
    <div class="train-result-title">✓ Model Trained Successfully</div>
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="metric-label">R² Score (Ensemble)</div>
        <div class="metric-value green">${ens.r2_score ?? '—'}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">MAE</div>
        <div class="metric-value accent">${ens.mae_hours?.toLocaleString() ?? '—'} hrs</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">${ens.rmse_hours?.toLocaleString() ?? '—'} hrs</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Training Rows</div>
        <div class="metric-value">${data.training_rows?.toLocaleString() ?? '—'}</div>
      </div>
    </div>
    <div style="margin-top:20px; display:flex; justify-content:flex-end;">
      <button class="btn btn-primary" onclick="proceedToBeltSetup()">
        Configure New Belt →
      </button>
    </div>
  `;
}

function proceedToBeltSetup() {
  if (!modelLoaded) {
    toast('Please train a model first.', 'error');
    return;
  }
  goToStep(2);
}

// ── Belt Setup → Inspections ───────────────────────────────────────────────
function startInspections() {
  const name    = document.getElementById('belt-name').value.trim() || 'BLT-001';
  const hard    = parseFloat(document.getElementById('hardness').value);
  const thick   = parseFloat(document.getElementById('thickness').value);
  const tensile = parseFloat(document.getElementById('tensile').value);

  if (isNaN(hard) || isNaN(thick) || isNaN(tensile)) {
    toast('Please fill in all belt properties.', 'error');
    return;
  }

  beltProperties = {
    rubber_hardness_shore_a: hard,
    belt_thickness_mm: thick,
    tensile_strength_kn_m: tensile,
  };

  inspections = [];

  // Populate info bar
  document.getElementById('insp-belt-name').textContent = name;
  document.getElementById('insp-hardness').textContent  = `${hard} Shore A`;
  document.getElementById('insp-thickness').textContent = `${thick} mm`;
  document.getElementById('insp-tensile').textContent   = `${tensile} kN/m`;
  document.getElementById('insp-count').textContent     = '0';

  goToStep(3);
}

// ── Add inspection & predict ───────────────────────────────────────────────
async function addInspection() {
  if (!modelLoaded) {
    toast('No model loaded. Go back and train a model first.', 'error');
    return;
  }

  const hour  = parseFloat(document.getElementById('insp-hour').value);
  const load  = parseFloat(document.getElementById('insp-load').value);
  const speed = parseFloat(document.getElementById('insp-speed').value);
  const temp  = parseFloat(document.getElementById('insp-temp').value);
  const vib   = parseFloat(document.getElementById('insp-vib').value);
  const wear  = parseFloat(document.getElementById('insp-wear').value);
  const elong = parseFloat(document.getElementById('insp-elong').value);
  const crack = parseFloat(document.getElementById('insp-crack').value);

  if ([hour, load, speed, temp, vib, wear, elong, crack].some(isNaN)) {
    toast('Please fill in all inspection fields.', 'error');
    return;
  }

  // Validate degradation doesn't exceed failure thresholds
  if (wear >= 100 || elong >= 4 || crack >= 50) {
    toast('⚠ Belt has reached failure threshold!', 'error');
  }

  const inspRecord = { hour, load_pct: load, belt_speed_mps: speed,
    temperature_c: temp, vibration_mms: vib,
    wear_pct: wear, elongation_pct: elong, crack_length_mm: crack };

  // Prevent duplicate hours
  const existingIdx = inspections.findIndex(i => i.hour === hour);
  if (existingIdx >= 0) {
    inspections[existingIdx] = inspRecord;
  } else {
    inspections.push(inspRecord);
    inspections.sort((a, b) => a.hour - b.hour);
  }

  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-spinner">⟳</span> Predicting…';

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        belt_properties: beltProperties,
        inspections: inspections,
      }),
    });
    const data = await res.json();

    if (data.error) {
      toast(`Prediction error: ${data.error}`, 'error');
    } else {
      updateResults(data);
      // Advance hour suggestion
      document.getElementById('insp-hour').value = hour + 100;
    }
  } catch (e) {
    toast(`Network error: ${e.message}`, 'error');
  }

  btn.disabled = false;
  btn.innerHTML = '<span class="btn-icon">🔮</span> Predict RUL';
}

// ── Update result panel ────────────────────────────────────────────────────
function updateResults(data) {
  const latest = data.latest_prediction;
  const history = data.prediction_history;
  const n = data.total_inspections;

  // Show/hide panels
  document.getElementById('results-empty').classList.add('hidden');
  document.getElementById('rul-card').classList.remove('hidden');
  document.getElementById('history-card').classList.remove('hidden');
  document.getElementById('history-table-card').classList.remove('hidden');

  // RUL value
  animateNumber('rul-value', latest.rul_hours, ' hrs');
  document.getElementById('rul-sub').textContent =
    `At ${latest.hour.toLocaleString()} hrs operation · ${n} inspection${n > 1 ? 's' : ''} submitted`;

  // Confidence
  const conf = latest.confidence_pct;
  document.getElementById('confidence-pct').textContent = `${conf}%`;
  document.getElementById('confidence-bar').style.width = `${conf}%`;

  // Health
  document.getElementById('health-score').textContent  = `${latest.health_score}%`;
  document.getElementById('dominant-risk').textContent = latest.dominant_driver;
  document.getElementById('runtime-hrs').textContent   = latest.hour.toLocaleString();
  document.getElementById('insp-count-card').textContent = n;
  document.getElementById('insp-count').textContent      = n;

  // Degradation gauges
  const deg = latest.degradation_state;
  setGauge('gauge-wear-circle',  'gauge-wear-pct',  deg.wear_pct / 100);
  setGauge('gauge-elong-circle', 'gauge-elong-pct', deg.elongation_pct / 4);
  setGauge('gauge-crack-circle', 'gauge-crack-pct', deg.crack_length_mm / 50);

  // Threshold bars (inline inputs)
  document.getElementById('wear-fill').style.width  = `${Math.min(deg.wear_pct, 100)}%`;
  document.getElementById('elong-fill').style.width = `${Math.min(deg.elongation_pct / 4 * 100, 100)}%`;
  document.getElementById('crack-fill').style.width = `${Math.min(deg.crack_length_mm / 50 * 100, 100)}%`;

  // Chart
  renderChart(history);

  // Table
  renderTable(history);
}

// Gauge ring animation (SVG stroke-dashoffset trick)
function setGauge(circleId, pctId, ratio) {
  const circumference = 213.6; // 2π×34
  ratio = Math.min(Math.max(ratio, 0), 1);
  const offset = circumference * (1 - ratio);
  const circle = document.getElementById(circleId);
  circle.style.strokeDashoffset = offset;
  document.getElementById(pctId).textContent = `${Math.round(ratio * 100)}%`;
}

// Animate number counter
function animateNumber(id, target, suffix = '') {
  const el = document.getElementById(id);
  const start = parseFloat(el.textContent) || 0;
  const duration = 600;
  const startTime = performance.now();
  function update(now) {
    const t = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = Math.round(start + (target - start) * eased);
    el.textContent = val.toLocaleString() + suffix;
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── Chart ──────────────────────────────────────────────────────────────────
function renderChart(history) {
  const labels    = history.map(h => `${h.hour}h`);
  const rulData   = history.map(h => h.rul_hours);
  const confData  = history.map(h => h.confidence_pct);

  if (rulChart) rulChart.destroy();

  const ctx = document.getElementById('rul-chart').getContext('2d');
  rulChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'RUL (hrs)',
          data: rulData,
          borderColor: '#4f8ef7',
          backgroundColor: 'rgba(79,142,247,0.08)',
          borderWidth: 2.5,
          pointRadius: 5,
          pointBackgroundColor: '#4f8ef7',
          pointHoverRadius: 8,
          tension: 0.35,
          fill: true,
          yAxisID: 'y',
        },
        {
          label: 'Confidence (%)',
          data: confData,
          borderColor: '#22d3a0',
          backgroundColor: 'rgba(34,211,160,0.05)',
          borderWidth: 2,
          borderDash: [5, 4],
          pointRadius: 4,
          pointBackgroundColor: '#22d3a0',
          tension: 0.35,
          fill: false,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { size: 12 } },
        },
        tooltip: {
          backgroundColor: '#111827',
          borderColor: '#1e2d47',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
        },
      },
      scales: {
        x: {
          ticks: { color: '#64748b', font: { size: 11 } },
          grid:  { color: 'rgba(30,45,71,0.6)' },
        },
        y: {
          type: 'linear', position: 'left',
          title: { display: true, text: 'RUL (hrs)', color: '#4f8ef7', font: { size: 11 } },
          ticks: { color: '#64748b', font: { size: 11 } },
          grid:  { color: 'rgba(30,45,71,0.6)' },
        },
        y1: {
          type: 'linear', position: 'right',
          title: { display: true, text: 'Confidence (%)', color: '#22d3a0', font: { size: 11 } },
          ticks: { color: '#64748b', font: { size: 11 } },
          grid:  { drawOnChartArea: false },
          min: 0, max: 100,
        },
      },
    },
  });
}

// ── Table ──────────────────────────────────────────────────────────────────
function renderTable(history) {
  const tbody = document.getElementById('inspection-tbody');
  tbody.innerHTML = '';
  [...history].reverse().forEach((h, idx) => {
    const isLatest = idx === 0;
    const tr = document.createElement('tr');
    if (isLatest) tr.classList.add('latest');
    tr.innerHTML = `
      <td>${h.inspection_index}</td>
      <td>${h.hour.toLocaleString()}</td>
      <td class="rul-cell">${h.rul_hours.toLocaleString()}</td>
      <td class="conf-cell">${h.confidence_pct}%</td>
      <td class="health-cell">${h.health_score}%</td>
      <td>${h.degradation_state.wear_pct}%</td>
      <td>${h.degradation_state.elongation_pct}%</td>
      <td>${h.degradation_state.crack_length_mm} mm</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Slider sync ────────────────────────────────────────────────────────────
function syncSlider(inputId, sliderId) {
  const val = document.getElementById(sliderId).value;
  document.getElementById(inputId).value = val;
  if (inputId === 'insp-wear')  document.getElementById('wear-fill').style.width  = `${val}%`;
  if (inputId === 'insp-elong') document.getElementById('elong-fill').style.width = `${val / 4 * 100}%`;
  if (inputId === 'insp-crack') document.getElementById('crack-fill').style.width = `${val / 50 * 100}%`;
}

function setupInputListeners() {
  // Keep sliders in sync with typed input values
  const pairs = [
    ['insp-load',  'insp-load-slider'],
    ['insp-speed', 'insp-speed-slider'],
    ['insp-temp',  'insp-temp-slider'],
    ['insp-vib',   'insp-vib-slider'],
  ];
  pairs.forEach(([inputId, sliderId]) => {
    document.getElementById(inputId)?.addEventListener('input', () => {
      document.getElementById(sliderId).value = document.getElementById(inputId).value;
    });
  });

  // Live threshold bars from typed inputs
  document.getElementById('insp-wear')?.addEventListener('input', e => {
    document.getElementById('wear-fill').style.width = `${Math.min(e.target.value, 100)}%`;
  });
  document.getElementById('insp-elong')?.addEventListener('input', e => {
    document.getElementById('elong-fill').style.width = `${Math.min(e.target.value / 4 * 100, 100)}%`;
  });
  document.getElementById('insp-crack')?.addEventListener('input', e => {
    document.getElementById('crack-fill').style.width = `${Math.min(e.target.value / 50 * 100, 100)}%`;
  });
}

// ── Drag-drop upload ───────────────────────────────────────────────────────
function setupDragDrop() {
  const drop = document.getElementById('upload-drop');
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', async e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      document.getElementById('upload-file').files = e.dataTransfer.files;
      await uploadDataset();
    } else {
      toast('Please drop a CSV file.', 'error');
    }
  });
}

// ── Overlay ────────────────────────────────────────────────────────────────
function showOverlay(text, sub) {
  document.getElementById('overlay-text').textContent = text;
  document.getElementById('overlay-sub').textContent  = sub;
  document.getElementById('overlay').classList.remove('hidden');
}
function hideOverlay() {
  document.getElementById('overlay').classList.add('hidden');
}

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast ${type}`;
  el.classList.remove('hidden');
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => el.classList.add('hidden'), 4000);
}
