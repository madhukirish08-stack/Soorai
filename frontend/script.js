/**
 * BeltRUL Studio — Main Application Script
 * ==========================================
 * Flow:
 *   Hub → (predict) → Setup → Inspect & Forecast
 *   Hub → (train)   → Training View → Comparison Studio → Hub
 *
 * Key design decisions:
 *   - Belt types cached in localStorage → instant first render
 *   - Views shown/hidden with showView() — no page reloads
 *   - Model zoo loads lazily when belt type selected
 */

const API = (window.location.origin && (window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')))
  ? window.location.origin
  : 'http://localhost:8001';

// ── State ──────────────────────────────────────────────────────────────────────
let beltTypes           = {};
let selectedBeltTypeId  = null;
let selectedDatasetPath = null;
let selectedDataset     = null;
let currentProposal     = null;
let canonicalFields     = {};
let modelLoaded         = false;
let activeModelDetails  = null;
let candidateModelId    = null;
let beltProperties      = {};
let inspections         = [];
let rulChart            = null;

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Load schema in background
  try { canonicalFields = (await (await fetch(`${API}/schema`)).json()).fields || {}; } catch {}

  // Show cached belt types instantly, then refresh from API
  const cached = getCachedBeltTypes();
  if (cached) {
    beltTypes = cached;
    renderBeltGrid();
  }
  await loadBeltTypes(); // refreshes from API and re-renders

  loadDatasets();
  checkModelStatus();
  setupDragDrop();
  setupInputListeners();
});

// ── View Navigation ────────────────────────────────────────────────────────────
function showView(viewId) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const view = document.getElementById(viewId);
  if (view) {
    view.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

// ── Model Status ───────────────────────────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res = await fetch(`${API}/metadata`);
    const data = await res.json();
    if (data.metrics) {
      activeModelDetails = data;
      const algo = data.algorithm || 'xgboost';
      const score = data.metrics[algo]?.r2_score ?? nextVal(data.metrics)?.r2_score ?? '—';
      setModelStatus('ready', `Champion: ${algo.toUpperCase()} · R² ${score}`);
      modelLoaded = true;
      configureFormForModel(data.feature_cols || [], data.feature_stats || {});
      updateHubPredictCard();
    }
  } catch {
    setModelStatus('idle', 'No model loaded');
  }
}

function nextVal(obj) { return obj ? Object.values(obj)[0] : null; }

function setModelStatus(type, text) {
  const badge = document.getElementById('model-status');
  const label = document.getElementById('model-status-text');
  if (badge) badge.className = `status-badge status-${type}`;
  if (label) label.textContent = text;
}

function updateHubPredictCard() {
  const btn = document.getElementById('btn-go-predict');
  const badge = document.getElementById('champion-badge');
  const noModel = document.getElementById('no-model-notice');
  const desc = document.getElementById('predict-desc');

  if (modelLoaded && activeModelDetails) {
    const algo = activeModelDetails.algorithm || 'xgboost';
    const metrics = activeModelDetails.metrics?.[algo] || nextVal(activeModelDetails.metrics) || {};
    const r2 = metrics.r2_score ?? '—';
    if (btn) btn.disabled = false;
    if (badge) {
      badge.classList.remove('hidden');
      document.getElementById('champ-algo-name').textContent = algo.toUpperCase();
      document.getElementById('champ-r2-val').textContent = `R² ${r2}`;
    }
    if (noModel) noModel.classList.add('hidden');
    if (desc) desc.textContent = 'Active champion model is ready. Configure your belt and start entering inspection readings.';
  } else {
    if (btn) btn.disabled = true;
    if (badge) badge.classList.add('hidden');
    if (noModel) noModel.classList.remove('hidden');
  }
}

// ── Belt Types & Profiles ──────────────────────────────────────────────────────

function getCachedBeltTypes() {
  try {
    const s = localStorage.getItem('beltTypes_cache');
    if (s) return JSON.parse(s);
  } catch {}
  return null;
}

function setCachedBeltTypes(data) {
  try { localStorage.setItem('beltTypes_cache', JSON.stringify(data)); } catch {}
}

async function loadBeltTypes() {
  try {
    const res = await fetch(`${API}/belt-types`);
    const data = await res.json();
    beltTypes = data.types || {};
    setCachedBeltTypes(beltTypes);
    renderBeltGrid();
    populateStep2BeltDropdown();
    // Auto-select first if nothing selected
    if (!selectedBeltTypeId) {
      const first = Object.keys(beltTypes)[0];
      if (first) selectBeltType(first, false);
    }
  } catch (e) {
    console.error('Error loading belt types:', e);
  }
}

function renderBeltGrid() {
  const grid = document.getElementById('belt-grid');
  if (!grid) return;
  grid.innerHTML = '';

  const types = Object.values(beltTypes);
  if (types.length === 0) {
    grid.innerHTML = '<div class="empty-state">No belt types found.</div>';
    return;
  }

  types.forEach(bt => {
    const card = document.createElement('div');
    card.className = `belt-card ${bt.id === selectedBeltTypeId ? 'selected' : ''}`;
    card.id = `bt-card-${bt.id}`;
    const customTag = bt.is_custom ? '<div class="belt-card-custom-tag">Custom</div>' : '';
    card.innerHTML = `
      ${customTag}
      <span class="belt-card-icon">${bt.icon || '⚙️'}</span>
      <div class="belt-card-name">${escapeHtml(bt.name)}</div>
      <div class="belt-card-geom">${escapeHtml(bt.geometry || 'Standard')}</div>
      <div class="belt-card-desc">${escapeHtml(bt.description || '')}</div>
    `;
    card.addEventListener('click', () => selectBeltType(bt.id));
    grid.appendChild(card);
  });
}

function selectBeltType(typeId, reloadZoo = true) {
  selectedBeltTypeId = typeId;
  const bt = beltTypes[typeId];
  if (!bt) return;

  // Highlight card
  document.querySelectorAll('.belt-card').forEach(el => el.classList.remove('selected'));
  document.getElementById(`bt-card-${typeId}`)?.classList.add('selected');

  // Show detail section
  const detailSection = document.getElementById('belt-detail-section');
  if (detailSection) detailSection.classList.remove('hidden');

  // Update belt strip
  setEl('bs-icon', bt.icon || '⚙️');
  setEl('bs-name', bt.name);
  setEl('bs-geom', bt.geometry || 'Custom Profile');
  setEl('bs-hardness', `${bt.nominal_hardness || '—'} Shore A`);
  setEl('bs-thickness', `${bt.nominal_thickness || '—'} mm`);
  setEl('bs-tensile', `${bt.nominal_tensile || '—'} kN/m`);

  // Risk pills
  const risksEl = document.getElementById('bs-risks');
  if (risksEl) {
    risksEl.innerHTML = (bt.typical_failure_modes || [])
      .map(r => `<span class="risk-pill">${escapeHtml(r)}</span>`).join('');
  }

  // Sync Setup view chips
  setEl('setup-chip-icon', bt.icon || '⚙️');
  setEl('setup-chip-name', bt.name);
  setEl('train-chip-icon', bt.icon || '⚙️');
  setEl('train-chip-name', bt.name);

  // Sync Step2 select
  const sel = document.getElementById('step2-belt-type');
  if (sel) sel.value = typeId;
  applyBeltTypePresetToInputs(typeId);

  // Reload model zoo
  if (reloadZoo) loadModelZoo(typeId);

  // Update predict card based on current model
  updateHubPredictCard();
}

function populateStep2BeltDropdown() {
  const sel = document.getElementById('step2-belt-type');
  if (!sel) return;
  sel.innerHTML = '';
  Object.values(beltTypes).forEach(bt => {
    const opt = document.createElement('option');
    opt.value = bt.id;
    opt.textContent = `${bt.name} (${bt.geometry})`;
    sel.appendChild(opt);
  });
  if (selectedBeltTypeId) sel.value = selectedBeltTypeId;
}

function applyBeltTypePresetToInputs(typeId) {
  const bt = beltTypes[typeId];
  if (!bt) return;
  setVal('hardness', bt.nominal_hardness || 65);
  setVal('thickness', bt.nominal_thickness || 15);
  setVal('tensile', bt.nominal_tensile || 1000);
  const badge = document.getElementById('preset-badge');
  if (badge) {
    badge.textContent = `Preset: ${bt.name}`;
    badge.style.animation = 'none';
    badge.offsetHeight;
    badge.style.animation = '';
  }
}

// ── Navigate to Predict ────────────────────────────────────────────────────────
function goPredict() {
  if (!modelLoaded) { toast('No model loaded.', 'error'); return; }
  showView('view-setup');
}

// ── Custom Belt Modal ──────────────────────────────────────────────────────────
function openAddBeltModal() {
  document.getElementById('custom-belt-modal')?.classList.remove('hidden');
}
function closeAddBeltModal() {
  document.getElementById('custom-belt-modal')?.classList.add('hidden');
}
// Legacy aliases
function openCustomBeltModal() { openAddBeltModal(); }
function closeCustomBeltModal() { closeAddBeltModal(); }

async function submitCustomBeltType() {
  const name = document.getElementById('custom-name').value.trim();
  const geom = document.getElementById('custom-geometry').value.trim();
  const hard = parseFloat(document.getElementById('custom-hardness').value);
  const thick = parseFloat(document.getElementById('custom-thickness').value);
  const tensile = parseFloat(document.getElementById('custom-tensile').value);
  const failuresStr = document.getElementById('custom-failures').value.trim();
  const desc = document.getElementById('custom-desc').value.trim();

  if (!name) { toast('Enter a belt profile name.', 'error'); return; }
  const failures = failuresStr ? failuresStr.split(',').map(s => s.trim()).filter(Boolean) : ['Surface Wear'];

  try {
    const res = await fetch(`${API}/belt-types`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, geometry: geom || 'Custom', nominal_hardness: hard||65, nominal_thickness: thick||15, nominal_tensile: tensile||1000, typical_failure_modes: failures, description: desc || 'Custom belt profile.' })
    });
    const data = await res.json();
    if (data.error) { toast(`Error: ${data.error}`, 'error'); return; }
    toast(`Belt profile "${name}" added!`, 'success');
    closeAddBeltModal();
    await loadBeltTypes();
    if (data.belt_type?.id) selectBeltType(data.belt_type.id);
  } catch (e) {
    toast(`Failed: ${e.message}`, 'error');
  }
}

// ── Model Zoo ─────────────────────────────────────────────────────────────────
async function loadModelZoo(beltType = selectedBeltTypeId) {
  const list = document.getElementById('model-zoo-list');
  const countBadge = document.getElementById('zoo-count-badge');
  if (!list) return;
  list.innerHTML = '<div class="empty-state">Loading models…</div>';

  try {
    const query = beltType ? `?belt_type=${encodeURIComponent(beltType)}` : '';
    const res = await fetch(`${API}/models${query}`);
    const data = await res.json();
    const models = data.models || [];
    if (countBadge) countBadge.textContent = `${models.length} model${models.length === 1 ? '' : 's'}`;

    if (models.length === 0) {
      list.innerHTML = '<div class="empty-state">No trained models yet for this belt type. Go to Training to build one.</div>';
      return;
    }

    list.innerHTML = '';
    models.forEach(m => {
      const card = document.createElement('div');
      card.className = `model-card-item ${m.is_champion ? 'is-champion' : ''}`;
      const algo = m.algorithm || 'xgboost';
      const mStats = m.metrics?.[algo] || nextVal(m.metrics) || {};
      const r2 = mStats.r2_score ?? '—';
      const mae = mStats.mae_hours ? `${mStats.mae_hours.toLocaleString()} h` : '—';
      const dateStr = m.created_at ? new Date(m.created_at).toLocaleDateString() : '';
      card.innerHTML = `
        <div class="model-card-header">
          <span class="algo-tag algo-${algo}">${algo.replace('_', ' ')}</span>
          ${m.is_champion ? '<span class="badge-champ">★ Champion</span>' : ''}
          <span style="margin-left:auto;font-size:11px;color:var(--text-muted)">${dateStr}</span>
        </div>
        <div class="model-id-label" title="${m.model_id}">${m.model_id}</div>
        <div class="model-metrics-row">
          <div class="metric-chip"><span class="m-title">R²</span><span class="m-val green">${r2}</span></div>
          <div class="metric-chip"><span class="m-title">MAE</span><span class="m-val">${mae}</span></div>
          <div class="metric-chip"><span class="m-title">Dataset</span><span class="m-val" style="font-size:11px">${m.dataset ? escapeHtml(m.dataset.substring(0,16)) : 'default'}</span></div>
        </div>
        <div class="model-card-actions">
          <button class="btn btn-sm btn-outline" onclick="openModelDetailsModal('${m.model_id}')">Details & Trade-offs</button>
          ${!m.is_champion ? `<button class="btn btn-sm btn-ghost" onclick="promoteModel('${m.model_id}')">★ Set Champion</button>` : ''}
        </div>
      `;
      list.appendChild(card);
    });
  } catch (e) {
    list.innerHTML = '<div class="empty-state" style="color:var(--red)">Failed to load models.</div>';
  }
}

async function promoteModel(modelId) {
  showOverlay('Promoting to Champion…', 'Updating active weights.');
  try {
    const res = await fetch(`${API}/models/${modelId}/promote`, { method: 'POST' });
    const data = await res.json();
    hideOverlay();
    if (data.error) { toast(`Error: ${data.error}`, 'error'); return; }
    toast(`Model ${modelId} promoted to champion!`, 'success');
    await checkModelStatus();
    await loadModelZoo();
  } catch (e) {
    hideOverlay();
    toast(`Failed: ${e.message}`, 'error');
  }
}

// ── Datasets ──────────────────────────────────────────────────────────────────
async function loadDatasets() {
  const list = document.getElementById('dataset-list');
  if (!list) return;
  list.innerHTML = '<div class="empty-state">Loading datasets…</div>';
  try {
    const res  = await fetch(`${API}/datasets`);
    const data = await res.json();
    if (!data.datasets || data.datasets.length === 0) {
      list.innerHTML = '<div class="empty-state">No CSV datasets found. Upload or generate one below.</div>';
      return;
    }
    list.innerHTML = '';
    data.datasets.forEach(ds => {
      const item = document.createElement('div');
      item.className = 'dataset-item';
      item.dataset.path = ds.path;
      const isDefault = ds.filename === 'belt_rul_training_data.csv';
      if (selectedDataset ? ds.filename === selectedDataset.filename : isDefault) {
        item.classList.add('selected');
        selectedDatasetPath = ds.path;
        selectedDataset = ds;
        updateSelectedDisplay(ds.filename);
        updateActionButtons();
      }
      const statusLabel = { approved: '✓ mapped', not_needed: 'canonical', required: '⚠ needs mapping' }[ds.mapping_status] || '';
      item.innerHTML = `
        <div class="dataset-icon">📊</div>
        <div>
          <div class="dataset-name">${escapeHtml(ds.filename)}</div>
          <div class="dataset-meta">${ds.size_mb} MB · ${ds.columns.length} cols
            <span class="map-status map-${ds.mapping_status}">${statusLabel}</span></div>
        </div>
        <div class="dataset-check">✓</div>
      `;
      item.addEventListener('click', () => selectDataset(item, ds));
      list.appendChild(item);
    });
  } catch (e) {
    list.innerHTML = '<div class="empty-state" style="color:var(--red)">Cannot reach API. Is the server running?</div>';
  }
}

function selectDataset(item, ds) {
  document.querySelectorAll('.dataset-item').forEach(el => el.classList.remove('selected'));
  item.classList.add('selected');
  selectedDatasetPath = ds.path;
  selectedDataset = ds;
  updateSelectedDisplay(ds.filename);
  closeMapping();
  updateActionButtons();
}

function updateSelectedDisplay(name) {
  const el = document.getElementById('selected-dataset-display');
  if (el) el.textContent = name;
}

function updateActionButtons() {
  const status = selectedDataset?.mapping_status;
  const isReady = selectedDataset && status !== 'required';
  const mapBtn = document.getElementById('btn-map');
  const trainBtn = document.getElementById('btn-train');
  if (mapBtn) mapBtn.disabled = !selectedDataset;
  if (trainBtn) trainBtn.disabled = !isReady;
}

// ── Training ──────────────────────────────────────────────────────────────────
async function trainCandidateModel() {
  if (!selectedDatasetPath) { toast('Select a dataset first.', 'error'); return; }
  const algo = document.getElementById('train-algorithm').value;
  showOverlay(`Training ${algo.toUpperCase()}…`, 'Computing features and fitting ensemble.');
  setModelStatus('training', `Training ${algo.toUpperCase()}…`);
  try {
    const res = await fetch(`${API}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        csv_path: selectedDataset.filename,
        algorithm: algo,
        belt_type: selectedBeltTypeId,
        notes: `Trained from Web Studio on ${selectedDataset.filename}`
      })
    });
    const data = await res.json();
    hideOverlay();
    if (data.mapping_required) {
      setModelStatus(modelLoaded ? 'ready' : 'idle', modelLoaded ? 'Model ready' : 'No model loaded');
      toast(data.error, 'error');
      openMapping();
      return;
    }
    if (data.error) {
      toast(`Training failed: ${data.error}`, 'error');
      setModelStatus('error', 'Training failed');
      return;
    }
    modelLoaded = true;
    candidateModelId = data.model_id;
    await checkModelStatus();
    await loadModelZoo();
    toast(`Candidate trained (${algo.toUpperCase()})! Opening Comparison Studio.`, 'success');
    if (data.comparison) openComparisonModal(data.comparison);
  } catch (e) {
    hideOverlay();
    toast(`Training error: ${e.message}`, 'error');
    setModelStatus('error', 'Error');
  }
}

// Legacy alias
function trainModel() { trainCandidateModel(); }

// ── Benchmarking ─────────────────────────────────────────────────────────────
async function benchmarkSelectedDataset() {
  if (!selectedDatasetPath) { toast('Select a dataset first.', 'error'); return; }
  showOverlay('Running Benchmark…', 'Evaluating champion model on selected dataset.');
  try {
    const res = await fetch(`${API}/models/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_path: selectedDataset.filename })
    });
    const data = await res.json();
    hideOverlay();
    if (data.mapping_required) { toast(data.error, 'error'); openMapping(); return; }
    if (data.error) { toast(`Benchmark failed: ${data.error}`, 'error'); return; }
    renderBenchmarkModal(data.benchmark);
  } catch (e) {
    hideOverlay();
    toast(`Error: ${e.message}`, 'error');
  }
}

function renderBenchmarkModal(bench) {
  setEl('bench-modal-sub', `Evaluated champion on "${bench.dataset}" (${bench.rows_evaluated?.toLocaleString()} rows)`);
  setEl('bench-rows', bench.rows_evaluated?.toLocaleString() ?? '—');
  setEl('bench-r2', bench.metrics?.r2_score ?? '—');
  setEl('bench-mae', bench.metrics?.mae_hours?.toLocaleString() ?? '—');
  setEl('bench-rmse', bench.metrics?.rmse_hours?.toLocaleString() ?? '—');
  const tbody = document.getElementById('bench-sample-tbody');
  if (tbody) {
    tbody.innerHTML = '';
    (bench.samples || []).forEach((s, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>#${idx+1}</td>
        <td>${s.actual_rul_hours?.toLocaleString()} hrs</td>
        <td style="color:var(--accent);font-weight:600">${s.predicted_rul_hours?.toLocaleString()} hrs</td>
        <td style="color:${s.error_hours<200?'var(--green)':'var(--amber)'}">${s.error_hours?.toLocaleString()} hrs</td>
      `;
      tbody.appendChild(tr);
    });
  }
  document.getElementById('benchmark-modal')?.classList.remove('hidden');
}

function closeBenchmarkModal() { document.getElementById('benchmark-modal')?.classList.add('hidden'); }

// ── Comparison Studio ─────────────────────────────────────────────────────────
function openComparisonModal(comp) {
  const cand = comp.candidate || {};
  const base = comp.baseline;
  const deltas = comp.deltas;
  candidateModelId = cand.model_id;

  setEl('comp-models-sub', `Comparing ${cand.model_id} against active champion`);
  const banner = document.getElementById('verdict-banner');
  const pill   = document.getElementById('verdict-pill');
  if (banner) banner.className = `verdict-banner verdict-${comp.verdict_color || 'green'}`;
  if (pill) pill.textContent = comp.verdict || 'COMPARISON';
  setEl('verdict-text', comp.verdict_label || 'Evaluation Complete');

  setEl('cand-algo-name', (cand.algorithm || 'xgboost').toUpperCase());
  setEl('cand-model-id', cand.model_id || 'Candidate');
  setEl('cand-r2', cand.r2_score ?? '—');
  setEl('cand-mae', `${cand.mae_hours?.toLocaleString() ?? '—'} hrs`);
  setEl('cand-rmse', `${cand.rmse_hours?.toLocaleString() ?? '—'} hrs`);
  setEl('cand-rows', cand.training_rows?.toLocaleString() ?? '—');

  if (deltas) {
    const r2El = document.getElementById('delta-r2');
    const maeEl = document.getElementById('delta-mae');
    const rmseEl = document.getElementById('delta-rmse');
    if (r2El) { r2El.textContent = (deltas.r2_diff >= 0 ? '+' : '') + deltas.r2_diff; r2El.className = `delta-num ${deltas.r2_diff >= 0 ? 'delta-pos' : 'delta-neg'}`; }
    if (maeEl) { maeEl.textContent = (deltas.mae_diff <= 0 ? '' : '+') + deltas.mae_diff + ' h'; maeEl.className = `delta-num ${deltas.mae_diff <= 0 ? 'delta-pos' : 'delta-neg'}`; }
    if (rmseEl) { rmseEl.textContent = (deltas.rmse_diff <= 0 ? '' : '+') + deltas.rmse_diff + ' h'; rmseEl.className = `delta-num ${deltas.rmse_diff <= 0 ? 'delta-pos' : 'delta-neg'}`; }
  }

  if (base) {
    setEl('champ-algo-name', (base.algorithm || 'Champion').toUpperCase());
    setEl('champ-model-id', base.model_id || 'Champion');
    setEl('champ-r2', base.r2_score ?? '—');
    setEl('champ-mae', `${base.mae_hours?.toLocaleString() ?? '—'} hrs`);
    setEl('champ-rmse', `${base.rmse_hours?.toLocaleString() ?? '—'} hrs`);
    setEl('champ-rows', base.training_rows?.toLocaleString() ?? '—');
    const champBox = document.getElementById('champ-tradeoff-box');
    if (champBox) champBox.style.display = 'flex';
    const champTrade = base.tradeoffs || {};
    setEl('champ-tradeoff-title', `${(base.algorithm || 'Champion').toUpperCase()} Architecture`);
    setEl('champ-tradeoff-cat', champTrade.category || 'Ensemble');
    renderList('champ-pros-list', champTrade.advantages || []);
    renderList('champ-cons-list', champTrade.disadvantages || []);
    setEl('champ-best-for', champTrade.best_for ? `Best for: ${champTrade.best_for}` : '');
  } else {
    setEl('champ-algo-name', 'None');
    setEl('champ-model-id', 'No previous baseline');
    ['champ-r2','champ-mae','champ-rmse','champ-rows'].forEach(id => setEl(id, '—'));
    const champBox = document.getElementById('champ-tradeoff-box');
    if (champBox) champBox.style.display = 'none';
  }

  const candTrade = cand.tradeoffs || {};
  setEl('cand-tradeoff-title', `${(cand.algorithm || 'Model').toUpperCase()} Architecture`);
  setEl('cand-tradeoff-cat', candTrade.category || 'Ensemble');
  renderList('cand-pros-list', candTrade.advantages || []);
  renderList('cand-cons-list', candTrade.disadvantages || []);
  setEl('cand-best-for', candTrade.best_for ? `Best for: ${candTrade.best_for}` : '');

  setEl('rec-text', comp.recommendation || 'Evaluation complete.');
  document.getElementById('comparison-modal')?.classList.remove('hidden');
}

function renderList(elemId, items) {
  const el = document.getElementById(elemId);
  if (!el) return;
  el.innerHTML = items.map(t => `<li>${escapeHtml(t)}</li>`).join('');
}

function closeComparisonModal() { document.getElementById('comparison-modal')?.classList.add('hidden'); }

async function promoteCurrentCandidate() {
  if (!candidateModelId) return;
  await promoteModel(candidateModelId);
  closeComparisonModal();
  showView('view-hub');
}

async function openModelDetailsModal(modelId) {
  showOverlay('Loading model details…', '');
  try {
    const res = await fetch(`${API}/models/${modelId}`);
    const meta = await res.json();
    hideOverlay();
    if (meta.error) { toast(meta.error, 'error'); return; }
    const compRes = await fetch(`${API}/models/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_id: modelId })
    });
    const compData = await compRes.json();
    openComparisonModal(compData.comparison);
  } catch (e) {
    hideOverlay();
    toast(`Error: ${e.message}`, 'error');
  }
}

// ── Column Mapping ────────────────────────────────────────────────────────────
async function openMapping() {
  if (!selectedDataset) return;
  showOverlay('Analysing dataset columns…', 'Matching against canonical schema.');
  try {
    const res = await fetch(`${API}/schema/propose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_path: selectedDataset.filename }),
    });
    const data = await res.json();
    hideOverlay();
    if (data.error) { toast(`Mapping failed: ${data.error}`, 'error'); return; }
    currentProposal = data;
    renderMapping(data);
  } catch (e) {
    hideOverlay();
    toast(`Network error: ${e.message}`, 'error');
  }
}

function renderMapping(data) {
  const sourceText = { llm:'Proposed by Gemini', heuristic:'Name matching', exact:'Exact match', saved_profile:'Saved profile', approved:'Previously approved' }[data.source] || data.source;
  setEl('mapping-source', sourceText);
  const warnEl = document.getElementById('mapping-warnings');
  if (warnEl) warnEl.innerHTML = (data.warnings||[]).map(w=>`<div class="notice notice-warn">${escapeHtml(w)}</div>`).join('');
  const resultEl = document.getElementById('mapping-result');
  if (resultEl) resultEl.innerHTML = '';

  const profile = Object.fromEntries((data.profile||[]).map(p=>[p.name,p]));
  const fields = data.canonical_fields || canonicalFields;
  const options = [
    ['__ignore__','— Ignore column —'],
    ['__extra__', '+ Extra feature (keep as-is)'],
    ...Object.entries(fields).map(([k,f])=>[k, `${f.label}${f.unit?` (${f.unit})`:''} · ${k}`]),
  ];

  const tbody = document.getElementById('mapping-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  data.mapping.forEach(m => {
    const p = profile[m.source] || {};
    const conf = Math.round((m.confidence ?? 0)*100);
    const confClass = conf>=80?'conf-high':conf>=50?'conf-mid':'conf-low';
    const range = p.min!==undefined ? `<div class="hint">${fmt(p.min)} … ${fmt(p.max)}</div>` : '';
    const tr = document.createElement('tr');
    tr.dataset.source = m.source;
    tr.innerHTML = `
      <td class="map-src">${escapeHtml(m.source)}</td>
      <td class="map-samples">${escapeHtml((p.samples||[]).slice(0,3).join(', '))}${range}</td>
      <td><select class="map-target">${options.map(([v,l])=>`<option value="${escapeHtml(v)}"${v===m.target?' selected':''}>${escapeHtml(l)}</option>`).join('')}</select></td>
      <td><span class="conf-pill ${confClass}">${conf}%</span></td>
      <td><input type="number" class="map-scale" step="any" value="${m.scale??1}"/></td>
      <td><input type="number" class="map-offset" step="any" value="${m.offset??0}"/></td>
      <td class="map-reason">${escapeHtml(m.reason||'')}</td>
    `;
    tbody.appendChild(tr);
  });
  resetMappingConfirmation();
  tbody.oninput = tbody.onchange = resetMappingConfirmation;
  const mc = document.getElementById('mapping-card');
  if (mc) { mc.classList.remove('hidden'); mc.scrollIntoView({ behavior:'smooth' }); }
}

function closeMapping() {
  document.getElementById('mapping-card')?.classList.add('hidden');
  currentProposal = null;
}

let mappingWarningsSeen = false;
function resetMappingConfirmation() {
  mappingWarningsSeen = false;
  const btn = document.getElementById('btn-approve-mapping');
  if (btn) btn.textContent = '✓ Approve Mapping';
}

function collectMappingRows() {
  const rows = [];
  document.querySelectorAll('#mapping-tbody tr').forEach(tr => {
    rows.push({
      source: tr.dataset.source,
      target: tr.querySelector('.map-target').value,
      scale:  parseFloat(tr.querySelector('.map-scale').value) || 1,
      offset: parseFloat(tr.querySelector('.map-offset').value) || 0,
      confidence: 1.0,
      reason: tr.querySelector('.map-reason').textContent.trim(),
    });
  });
  return rows;
}

async function approveMapping() {
  const btn = document.getElementById('btn-approve-mapping');
  if (btn) btn.disabled = true;
  const mapping = collectMappingRows();
  const box = document.getElementById('mapping-result');
  if (box) box.innerHTML = '';

  const post = async dry => {
    const res = await fetch(`${API}/schema/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_path: selectedDataset.filename, mapping, dry_run: dry, proposal_source: currentProposal?.source||'manual' }),
    });
    return res.json();
  };

  const showNotices = data => {
    if (!box) return;
    const warns = (data.warnings||[]).map(w=>`<div class="notice notice-warn">⚠ ${escapeHtml(w)}</div>`).join('');
    const errs  = (data.errors||[]).map(e=>`<div class="notice notice-error">✕ ${escapeHtml(e)}</div>`).join('');
    box.innerHTML = errs + warns;
  };

  try {
    let data;
    if (!mappingWarningsSeen) {
      data = await post(true);
      showNotices(data);
      if (data.errors?.length > 0) { toast('Fix errors before approving.', 'error'); if(btn)btn.disabled=false; return; }
      if (data.warnings?.length > 0) {
        mappingWarningsSeen = true;
        if (btn) btn.textContent = '⚠ Approve anyway';
        toast('Review warnings, then approve again.', 'info');
        if (btn) btn.disabled = false;
        return;
      }
    }
    data = await post(false);
    showNotices(data);
    if (data.saved) {
      if (box) box.innerHTML = `<div class="notice notice-ok">✓ Mapping approved.</div>` + box.innerHTML;
      if (selectedDataset) selectedDataset.mapping_status = 'approved';
      updateActionButtons();
      toast('Mapping approved.', 'success');
      loadDatasets();
    } else {
      toast('Mapping has errors — fix and approve again.', 'error');
    }
    resetMappingConfirmation();
  } catch (e) {
    toast(`Network error: ${e.message}`, 'error');
  }
  if (btn) btn.disabled = false;
}

// ── Upload & Generate ─────────────────────────────────────────────────────────
async function generateData() {
  const belts    = parseInt(getVal('gen-belts')) || 300;
  const maxHours = parseInt(getVal('gen-max-hours')) || 20000;
  const step     = parseInt(getVal('gen-step') || 100);
  const seed     = parseInt(getVal('gen-seed')) || 42;
  const logBox   = document.getElementById('gen-log');
  const btn      = document.getElementById('btn-generate');

  if (btn) { btn.disabled = true; btn.textContent = '⟳ Generating…'; }
  if (logBox) { logBox.classList.remove('hidden'); logBox.textContent = 'Starting…\n'; }

  try {
    const res = await fetch(`${API}/generate-data`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ belts, max_hours: maxHours, step, seed }),
    });
    const data = await res.json();
    if (data.error) {
      if (logBox) logBox.textContent += `ERROR: ${data.error}\n${data.stderr||''}`;
      toast('Generation failed.', 'error');
    } else {
      if (logBox) logBox.textContent += data.log + '\n\n✓ Saved to: ' + data.output_path;
      toast('Dataset generated!', 'success');
      await loadDatasets();
    }
  } catch (e) {
    if (logBox) logBox.textContent += `Error: ${e.message}`;
    toast('Could not reach API.', 'error');
  }
  if (btn) { btn.disabled = false; btn.innerHTML = '⚡ Generate Synthetic Dataset'; }
}

async function uploadDataset() {
  const file = document.getElementById('upload-file')?.files[0];
  if (!file) return;
  const status = document.getElementById('upload-status');
  if (status) { status.classList.remove('hidden'); status.textContent = `Uploading ${file.name}…`; }

  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch(`${API}/upload-dataset`, { method:'POST', body:form });
    const data = await res.json();
    if (data.error) {
      if (status) status.textContent = `Error: ${data.error}`;
      toast('Upload failed.', 'error');
    } else {
      if (status) status.textContent = `✓ Uploaded: ${data.filename}`;
      toast(`Uploaded ${data.filename}`, 'success');
      await loadDatasets();
    }
  } catch (e) {
    if (status) status.textContent = `Error: ${e.message}`;
    toast('Upload failed.', 'error');
  }
}

// ── Setup → Inspect ───────────────────────────────────────────────────────────
function startInspections() {
  const name    = getVal('belt-name') || 'BLT-001';
  const hard    = parseFloat(getVal('hardness'));
  const thick   = parseFloat(getVal('thickness'));
  const tensile = parseFloat(getVal('tensile'));

  if (isNaN(hard) || isNaN(thick) || isNaN(tensile)) {
    toast('Fill in all belt properties.', 'error');
    return;
  }

  beltProperties = {
    rubber_hardness_shore_a: hard,
    belt_thickness_mm: thick,
    tensile_strength_kn_m: tensile,
  };
  inspections = [];

  const bt = beltTypes[selectedBeltTypeId];
  setEl('insp-belt-name', name);
  setEl('insp-type-badge', bt?.geometry || 'Standard');
  setEl('insp-hardness', `${hard} Shore A`);
  setEl('insp-thickness', `${thick} mm`);
  setEl('insp-tensile', `${tensile} kN/m`);
  setEl('insp-count', '0');

  showView('view-inspect');
}

// ── Inspection → Forecast ─────────────────────────────────────────────────────
const INSPECTION_INPUTS = {
  load_pct:       'insp-load',
  belt_speed_mps: 'insp-speed',
  temperature_c:  'insp-temp',
  vibration_mms:  'insp-vib',
  wear_pct:       'insp-wear',
  elongation_pct: 'insp-elong',
  crack_length_mm:'insp-crack',
};

function configureFormForModel(modelFeatures, featureStats) {
  const trained = new Set(modelFeatures);
  Object.entries(INSPECTION_INPUTS).forEach(([feat, inputId]) => {
    const el  = document.getElementById(inputId);
    const grp = el?.closest('.form-group');
    if (!el || !grp) return;
    const isUsed = trained.has(feat);
    el.disabled = !isUsed;
    grp.classList.toggle('unused', !isUsed);
    grp.title = isUsed ? '' : 'Model not trained on this feature.';
  });

  const fixed = new Set([...Object.keys(INSPECTION_INPUTS), 'hour','belt_id','timestamp']);
  const extras = modelFeatures.filter(f => !fixed.has(f) && !f.startsWith('__'));
  const box = document.getElementById('extra-inputs');
  if (box) {
    box.innerHTML = '';
    extras.forEach(f => {
      const stat = featureStats[f] || {};
      const med  = stat.median !== undefined ? stat.median : 0.0;
      const label = canonicalFields[f]?.label || f;
      const unit  = canonicalFields[f]?.unit ? ` (${canonicalFields[f].unit})` : '';
      const div = document.createElement('div');
      div.className = 'form-group';
      div.innerHTML = `<label>${escapeHtml(label)}${unit}</label>
        <input type="number" step="any" data-feature="${escapeHtml(f)}" value="${med}"/>
        <span class="hint">Training median: ${fmt(med)}</span>`;
      box.appendChild(div);
    });
  }
  document.getElementById('extra-inputs-card')?.classList.toggle('hidden', extras.length === 0);
}

async function addInspection() {
  if (!modelLoaded) {
    toast('No model loaded. Go back to Hub and train a model.', 'error');
    return;
  }

  const hour = parseFloat(getVal('insp-hour'));
  const inspRecord = { hour };
  Object.entries(INSPECTION_INPUTS).forEach(([feat, id]) => {
    const el = document.getElementById(id);
    if (!el.disabled) inspRecord[feat] = parseFloat(el.value);
  });
  document.querySelectorAll('#extra-inputs input').forEach(el => {
    inspRecord[el.dataset.feature] = parseFloat(el.value);
  });

  if (Object.values(inspRecord).some(isNaN)) { toast('Fill in all inspection fields.', 'error'); return; }

  const failed = Object.entries(inspRecord).some(([f,v]) =>
    canonicalFields[f]?.failure_threshold && v >= canonicalFields[f].failure_threshold);
  if (failed) toast('⚠ Belt has reached failure threshold!', 'error');

  const existingIdx = inspections.findIndex(i => i.hour === hour);
  if (existingIdx >= 0) inspections[existingIdx] = inspRecord;
  else { inspections.push(inspRecord); inspections.sort((a,b)=>a.hour-b.hour); }

  const btn = document.getElementById('btn-predict');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="btn-spinner">⟳</span> Forecasting…'; }

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ belt_properties: beltProperties, inspections }),
    });
    const data = await res.json();
    if (data.error) { toast(`Prediction error: ${data.error}`, 'error'); }
    else {
      updateResults(data);
      setVal('insp-hour', hour + 100);
    }
  } catch (e) {
    toast(`Network error: ${e.message}`, 'error');
  }
  if (btn) { btn.disabled = false; btn.innerHTML = '<span class="btn-icon">🔮</span> Forecast RUL'; }
}

function updateResults(data) {
  const latest  = data.latest_prediction;
  const history = data.prediction_history;
  const n = data.total_inspections;

  document.getElementById('results-empty')?.classList.add('hidden');
  document.getElementById('rul-card')?.classList.remove('hidden');
  document.getElementById('history-card')?.classList.remove('hidden');
  document.getElementById('history-table-card')?.classList.remove('hidden');

  animateNumber('rul-value', latest.rul_hours, '');
  setEl('rul-sub', `At ${latest.hour.toLocaleString()} hrs · ${n} inspection${n>1?'s':''}`);

  const conf = latest.confidence_pct;
  setEl('confidence-pct', `${conf}%`);
  const bar = document.getElementById('confidence-bar');
  if (bar) bar.style.width = `${conf}%`;

  setEl('health-score', latest.health_score===null ? '—' : `${latest.health_score}%`);
  setEl('dominant-risk', latest.dominant_driver);
  setEl('runtime-hrs', latest.hour.toLocaleString());
  setEl('active-algo-name', (data.algorithm||'Model').toUpperCase());
  setEl('insp-count', n);

  const deg = latest.degradation_state;
  setGauge('gauge-wear',  'gauge-wear-pct',  deg.wear_pct===null ? null : deg.wear_pct/100);
  setGauge('gauge-elong', 'gauge-elong-pct', deg.elongation_pct===null ? null : deg.elongation_pct/4);
  setGauge('gauge-crack', 'gauge-crack-pct', deg.crack_length_mm===null ? null : deg.crack_length_mm/50);

  const wf = document.getElementById('wear-fill');  if(wf)  wf.style.width  = `${Math.min(deg.wear_pct??0,100)}%`;
  const ef = document.getElementById('elong-fill'); if(ef)  ef.style.width  = `${Math.min((deg.elongation_pct??0)/4*100,100)}%`;
  const cf = document.getElementById('crack-fill'); if(cf)  cf.style.width  = `${Math.min((deg.crack_length_mm??0)/50*100,100)}%`;

  renderChart(history);
  renderTable(history);
}

function setGauge(circleId, pctId, ratio) {
  const circumference = 213.6;
  const circle = document.getElementById(circleId);
  if (!circle) return;
  if (ratio === null) { circle.style.strokeDashoffset = circumference; setEl(pctId, 'n/a'); return; }
  ratio = Math.min(Math.max(ratio, 0), 1);
  circle.style.strokeDashoffset = circumference * (1 - ratio);
  setEl(pctId, `${Math.round(ratio*100)}%`);
}

function animateNumber(id, target, suffix='') {
  const el = document.getElementById(id);
  if (!el) return;
  const start = parseFloat(el.textContent) || 0;
  const duration = 600;
  const startTime = performance.now();
  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed/duration, 1);
    const ease = 1 - Math.pow(1-progress, 3);
    el.textContent = `${Math.round(start + (target-start)*ease).toLocaleString()}${suffix}`;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function renderChart(history) {
  const ctx = document.getElementById('rul-chart')?.getContext('2d');
  if (!ctx) return;
  const labels  = history.map(h => `${h.hour}h`);
  const rulData = history.map(h => h.rul_hours);
  const confData= history.map(h => h.confidence_pct);

  if (rulChart) {
    rulChart.data.labels = labels;
    rulChart.data.datasets[0].data = rulData;
    rulChart.data.datasets[1].data = confData;
    rulChart.update();
    return;
  }
  rulChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'RUL (hrs)', data:rulData, borderColor:'#4f8ef7', backgroundColor:'rgba(79,142,247,0.08)', fill:true, tension:0.3, pointBackgroundColor:'#4f8ef7', pointRadius:4, yAxisID:'y' },
        { label:'Confidence (%)', data:confData, borderColor:'#22d3a0', borderDash:[4,4], backgroundColor:'transparent', tension:0.3, pointBackgroundColor:'#22d3a0', pointRadius:3, yAxisID:'y1' },
      ],
    },
    options: {
      responsive:true,
      interaction:{ mode:'index', intersect:false },
      plugins:{ legend:{ labels:{ color:'#94a3b8', font:{ family:'Inter', size:12 } } } },
      scales:{
        x:{ grid:{ color:'#1e2d47' }, ticks:{ color:'#64748b' } },
        y:{ grid:{ color:'#1e2d47' }, ticks:{ color:'#64748b' }, title:{ display:true, text:'RUL (hours)', color:'#4f8ef7' } },
        y1:{ position:'right', min:0, max:100, grid:{ drawOnChartArea:false }, ticks:{ color:'#22d3a0' }, title:{ display:true, text:'Confidence (%)', color:'#22d3a0' } },
      },
    },
  });
}

function renderTable(history) {
  const tbody = document.getElementById('inspection-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  history.slice().reverse().forEach(h => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${h.inspection_index}</td>
      <td>${h.hour.toLocaleString()}</td>
      <td class="bold accent">${h.rul_hours.toLocaleString()}</td>
      <td>${h.confidence_pct}%</td>
      <td>${h.health_score===null?'—':`${h.health_score}%`}</td>
      <td>${withUnit(h.degradation_state?.wear_pct,'%')}</td>
      <td>${withUnit(h.degradation_state?.elongation_pct,'%')}</td>
      <td>${withUnit(h.degradation_state?.crack_length_mm,'mm')}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(v) {
  return String(v).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}
function fmt(n) { return Number(n).toLocaleString(undefined, { maximumSignificantDigits: 4 }); }
function withUnit(v, unit) { return v===null||v===undefined ? '—' : `${v}${unit}`; }
function setEl(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }
function getVal(id) { return document.getElementById(id)?.value ?? ''; }
function setVal(id, v) { const el = document.getElementById(id); if (el) el.value = v; }

function syncSlider(inputId, sliderId) {
  const val = getVal(sliderId);
  setVal(inputId, val);
  if (inputId === 'insp-wear')  { const f=document.getElementById('wear-fill');  if(f)f.style.width=`${val}%`; }
  if (inputId === 'insp-elong') { const f=document.getElementById('elong-fill'); if(f)f.style.width=`${val/4*100}%`; }
  if (inputId === 'insp-crack') { const f=document.getElementById('crack-fill'); if(f)f.style.width=`${val/50*100}%`; }
}

function setupInputListeners() {
  [['insp-load','insp-load-slider'],['insp-speed','insp-speed-slider'],['insp-temp','insp-temp-slider'],['insp-vib','insp-vib-slider']].forEach(([iId,sId]) => {
    document.getElementById(iId)?.addEventListener('input', () => setVal(sId, getVal(iId)));
  });
  document.getElementById('insp-wear')?.addEventListener('input', e => {
    const f=document.getElementById('wear-fill'); if(f)f.style.width=`${Math.min(e.target.value,100)}%`;
  });
  document.getElementById('insp-elong')?.addEventListener('input', e => {
    const f=document.getElementById('elong-fill'); if(f)f.style.width=`${Math.min(e.target.value/4*100,100)}%`;
  });
  document.getElementById('insp-crack')?.addEventListener('input', e => {
    const f=document.getElementById('crack-fill'); if(f)f.style.width=`${Math.min(e.target.value/50*100,100)}%`;
  });
}

function setupDragDrop() {
  const drop = document.getElementById('upload-drop');
  if (!drop) return;
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', async e => {
    e.preventDefault(); drop.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith('.csv')) {
      document.getElementById('upload-file').files = e.dataTransfer.files;
      await uploadDataset();
    } else toast('Please drop a CSV file.', 'error');
  });
}

function showOverlay(text, sub) {
  setEl('overlay-text', text);
  setEl('overlay-sub', sub);
  document.getElementById('overlay')?.classList.remove('hidden');
}
function hideOverlay() { document.getElementById('overlay')?.classList.add('hidden'); }

function toast(msg, type='info') {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.className = `toast ${type}`;
  el.classList.remove('hidden');
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => el.classList.add('hidden'), 4000);
}

// Legacy step navigation — kept so any old calls don't break
function goToStep(n) {
  if (n === 1) showView('view-hub');
  else if (n === 2) showView('view-setup');
  else if (n === 3) showView('view-inspect');
}
