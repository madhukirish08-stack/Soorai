/**
 * Belt RUL — frontend
 * ===================
 * Views (hash routes): #fleet, #belt/<id>, #types, #data, #models, #users
 * Planner: fleet + belt detail. Engineer: everything.
 *
 * CSP: no inline handlers. All clicks go through one delegated listener on
 * [data-action] elements (see ACTIONS). Keep it that way - the server's
 * Content-Security-Policy blocks inline scripts.
 */
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  user: null,
  fields: {},            // canonical schema fields
  types: [],             // belt types
  families: [],
  degFields: [],
  fleet: [],
  importReport: null,   // last bulk-import dry-run: { kind, file, report }
  accuracy: null,       // /api/accuracy (engineer)
  filter: { q: '', type: '' },
  belt: null,            // current belt detail payload
  editBelt: null,        // belt being edited in the belt form (null = new)
  editType: null,        // belt type being edited (null = new)
  datasets: [],
  models: [],
  compareSel: new Set(),
  trainPreselect: null,  // dataset specs to preselect after "Retrain + data"
  mapping: { dataset: null, proposal: null, warningsSeen: false },
  chart: null,
  pollTimer: null,
};

const TIER_LABEL = { red: 'Critical', amber: 'Watch', green: 'Healthy', unknown: 'No data' };
const TIER_RANK  = { red: 0, amber: 1, unknown: 2, green: 3 };
const FAMILY_LABEL = {
  classical_wrapped: 'Classical wrapped', wedge: 'Wedge / narrow', raw_edge_cogged: 'Raw-edge cogged',
  poly_v: 'Poly-V / multi-ribbed', timing: 'Timing / synchronous', specialty: 'Specialty',
};

// ── Helpers ────────────────────────────────────────────────────────────────
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function fmt(n, digits = 4) {
  if (n === null || n === undefined || n === '' || Number.isNaN(Number(n))) return '—';
  return Number(n).toLocaleString(undefined, { maximumSignificantDigits: digits });
}
// Survival models give a median RUL plus an 80 % interval; legacy point models have no interval.
function rulRange(p) {
  return p && p.rul_low != null && p.rul_high != null ? `${Math.round(p.rul_low).toLocaleString()}–${Math.round(p.rul_high).toLocaleString()} h` : null;
}
function pct(v) { return v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`; }
function hours(n) { return n === null || n === undefined ? '—' : `${Math.round(n).toLocaleString()} h`; }
function date(iso) { return iso ? new Date(iso).toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' }) : '—'; }
function typeName(id) { return state.types.find(t => t.id === id)?.name ?? `Type ${id}`; }
function fieldLabel(f) {
  const s = state.fields[f];
  if (s) return s.label;
  if (f.startsWith('type_')) return `Belt type: ${typeName(Number(f.slice(5)))}`;
  return f.replace(/^x_/, '').replace(/_/g, ' ');
}
function unit(f) { const u = state.fields[f]?.unit; return u ? ` <span class="unit">(${esc(u)})</span>` : ''; }
function fieldsOfGroup(group) { return Object.entries(state.fields).filter(([, s]) => s.group === group).map(([k]) => k); }
function isEngineer() { return state.user?.role === 'engineer'; }
function tierChip(tier) { return `<span class="tier tier-${tier}">${TIER_LABEL[tier] || tier}</span>`; }

class ApiError extends Error {
  constructor(msg, status, data) { super(msg); this.status = status; this.data = data; }
}

async function api(path, { method = 'GET', body, form } = {}) {
  const opts = { method, headers: {}, credentials: 'same-origin' };
  if (form) opts.body = form;
  else if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (res.status === 401 && !path.startsWith('/api/auth/')) {
    showLogin('Your session expired. Please sign in again.');
    throw new ApiError('Session expired', 401, data);
  }
  if (!res.ok) throw new ApiError(errorText(data) || `Request failed (${res.status})`, res.status, data);
  return data;
}

function errorText(data) {
  if (!data) return '';
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
  if (Array.isArray(data.errors)) return data.errors.join(' ');
  return '';
}

function toast(msg, type = 'info') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `toast ${type}`;
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => el.classList.add('hidden'), 4500);
}
function showOverlay(text, sub = '') {
  $('#overlay-text').textContent = text;
  $('#overlay-sub').textContent = sub;
  $('#overlay').classList.remove('hidden');
}
function hideOverlay() { $('#overlay').classList.add('hidden'); }

async function guarded(fn) {
  try { return await fn(); }
  catch (e) { hideOverlay(); if (e.status !== 401) toast(e.message, 'error'); }
}

function readNumber(input) {
  const v = input.value.trim();
  return v === '' ? null : Number(v);
}

// ── Boot / auth ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  try {
    state.user = await api('/api/auth/me');
    await startApp();
  } catch {
    showLogin();
  }
});

function showLogin(message) {
  state.user = null;
  stopPolling();
  $('#app-view').classList.add('hidden');
  $('#login-view').classList.remove('hidden');
  const err = $('#login-error');
  err.textContent = message || '';
  err.classList.toggle('hidden', !message);
  $('#login-username').focus();
}

async function login(e) {
  e.preventDefault();
  const err = $('#login-error');
  err.classList.add('hidden');
  try {
    state.user = await api('/api/auth/login', { method: 'POST', body: {
      username: $('#login-username').value.trim(), password: $('#login-password').value,
    } });
    $('#login-password').value = '';
    await startApp();
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove('hidden');
  }
}

async function startApp() {
  $('#login-view').classList.add('hidden');
  $('#app-view').classList.remove('hidden');
  $('#user-name').textContent = state.user.username;
  $('#user-role').textContent = state.user.role;
  const nav = [['fleet', 'Fleet']];
  if (isEngineer()) nav.push(['types', 'Belt Types'], ['data', 'Data & Training'], ['models', 'Models'], ['users', 'Users']);
  $('#main-nav').innerHTML = nav.map(([v, l]) => `<a href="#${v}" class="nav-link" data-nav="${v}">${l}</a>`).join('');
  const [schema] = await Promise.all([api('/api/schema'), loadTypes()]);
  state.fields = schema.fields;
  route();
}

async function loadTypes() {
  const res = await api('/api/belt-types');
  state.types = res.types;
  state.families = res.families;
  state.degFields = res.degradation_fields;
}

// ── Routing ────────────────────────────────────────────────────────────────
window.addEventListener('hashchange', () => state.user && route());

function route() {
  const [view, id] = (location.hash.slice(1) || 'fleet').split('/');
  const allowed = isEngineer() ? ['fleet', 'belt', 'types', 'data', 'models', 'users'] : ['fleet', 'belt'];
  const target = allowed.includes(view) ? view : 'fleet';
  stopPolling();
  $$('.view').forEach(v => v.classList.add('hidden'));
  $(`#view-${target}`).classList.remove('hidden');
  $$('.nav-link').forEach(a => a.classList.toggle('active', a.dataset.nav === (target === 'belt' ? 'fleet' : target)));
  window.scrollTo({ top: 0 });
  const render = { fleet: renderFleet, belt: () => renderBelt(Number(id)), types: renderTypes,
                   data: renderData, models: renderModels, users: renderUsers }[target];
  guarded(render);
}

// ═══════════════════════════════════════════════════════════════════════════
// FLEET
// ═══════════════════════════════════════════════════════════════════════════
async function renderFleet() {
  state.fleet = await api('/api/belts');
  const view = $('#view-fleet');
  const typeOpts = state.types.map(t => `<option value="${t.id}">${esc(t.name)}</option>`).join('');
  view.innerHTML = `
    <div class="view-header">
      <div>
        <h1 class="panel-title">Belt Fleet</h1>
        <p class="panel-subtitle">Most urgent belts first. Click a belt to record an inspection.</p>
      </div>
      <div class="header-actions">
        <button class="btn btn-outline" data-action="open-import">⬆ Import from CSV</button>
        <button class="btn btn-primary" data-action="new-belt">+ Add belt</button>
      </div>
    </div>
    <div id="fleet-summary" class="summary-row"></div>
    <div id="import-card"></div>
    <div id="belt-form-card"></div>
    <div class="card">
      <div class="toolbar">
        <input type="text" id="fleet-search" placeholder="Search tag or location…" value="${esc(state.filter.q)}" />
        <select id="fleet-type"><option value="">All belt types</option>${typeOpts}</select>
      </div>
      <div class="table-wrap"><table class="data-table" id="fleet-table"></table></div>
    </div>`;
  $('#fleet-type').value = state.filter.type;
  $('#fleet-search').addEventListener('input', e => { state.filter.q = e.target.value; drawFleetTable(); });
  $('#fleet-type').addEventListener('change', e => { state.filter.type = e.target.value; drawFleetTable(); });
  drawFleetSummary();
  drawFleetTable();
}

function fleetTier(b) {
  if (b.status !== 'in_service') return null;
  return b.prediction?.tier ?? 'unknown';
}

function drawFleetSummary() {
  const counts = { red: 0, amber: 0, green: 0, unknown: 0 };
  state.fleet.forEach(b => { const t = fleetTier(b); if (t) counts[t]++; });
  const noModel = state.fleet.filter(b => b.status === 'in_service' && !b.has_active_model).length;
  $('#fleet-summary').innerHTML = ['red', 'amber', 'green', 'unknown'].map(t => `
      <div class="summary-tile tile-${t}"><div class="summary-num">${counts[t]}</div><div class="summary-label">${TIER_LABEL[t]}</div></div>`).join('') +
    (noModel ? `<div class="notice notice-warn summary-notice">${noModel} belt(s) have no active model for their type — RUL is unavailable for them${isEngineer() ? ' (see Models)' : ' until an engineer activates one'}.</div>` : '');
}

function drawFleetTable() {
  const q = state.filter.q.toLowerCase();
  const rows = state.fleet
    .filter(b => (!state.filter.type || String(b.belt_type_id) === state.filter.type) &&
                 (!q || b.tag.toLowerCase().includes(q) || (b.location || '').toLowerCase().includes(q)))
    .sort((a, b) => {
      const ta = fleetTier(a), tb = fleetTier(b);
      if (!ta !== !tb) return ta ? -1 : 1;                       // out-of-service last
      if (ta && TIER_RANK[ta] !== TIER_RANK[tb]) return TIER_RANK[ta] - TIER_RANK[tb];
      return (a.prediction?.rul_hours ?? Infinity) - (b.prediction?.rul_hours ?? Infinity);
    });
  const table = $('#fleet-table');
  if (!rows.length) {
    table.innerHTML = `<tbody><tr><td class="empty-state">${state.fleet.length ? 'No belts match the filter.' : 'No belts yet. Click “Add belt” to register the first one.'}</td></tr></tbody>`;
    return;
  }
  table.innerHTML = `
    <thead><tr><th>Status</th><th>Tag</th><th>Type</th><th>Location</th><th>Op. hours</th><th>RUL</th><th>Health</th><th>Last inspection</th><th>#</th></tr></thead>
    <tbody>${rows.map(b => {
      const p = b.prediction;
      const status = b.status === 'in_service' ? tierChip(fleetTier(b)) : `<span class="tier tier-out">${b.status === 'failed' ? 'Failed' : 'Replaced'}</span>`;
      return `<tr class="clickable" data-action="open-belt" data-id="${b.id}">
        <td>${status}</td><td class="strong">${esc(b.tag)}</td><td>${esc(b.belt_type_name)}</td><td class="wrap">${esc(b.location)}</td>
        <td>${hours(b.last_hour)}</td>
        <td class="rul-cell">${p ? (p.rul_hours === null ? '<span class="muted">no model</span>' : `${hours(p.rul_hours)}${rulRange(p) ? `<div class="hint">${rulRange(p)}</div>` : ''}`) : '—'}</td>
        <td>${p?.health_score ?? '—'}${p?.health_score != null ? '%' : ''}</td>
        <td>${date(b.last_inspected_at)}</td><td>${b.inspection_count}</td></tr>`;
    }).join('')}</tbody>`;
}

// ── Belt create / edit form ────────────────────────────────────────────────
function beltFormHtml(belt) {
  const st = belt?.static || {};
  const typeOpts = state.types.map(t => `<option value="${t.id}"${belt?.belt_type_id === t.id ? ' selected' : ''}>${esc(t.name)}</option>`).join('');
  const statics = fieldsOfGroup('static').map(f => `
    <div class="form-group"><label>${esc(fieldLabel(f))}${unit(f)}</label>
      <input type="number" step="any" data-static="${f}" value="${st[f] ?? ''}" /></div>`).join('');
  return `
    <div class="card">
      <div class="card-title">${belt ? `Edit belt ${esc(belt.tag)}` : 'Register a belt'}</div>
      <div class="form-row cols-2">
        <div class="form-group"><label>Belt tag / ID *</label><input type="text" id="bf-tag" maxlength="60" value="${esc(belt?.tag)}" placeholder="e.g. PUMP-07-B1" /></div>
        <div class="form-group"><label>Belt type *</label><select id="bf-type">${typeOpts}</select></div>
        <div class="form-group"><label>Location / drive</label><input type="text" id="bf-location" maxlength="120" value="${esc(belt?.location)}" placeholder="e.g. Cooling tower pump 7" /></div>
        <div class="form-group"><label>Installed on</label><input type="date" id="bf-installed" value="${esc(belt?.installed_on)}" /></div>
      </div>
      <p class="card-desc">Drive details (fill what you know — they improve the prediction):</p>
      <div class="form-row cols-3">${statics}</div>
      <div class="action-row">
        <button class="btn btn-ghost" data-action="cancel-belt-form">Cancel</button>
        <button class="btn btn-primary" data-action="save-belt">${belt ? 'Save changes' : 'Register belt'}</button>
      </div>
    </div>`;
}

async function saveBelt() {
  const staticVals = {};
  $$('[data-static]').forEach(i => { const v = readNumber(i); if (v !== null) staticVals[i.dataset.static] = v; });
  const body = {
    tag: $('#bf-tag').value.trim(), belt_type_id: Number($('#bf-type').value),
    location: $('#bf-location').value.trim(), installed_on: $('#bf-installed').value || null, static: staticVals,
  };
  if (!body.tag) { toast('Enter a belt tag.', 'error'); return; }
  if (state.editBelt) {
    await api(`/api/belts/${state.editBelt.id}`, { method: 'PUT', body });
    toast('Belt updated.', 'success');
    state.editBelt = null;
    renderBelt(state.belt.belt.id);
  } else {
    const res = await api('/api/belts', { method: 'POST', body });
    toast('Belt registered.', 'success');
    location.hash = `belt/${res.id}`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// BELT DETAIL
// ═══════════════════════════════════════════════════════════════════════════
async function renderBelt(id) {
  state.belt = await api(`/api/belts/${id}`);
  const { belt, belt_type: bt, inspections, predictions, model } = state.belt;
  const latest = predictions[predictions.length - 1];
  const inService = belt.status === 'in_service';
  const lastHour = inspections.length ? inspections[inspections.length - 1].hour : 0;

  const banners = [];
  if (!bt.thresholds_confirmed) banners.push(`<div class="notice notice-warn">⚠ The failure limits for <strong>${esc(bt.name)}</strong> are unconfirmed placeholders — health scores are indicative only.${isEngineer() ? ' Confirm them under Belt Types.' : ''}</div>`);
  if (!model) banners.push(`<div class="notice notice-warn">No active model for <strong>${esc(bt.name)}</strong> belts — inspections are recorded and health is shown, but RUL is unavailable until an engineer activates a model.</div>`);
  else if (model.rul_estimated) banners.push(`<div class="notice notice-info">The active model (v${model.id}) was trained partly on <strong>estimated</strong> belt lives. Treat RUL as indicative and record real failures to improve it.</div>`);
  if (!inService) banners.push(`<div class="notice notice-error">This belt was <strong>${belt.status}</strong> at ${hours(belt.end_hour)}${belt.end_reason ? ` — ${esc(belt.end_reason)}` : ''}.</div>`);

  const opFields = fieldsOfGroup('operating');
  const extraFields = (model?.feature_cols || []).filter(f => f.startsWith('x_'));
  const inputFor = f => {
    const med = model?.feature_stats?.[f]?.median;
    return `<div class="form-group"><label>${esc(fieldLabel(f))}${unit(f)}</label>
      <input type="number" step="any" data-reading="${f}" placeholder="${med !== undefined ? `typical ${fmt(med, 3)}` : ''}" /></div>`;
  };
  const degInputs = state.degFields.map(f => `
    <div class="form-group"><label>${esc(fieldLabel(f))}${unit(f)}</label>
      <input type="number" step="any" min="0" data-reading="${f}" />
      <span class="hint">Failure limit: ${bt.thresholds[f] !== undefined ? `${fmt(bt.thresholds[f])} ${esc(state.fields[f]?.unit || '')}` : 'not set'}</span></div>`).join('');

  $('#view-belt').innerHTML = `
    <a href="#fleet" class="back-link">← Fleet</a>
    <div class="view-header">
      <div>
        <h1 class="panel-title">${esc(belt.tag)}</h1>
        <p class="panel-subtitle">${esc(bt.name)}${belt.location ? ` · ${esc(belt.location)}` : ''}${belt.installed_on ? ` · installed ${date(belt.installed_on)}` : ''}</p>
      </div>
      <button class="btn btn-ghost" data-action="edit-belt">Edit belt</button>
    </div>
    ${banners.join('')}
    ${outcomeBannerHtml(belt, inspections)}
    <div id="belt-form-card"></div>
    <div class="belt-layout">
      <div>
        ${inService ? `
        <div class="card">
          <div class="card-title">📝 Record inspection</div>
          <p class="card-desc">Only operating hours are required. Leave a field empty if it wasn't measured — never enter 0 for “not measured”.</p>
          <div class="form-row cols-2">
            <div class="form-group full-width"><label>Operating hours since installation *</label>
              <input type="number" step="any" min="0" id="insp-hour" value="${inspections.length ? lastHour + 250 : 0}" /></div>
          </div>
          <div class="sub-title">Degradation measurements</div>
          <div class="form-row cols-3">${degInputs}</div>
          <div class="sub-title">Operating conditions</div>
          <div class="form-row cols-3">${opFields.map(inputFor).join('')}${extraFields.map(inputFor).join('')}</div>
          <div class="action-row"><button class="btn btn-primary btn-large" data-action="add-inspection">Save inspection &amp; predict</button></div>
        </div>` : ''}
        <div class="card">
          <div class="card-title">🏁 End of life</div>
          ${inService ? `
          <p class="card-desc">Recording when and why a belt failed or was replaced is the most valuable data for improving predictions.</p>
          <div class="form-row cols-3">
            <div class="form-group"><label>Outcome</label><select id="eol-status"><option value="failed">Failed</option><option value="replaced">Replaced (planned)</option></select></div>
            <div class="form-group"><label>At operating hour</label><input type="number" step="any" min="0" id="eol-hour" value="${lastHour || ''}" /></div>
            <div class="form-group"><label>Reason / failure mode</label><input type="text" id="eol-reason" maxlength="300" placeholder="e.g. Rib cracking" /></div>
          </div>
          <div class="action-row"><button class="btn btn-outline" data-action="end-of-life">Record</button></div>` : `
          <p class="card-desc">Recorded as ${esc(belt.status)} at ${hours(belt.end_hour)}.</p>
          <div class="action-row"><button class="btn btn-ghost" data-action="undo-end-of-life">Undo — back in service</button></div>`}
        </div>
      </div>
      <div>${latestCardHtml(latest, bt, model)}</div>
    </div>
    ${inspections.length ? `
    <div class="card"><div class="card-title">📈 History</div><canvas id="belt-chart" height="110"></canvas></div>
    <div class="card"><div class="card-title">📋 Inspection log</div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>Hour</th><th>RUL now</th><th>RUL at the time</th>${inService ? '' : '<th>What happened</th>'}<th>Health</th><th>Elong.</th><th>Wear</th><th>Crack</th><th>By</th><th></th></tr></thead>
        <tbody>${inspections.map((i, k) => ({ i, p: predictions[k] })).reverse().map(({ i, p }) => `
          <tr><td>${hours(i.hour)}</td><td class="rul-cell">${hours(p.rul_hours)}</td>
          <td class="muted">${i.predicted_rul !== null ? `${hours(i.predicted_rul)}${i.predicted_low != null ? ` <span class="hint">(${Math.round(i.predicted_low).toLocaleString()}–${Math.round(i.predicted_high).toLocaleString()})</span>` : ''}${i.model_id ? ` · v${i.model_id}` : ''}` : '—'}</td>
          ${inService ? '' : `<td>${actualCellHtml(i)}</td>`}
          <td>${p.health_score ?? '—'}${p.health_score != null ? '%' : ''} ${tierChip(p.tier)}</td>
          <td>${fmt(i.readings.elongation_pct)}</td><td>${fmt(i.readings.wear_pct)}</td><td>${fmt(i.readings.crack_depth_mm)}</td>
          <td class="muted">${esc(i.created_by_name || '')}</td>
          <td>${inService ? `<button class="btn-icon-only" title="Delete this inspection" data-action="delete-inspection" data-id="${i.id}">✕</button>` : ''}</td></tr>`).join('')}
        </tbody></table></div></div>` : ''}`;
  drawBeltChart(predictions, inspections);
}

function latestCardHtml(p, bt, model) {
  if (!p) return `<div class="results-empty"><div class="results-empty-icon">🔮</div><div>Record the first inspection to see this belt's health and remaining life.</div></div>`;
  const ratios = p.degradation_ratios || {};
  const gauge = f => {
    const r = ratios[f];
    const off = r === undefined ? 213.6 : 213.6 * (1 - Math.min(Math.max(r, 0), 1));
    return `<div class="gauge"><div class="gauge-label">${esc(fieldLabel(f))}</div><div class="gauge-ring">
      <svg viewBox="0 0 80 80"><circle cx="40" cy="40" r="34" class="gauge-bg"/>
      <circle cx="40" cy="40" r="34" class="gauge-fg gauge-${f}" stroke-dasharray="213.6" stroke-dashoffset="${off}"/></svg>
      <div class="gauge-pct">${r === undefined ? 'n/a' : `${Math.round(r * 100)}%`}</div></div></div>`;
  };
  return `
    <div class="rul-card tier-border-${p.tier}">
      <div class="rul-header">Remaining useful life ${tierChip(p.tier)}</div>
      <div class="rul-value">${p.rul_hours === null ? '—' : `${Math.round(p.rul_hours).toLocaleString()} h`}</div>
      <div class="rul-sub">${p.rul_hours === null ? 'No active model for this belt type' :
        `At ${hours(p.hour)} · model v${p.model_id}${p.limit_reached ? ' · <strong>failure limit reached</strong>' : ''}`}</div>
      ${rulRange(p) ? `
      <div class="rul-interval" title="The true remaining life should fall in this range ${Math.round(p.interval_level * 100)} times out of 100">
        <span class="rul-interval-label">${Math.round(p.interval_level * 100)}% interval</span>
        <span class="rul-interval-value">RUL ${rulRange(p)}</span>
      </div>` : ''}
      <div class="health-grid">
        <div class="health-item"><div class="health-label">Health</div><div class="health-value">${p.health_score ?? '—'}${p.health_score != null ? '%' : ''}</div></div>
        <div class="health-item"><div class="health-label">Closest to limit</div><div class="health-value risk">${esc(p.dominant_driver || '—')}</div></div>
        ${model ? `<div class="health-item"><div class="health-label">${model.metrics.interval_coverage != null ? 'Interval hit rate (held-out)' : 'Model error (held-out)'}</div><div class="health-value">${model.metrics.interval_coverage != null ? pct(model.metrics.interval_coverage) : `± ${hours(model.metrics.mae_hours)}`}</div></div>` : ''}
      </div>
      <div class="gauge-row">${state.degFields.map(gauge).join('')}</div>
      <div class="hint">Gauges show each measurement as a share of the ${esc(bt.name)} failure limit.</div>
    </div>`;
}

// What actually happened after an inspection, next to what the model said at the time.
function actualCellHtml(i) {
  if (i.actual_rul === null || i.actual_rul === undefined) return '<span class="muted">—</span>';
  if (i.actual_kind === 'replaced') return `<span class="muted">lasted ≥ ${hours(i.actual_rul)} more</span>`;
  const mark = i.in_interval === true ? '<span class="hit-yes" title="The actual remaining life was inside the predicted range">✓ in range</span>'
             : i.in_interval === false ? '<span class="hit-no" title="The actual remaining life was outside the predicted range">✗ outside</span>' : '';
  return `${hours(i.actual_rul)} ${mark}`;
}

function outcomeBannerHtml(belt, inspections) {
  if (belt.status === 'in_service') return '';
  const judged = inspections.filter(i => i.actual_rul !== null && i.predicted_rul !== null && i.model_id);
  if (!judged.length) return '';
  const last = judged[judged.length - 1];
  if (belt.status === 'replaced') {
    return `<div class="notice notice-info"><strong>Replaced, not failed.</strong> At its last inspection (${hours(last.hour)}) the model predicted ${hours(last.predicted_rul)} left; the belt then ran at least ${hours(last.actual_rul)} more before it was replaced, so this belt is not counted as a failure when judging model accuracy.</div>`;
  }
  const err = last.predicted_rul - last.actual_rul;
  const range = last.predicted_low != null ? ` (range ${Math.round(last.predicted_low).toLocaleString()}–${Math.round(last.predicted_high).toLocaleString()} h)` : '';
  const verdict = last.in_interval === true ? 'The actual outcome was inside the predicted range.' : last.in_interval === false ? 'The actual outcome was outside the predicted range.' : '';
  return `<div class="notice notice-info"><strong>How the model did.</strong> At its last inspection (${hours(last.hour)}, model v${last.model_id}) it predicted ${hours(last.predicted_rul)} left${range}. The belt failed ${hours(last.actual_rul)} later — the prediction was ${err > 0 ? `${hours(err)} too optimistic` : err < 0 ? `${hours(-err)} too cautious` : 'exact'}. ${verdict}</div>`;
}

function drawBeltChart(predictions, inspections = []) {
  if (state.chart) { state.chart.destroy(); state.chart = null; }
  const canvas = $('#belt-chart');
  if (!canvas || typeof Chart === 'undefined') return;
  state.chart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: predictions.map(p => `${Math.round(p.hour)} h`),
      datasets: [
        { label: '80% interval', data: predictions.map(p => p.rul_high), borderColor: 'transparent', backgroundColor: 'rgba(79,142,247,0.14)',
          pointRadius: 0, fill: '+1', tension: 0.3, yAxisID: 'y' },
        { label: 'RUL low', data: predictions.map(p => p.rul_low), borderColor: 'transparent', pointRadius: 0, fill: false, tension: 0.3, yAxisID: 'y' },
        { label: 'RUL median (h)', data: predictions.map(p => p.rul_hours), borderColor: '#4f8ef7', backgroundColor: 'rgba(79,142,247,0.08)',
          borderWidth: 2.5, pointRadius: 4, tension: 0.3, fill: false, yAxisID: 'y' },
        { label: 'Health (%)', data: predictions.map(p => p.health_score), borderColor: '#22d3a0', borderDash: [5, 4],
          borderWidth: 2, pointRadius: 3, tension: 0.3, yAxisID: 'y1' },
        ...(inspections.some(i => i.actual_kind === 'failed') ? [
          { label: 'Actual remaining life (h)', data: inspections.map(i => i.actual_kind === 'failed' ? i.actual_rul : null),
            borderColor: '#f59e0b', borderWidth: 2, pointRadius: 4, pointStyle: 'rectRot', tension: 0.3, spanGaps: true, yAxisID: 'y' }] : []),
      ],
    },
    options: {
      responsive: true, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#94a3b8', filter: item => item.text !== 'RUL low' } } },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(30,45,71,0.6)' } },
        y: { position: 'left', title: { display: true, text: 'RUL (h)', color: '#4f8ef7' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(30,45,71,0.6)' } },
        y1: { position: 'right', min: 0, max: 100, title: { display: true, text: 'Health (%)', color: '#22d3a0' }, ticks: { color: '#64748b' }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

async function addInspection() {
  const hour = readNumber($('#insp-hour'));
  if (hour === null || hour < 0) { toast('Enter the operating hours.', 'error'); return; }
  const readings = {};
  $$('[data-reading]').forEach(i => { const v = readNumber(i); if (v !== null) readings[i.dataset.reading] = v; });
  if (!state.degFields.some(f => f in readings)) {
    if (!confirm('No degradation measurement entered. The health score needs at least one. Save anyway?')) return;
  }
  const id = state.belt.belt.id;
  await api(`/api/belts/${id}/inspections`, { method: 'POST', body: { hour, readings } });
  toast('Inspection saved.', 'success');
  renderBelt(id);
}

async function endOfLife(undo) {
  const id = state.belt.belt.id;
  const body = undo ? { status: 'in_service' } : {
    status: $('#eol-status').value, hour: readNumber($('#eol-hour')), reason: $('#eol-reason').value.trim(),
  };
  if (!undo && body.hour === null) { toast('Enter the operating hour.', 'error'); return; }
  if (!undo && !confirm(`Record ${state.belt.belt.tag} as ${body.status} at ${body.hour} h? No further inspections can be added.`)) return;
  await api(`/api/belts/${id}/end-of-life`, { method: 'PUT', body });
  toast(undo ? 'Belt back in service.' : 'Recorded. Thank you — this improves future models.', 'success');
  renderBelt(id);
}

// ═══════════════════════════════════════════════════════════════════════════
// BELT TYPES (engineer)
// ═══════════════════════════════════════════════════════════════════════════
async function renderTypes() {
  await loadTypes();
  $('#view-types').innerHTML = `
    <div class="view-header">
      <div><h1 class="panel-title">Belt Types</h1>
      <p class="panel-subtitle">Each type has its own failure limits. Dataset values are matched to a type by its name or any alias (section codes, product names).</p></div>
      <button class="btn btn-primary" data-action="new-type">+ Add belt type</button>
    </div>
    <div id="type-form-card"></div>
    <div class="card"><div class="table-wrap"><table class="data-table">
      <thead><tr><th>Name</th><th>Family</th><th>Aliases</th><th>Failure limits (elong. / wear / crack)</th><th>Limits</th><th>Belts</th><th>Active model</th><th></th></tr></thead>
      <tbody>${state.types.map(t => `<tr>
        <td class="strong">${esc(t.name)}</td><td>${esc(FAMILY_LABEL[t.family] || t.family)}</td>
        <td class="muted small wrap">${esc(t.aliases.join(', '))}</td>
        <td>${state.degFields.map(f => t.thresholds[f] !== undefined ? `${fmt(t.thresholds[f])} ${esc(state.fields[f]?.unit || '')}` : '—').join(' / ')}</td>
        <td>${t.thresholds_confirmed ? '<span class="tier tier-green">Confirmed</span>' : '<span class="tier tier-amber">Unconfirmed</span>'}</td>
        <td>${t.belt_count}</td><td>${t.active_model_id ? `v${t.active_model_id}` : '<span class="muted">none</span>'}</td>
        <td><button class="btn btn-ghost btn-sm" data-action="edit-type" data-id="${t.id}">Edit</button></td></tr>`).join('')}
      </tbody></table></div></div>`;
}

function typeFormHtml(t) {
  const famOpts = state.families.map(f => `<option value="${f}"${t?.family === f ? ' selected' : ''}>${esc(FAMILY_LABEL[f] || f)}</option>`).join('');
  return `<div class="card">
    <div class="card-title">${t ? `Edit ${esc(t.name)}` : 'New belt type'}</div>
    <div class="form-row cols-2">
      <div class="form-group"><label>Name *</label><input type="text" id="tf-name" maxlength="80" value="${esc(t?.name)}" /></div>
      <div class="form-group"><label>Family</label><select id="tf-family">${famOpts}</select></div>
      <div class="form-group"><label>Aliases (comma separated)</label><input type="text" id="tf-aliases" value="${esc(t?.aliases?.join(', '))}" placeholder="e.g. SPB, SPBX, SpacesaVer" /></div>
      <div class="form-group"><label>Cord material</label><input type="text" id="tf-cord" maxlength="80" value="${esc(t?.cord_material)}" placeholder="e.g. Polyester, Aramid" /></div>
    </div>
    <p class="card-desc">Failure limits — the point at which a belt of this type must be replaced. Leave a limit empty if it doesn't apply.</p>
    <div class="form-row cols-3">${state.degFields.map(f => `
      <div class="form-group"><label>${esc(fieldLabel(f))}${unit(f)}</label>
      <input type="number" step="any" min="0" data-threshold="${f}" value="${t?.thresholds?.[f] ?? ''}" /></div>`).join('')}</div>
    <div class="form-group full-width"><label>Notes</label><input type="text" id="tf-notes" maxlength="1000" value="${esc(t?.notes)}" /></div>
    <label class="check-row"><input type="checkbox" id="tf-confirmed" ${t?.thresholds_confirmed ? 'checked' : ''}/>
      These limits are confirmed (manufacturer data or site engineering standard)</label>
    <div class="action-row">
      <button class="btn btn-ghost" data-action="cancel-type-form">Cancel</button>
      <button class="btn btn-primary" data-action="save-type">Save belt type</button>
    </div></div>`;
}

async function saveType() {
  const thresholds = {};
  $$('[data-threshold]').forEach(i => { const v = readNumber(i); if (v !== null) thresholds[i.dataset.threshold] = v; });
  const body = {
    name: $('#tf-name').value.trim(), family: $('#tf-family').value,
    aliases: $('#tf-aliases').value.split(',').map(s => s.trim()).filter(Boolean),
    cord_material: $('#tf-cord').value.trim(), notes: $('#tf-notes').value.trim(),
    thresholds, thresholds_confirmed: $('#tf-confirmed').checked,
  };
  if (state.editType) await api(`/api/belt-types/${state.editType.id}`, { method: 'PUT', body });
  else await api('/api/belt-types', { method: 'POST', body });
  toast('Belt type saved.', 'success');
  state.editType = null;
  renderTypes();
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA & TRAINING (engineer)
// ═══════════════════════════════════════════════════════════════════════════
async function renderData() {
  [state.datasets, state.models] = await Promise.all([api('/api/datasets'), api('/api/models')]);
  const statusLabel = { approved: '<span class="tier tier-green">Mapping approved</span>',
                        not_needed: '<span class="tier tier-unknown">Model schema</span>',
                        required: '<span class="tier tier-amber">Mapping required</span>' };
  const pre = state.trainPreselect;
  const preMap = new Map((pre || []).map(d => [d.filename, d]));
  $('#view-data').innerHTML = `
    <div class="view-header"><div><h1 class="panel-title">Data &amp; Training</h1>
      <p class="panel-subtitle">1. Upload plant data → 2. review the column mapping → 3. train a new model version → 4. review and activate it under Models.</p></div></div>

    <div class="card">
      <div class="card-title">📂 Datasets
        <button class="btn btn-outline btn-sm" data-action="dataset-from-fleet" title="Turns the inspections and failures recorded in the app into a training dataset">Create from fleet records</button></div>
      <label class="upload-drop" id="upload-drop">
        <input type="file" id="upload-file" accept=".csv" hidden />
        <div class="upload-text">Drop a CSV here or <span class="link">browse</span></div>
        <div class="upload-hint">Any column layout — columns are matched to the model's fields in the review step.</div>
      </label>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>File</th><th>Size</th><th>Columns</th><th>Mapping</th><th></th></tr></thead>
        <tbody>${state.datasets.length ? state.datasets.map(d => `<tr>
          <td class="strong">${esc(d.filename)}</td><td>${d.size_mb} MB</td><td>${d.columns.length}</td><td>${statusLabel[d.mapping_status]}</td>
          <td class="row-actions"><button class="btn btn-outline btn-sm" data-action="open-mapping" data-name="${esc(d.filename)}">Review mapping</button>
          <button class="btn-icon-only" title="Delete dataset" data-action="delete-dataset" data-name="${esc(d.filename)}">✕</button></td></tr>`).join('')
          : '<tr><td colspan="5" class="empty-state">No datasets uploaded yet.</td></tr>'}</tbody>
      </table></div>
    </div>

    <div id="mapping-card"></div>

    <div class="card">
      <div class="card-title">🧠 Train a new model version</div>
      <p class="card-desc">Select every dataset the new version should learn from. To add new data to an existing model, include that model's datasets plus the new one (Models → “Retrain + data” does this for you).</p>
      <div class="train-list">${state.datasets.map(d => {
        const usable = d.mapping_status !== 'required';
        const sel = preMap.get(d.filename);
        const typeOpts = state.types.map(t => `<option value="${t.id}"${sel?.belt_type_id === t.id ? ' selected' : ''}>${esc(t.name)}</option>`).join('');
        return `<div class="train-item${usable ? '' : ' disabled'}">
          <label class="check-row"><input type="checkbox" data-train="${esc(d.filename)}" ${usable ? '' : 'disabled'} ${sel ? 'checked' : ''}/>
            <span class="strong">${esc(d.filename)}</span>${usable ? '' : ' <span class="muted">— approve its mapping first</span>'}</label>
          ${usable && !d.has_belt_type_column ? `<div class="train-type">Belt type for this file: <select data-train-type="${esc(d.filename)}"><option value="">Choose…</option>${typeOpts}</select></div>` : ''}
        </div>`;
      }).join('') || '<div class="empty-state">Upload a dataset first.</div>'}</div>
      <div class="form-group full-width"><label>Notes for this version</label><input type="text" id="train-notes" maxlength="500" placeholder="e.g. Added Q3 plant data" /></div>
      <div class="action-row"><button class="btn btn-primary btn-large" data-action="train">Train new version</button></div>
      <div id="train-status"></div>
    </div>

    ${state.user.dev_mode ? `
    <div class="card generate-card">
      <div class="card-title">⚡ Synthetic V-belt data <span class="role-chip">dev mode</span></div>
      <p class="card-desc">Simulated run-to-failure data for demos. Not calibrated against JK Fenner tests — never present results as real-world accuracy.</p>
      <div class="form-row cols-3">
        <div class="form-group"><label>Belts</label><input type="number" id="gen-belts" value="300" min="10" max="3000" /></div>
        <div class="form-group"><label>Inspection step (h)</label><input type="number" id="gen-step" value="250" min="10" max="2000" /></div>
        <div class="form-group"><label>Seed</label><input type="number" id="gen-seed" value="42" min="0" /></div>
        <div class="form-group"><label>Censored belts (%)</label><input type="number" id="gen-censor" value="30" min="0" max="90" /></div>
      </div>
      <button class="btn btn-outline" data-action="generate">Generate dataset</button>
    </div>` : ''}`;
  state.trainPreselect = null;
  setupUpload();
  const training = state.models.find(m => m.status === 'training');
  if (training) watchTraining(training.id);
}

function setupUpload() {
  const drop = $('#upload-drop');
  $('#upload-file').addEventListener('change', e => e.target.files[0] && uploadDataset(e.target.files[0]));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) uploadDataset(f);
  });
}

async function uploadDataset(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) { toast('Only CSV files are accepted.', 'error'); return; }
  const form = new FormData();
  form.append('file', file);
  showOverlay('Uploading…', file.name);
  const res = await guarded(() => api('/api/datasets/upload', { method: 'POST', form }));
  hideOverlay();
  if (!res) return;
  toast(`Uploaded ${res.filename} (${res.rows.toLocaleString()} rows).`, 'success');
  await renderData();
  if (res.mapping_status === 'required') openMapping(res.filename);
}

// ── Column mapping review (AI proposes → engineer approves) ────────────────
async function openMapping(filename) {
  showOverlay('Analysing columns…', 'Matching your columns to the model fields.');
  const proposal = await guarded(() => api('/api/schema/propose', { method: 'POST', body: { filename } }));
  hideOverlay();
  if (!proposal) return;
  state.mapping = { dataset: filename, proposal, warningsSeen: false };
  const sourceText = { llm: 'Proposed by Gemini', heuristic: 'Name matching (AI unavailable)', exact: 'Exact name match',
                       saved_profile: 'Re-used saved profile', approved: 'Previously approved' }[proposal.source] || proposal.source;
  const profile = Object.fromEntries((proposal.profile || []).map(p => [p.name, p]));
  const options = [['__ignore__', '— Ignore column —'], ['__extra__', '+ Extra feature (keep as-is)'],
    ...Object.entries(proposal.canonical_fields).map(([k, f]) => [k, `${f.label}${f.unit ? ` (${f.unit})` : ''} · ${k}`])];
  $('#mapping-card').innerHTML = `
    <div class="card mapping-card">
      <div class="card-title">🧭 Column mapping — ${esc(filename)} <span class="mapping-source">${esc(sourceText)}</span></div>
      <p class="card-desc">Check every row, especially low-confidence ones and unit conversions (<code>model value = raw × scale + offset</code>). Nothing is used for training until you approve.</p>
      ${(proposal.warnings || []).map(w => `<div class="notice notice-warn">${esc(w)}</div>`).join('')}
      <div class="table-wrap"><table class="data-table mapping-table">
        <thead><tr><th>Your column</th><th>Sample values</th><th>Maps to</th><th>Confidence</th><th>Scale</th><th>Offset</th><th>Why</th></tr></thead>
        <tbody id="mapping-tbody">${proposal.mapping.map(m => {
          const p = profile[m.source] || {};
          const conf = Math.round((m.confidence ?? 0) * 100);
          return `<tr data-source="${esc(m.source)}">
            <td class="strong">${esc(m.source)}</td>
            <td class="map-samples">${esc((p.samples || []).slice(0, 3).join(', '))}${p.min !== undefined ? `<div class="hint">${fmt(p.min)} … ${fmt(p.max)}</div>` : ''}</td>
            <td><select class="map-target">${options.map(([v, l]) => `<option value="${esc(v)}"${v === m.target ? ' selected' : ''}>${esc(l)}</option>`).join('')}</select></td>
            <td><span class="conf-pill ${conf >= 80 ? 'conf-high' : conf >= 50 ? 'conf-mid' : 'conf-low'}">${conf}%</span></td>
            <td><input type="number" class="map-scale" step="any" value="${m.scale ?? 1}" /></td>
            <td><input type="number" class="map-offset" step="any" value="${m.offset ?? 0}" /></td>
            <td class="map-reason">${esc(m.reason || '')}</td></tr>`;
        }).join('')}</tbody></table></div>
      <div id="mapping-result"></div>
      <div class="action-row">
        <button class="btn btn-ghost" data-action="close-mapping">Cancel</button>
        <button class="btn btn-primary" id="btn-approve-mapping" data-action="approve-mapping">✓ Approve mapping</button>
      </div>
    </div>`;
  // Editing any row resets the "Approve anyway" confirmation (warnings must be re-seen).
  $('#mapping-tbody').addEventListener('input', resetMappingConfirmation);
  $('#mapping-tbody').addEventListener('change', resetMappingConfirmation);
  $('#mapping-card').scrollIntoView({ behavior: 'smooth' });
}

function resetMappingConfirmation() {
  state.mapping.warningsSeen = false;
  const btn = $('#btn-approve-mapping');
  if (btn) btn.textContent = '✓ Approve mapping';
}

// Two-step approval: validate (dry run) and show warnings first; saving with warnings
// needs an explicit "Approve anyway". Keep this — unit/mapping warnings flag silent
// model corruption and must be seen BEFORE the mapping is saved.
async function approveMapping() {
  const mapping = $$('#mapping-tbody tr').map(tr => ({
    source: tr.dataset.source,
    target: $('.map-target', tr).value,
    scale: Number($('.map-scale', tr).value),
    offset: Number($('.map-offset', tr).value),
    confidence: state.mapping.proposal.mapping.find(m => m.source === tr.dataset.source)?.confidence ?? 1,
    reason: $('.map-reason', tr).textContent,
  }));
  if (mapping.some(m => !Number.isFinite(m.scale) || !Number.isFinite(m.offset))) { toast('Scale and offset must be numbers.', 'error'); return; }
  const box = $('#mapping-result');
  const show = d => {
    box.innerHTML = (d.errors || []).map(e => `<div class="notice notice-error">✗ ${esc(e)}</div>`).join('') +
                    (d.warnings || []).map(w => `<div class="notice notice-warn">${esc(w)}</div>`).join('');
  };
  const post = dry_run => api('/api/schema/approve', { method: 'POST', body: {
    filename: state.mapping.dataset, mapping, proposal_source: state.mapping.proposal.source, dry_run } });
  try {
    if (!state.mapping.warningsSeen) {
      const d = await post(true);
      show(d);
      if (d.warnings.length) {
        state.mapping.warningsSeen = true;
        $('#btn-approve-mapping').textContent = '⚠ Approve anyway';
        toast('Review the warnings, then approve again to confirm.', 'info');
        return;
      }
    }
    const d = await post(false);
    show(d);
    box.insertAdjacentHTML('afterbegin', '<div class="notice notice-ok">✓ Mapping approved and saved.</div>');
    toast('Mapping approved.', 'success');
    resetMappingConfirmation();
    const name = state.mapping.dataset;
    await renderData();
    toast(`${name} is ready for training.`, 'success');
  } catch (e) {
    if (e.data?.errors) { show(e.data); toast('The mapping has errors — fix them and approve again.', 'error'); }
    else throw e;
  }
}

// ── Training ───────────────────────────────────────────────────────────────
async function startTraining() {
  const datasets = $$('[data-train]:checked').map(cb => {
    const sel = $(`[data-train-type="${CSS.escape(cb.dataset.train)}"]`);
    return { filename: cb.dataset.train, belt_type_id: sel ? (Number(sel.value) || null) : null };
  });
  if (!datasets.length) { toast('Select at least one dataset.', 'error'); return; }
  const missing = datasets.filter(d => $(`[data-train-type="${CSS.escape(d.filename)}"]`) && !d.belt_type_id);
  if (missing.length) { toast(`Choose the belt type for ${missing.map(d => d.filename).join(', ')}.`, 'error'); return; }
  const res = await api('/api/models/train', { method: 'POST', body: { datasets, notes: $('#train-notes').value.trim() } });
  toast(`Training version v${res.id}…`, 'info');
  watchTraining(res.id);
}

function watchTraining(id) {
  const box = $('#train-status');
  if (box) box.innerHTML = `<div class="notice notice-info"><span class="btn-spinner">⟳</span> Training version v${id}… this usually takes under a minute.</div>`;
  stopPolling();
  state.pollTimer = setInterval(async () => {
    let m;
    try { m = await api(`/api/models/${id}`); } catch { stopPolling(); return; }
    if (m.status === 'training') return;
    stopPolling();
    const box2 = $('#train-status');
    if (m.status === 'failed') {
      if (box2) box2.innerHTML = `<div class="notice notice-error">✗ v${id} failed: ${esc(m.error)}</div>`;
      toast(`Training v${id} failed.`, 'error');
    } else {
      const k = m.info.metrics;
      if (box2) box2.innerHTML = `<div class="notice notice-ok">✓ v${id} trained — held-out belts: average error ± ${hours(k.mae_hours)}${k.c_index != null ? `, concordance ${k.c_index}` : ''}${k.interval_coverage != null ? `, 80% interval covers ${pct(k.interval_coverage)}` : ''}.
        It is <strong>not active yet</strong>. <a href="#models" class="link">Review and activate it under Models →</a></div>`;
      toast(`v${id} trained.`, 'success');
      if (location.hash === '#models') renderModels();
    }
  }, 2500);
}

function stopPolling() { clearInterval(state.pollTimer); state.pollTimer = null; }

async function generateData() {
  showOverlay('Generating synthetic data…', 'Simulating belts to failure.');
  const res = await guarded(() => api('/api/dev/generate-data', { method: 'POST', body: {
    belts: Number($('#gen-belts').value), step: Number($('#gen-step').value), seed: Number($('#gen-seed').value),
    censor_frac: Number($('#gen-censor').value) / 100 } }));
  hideOverlay();
  if (res) { toast(`Created ${res.filename}.`, 'success'); renderData(); }
}

// ═══════════════════════════════════════════════════════════════════════════
// MODELS (engineer)
// ═══════════════════════════════════════════════════════════════════════════
async function renderModels() {
  [state.models, state.accuracy] = await Promise.all([api('/api/models'), api('/api/accuracy'), loadTypes()]);
  state.compareSel = new Set([...state.compareSel].filter(id => state.models.some(m => m.id === id && m.status === 'ready')));
  const m = state.models;
  $('#view-models').innerHTML = `
    <div class="view-header"><div><h1 class="panel-title">Model Versions</h1>
      <p class="panel-subtitle">Every training run is kept. Activate a version per belt type; roll back at any time. Metrics are measured on belts the model never saw.</p></div>
      <button class="btn btn-outline" data-action="compare" ${state.compareSel.size === 2 ? '' : 'disabled'}>Compare selected (${state.compareSel.size}/2)</button>
    </div>
    <div id="compare-card"></div>
    <div id="model-detail-card"></div>
    <div class="card"><div class="table-wrap"><table class="data-table">
      <thead><tr><th></th><th>Version</th><th>Created</th><th>Belt types</th><th>Belts</th><th>Avg error</th><th>Near-failure</th><th>Bias</th><th title="Measured on belts that really failed in your fleet, using the prediction made at each inspection">Real failures</th><th>Status</th><th></th></tr></thead>
      <tbody>${m.length ? m.map(modelRowHtml).join('') : '<tr><td colspan="11" class="empty-state">No model versions yet. Train one under Data &amp; Training.</td></tr>'}</tbody>
    </table></div></div>
    ${accuracyCardHtml(state.accuracy)}`;
  if (m.some(x => x.status === 'training')) {
    stopPolling();
    state.pollTimer = setInterval(async () => {
      const list = await api('/api/models').catch(() => null);
      if (list && !list.some(x => x.status === 'training')) { stopPolling(); renderModels(); }
    }, 3000);
  }
}

function modelRowHtml(m) {
  const i = m.info || {};
  const k = i.metrics || {};
  const warn = [];
  if (i.rul_estimated) warn.push('<span class="flag" title="Trained partly on estimated belt lives">EST</span>');
  if (i.thresholds_unconfirmed?.length) warn.push('<span class="flag" title="Unconfirmed failure limits">LIM</span>');
  let status;
  if (m.status === 'training') status = '<span class="tier tier-unknown"><span class="btn-spinner">⟳</span> Training</span>';
  else if (m.status === 'failed') status = `<span class="tier tier-red" title="${esc(m.error)}">Failed</span>`;
  else if (m.active_for.length) status = `<span class="tier tier-green" title="Active for: ${esc(m.active_for.map(typeName).join(', '))}">Active · ${m.active_for.length === 1 ? esc(typeName(m.active_for[0])) : `${m.active_for.length} types`}</span>`;
  else status = '<span class="tier tier-unknown">Inactive</span>';
  const ready = m.status === 'ready';
  return `<tr>
    <td>${ready ? `<input type="checkbox" data-action="toggle-compare" data-id="${m.id}" ${state.compareSel.has(m.id) ? 'checked' : ''}/>` : ''}</td>
    <td class="strong">v${m.id} ${warn.join('')}</td>
    <td>${date(m.created_at)}<div class="hint">${esc(m.created_by_name || '')}</div></td>
    <td class="wrap small">${ready ? esc(i.belt_type_ids.map(typeName).join(', ')) : esc(m.error || '')}</td>
    <td>${ready ? `${i.n_belts} <span class="hint">(${i.n_test_belts} test)</span>` : ''}</td>
    <td>${ready ? `± ${hours(k.mae_hours)}` : ''}</td>
    <td>${ready ? `± ${hours(k.near_failure_mae_hours)}` : ''}</td>
    <td>${ready ? biasText(k.bias_hours) : ''}</td>
    <td>${ready ? realAccuracyCell(m.id) : ''}</td>
    <td>${status}</td>
    <td class="row-actions">
      ${ready ? `<button class="btn btn-primary btn-sm" data-action="activate-model" data-id="${m.id}">Activate</button>
      <button class="btn btn-ghost btn-sm" data-action="model-detail" data-id="${m.id}">Details</button>
      <button class="btn btn-ghost btn-sm" data-action="retrain-model" data-id="${m.id}">Retrain + data</button>` : ''}
      ${!m.active_for.length && m.status !== 'training' ? `<button class="btn-icon-only" title="Delete version" data-action="delete-model" data-id="${m.id}">✕</button>` : ''}
    </td></tr>`;
}

function realAccuracyCell(modelId) {
  const a = state.accuracy?.models.find(x => x.model_id === modelId);
  if (!a) return '<span class="muted" title="No belt that this version predicted has failed yet">—</span>';
  return `± ${hours(a.mae_hours)} <span class="hint">· ${a.failed_belts} belt${a.failed_belts === 1 ? '' : 's'}${a.low_sample ? ' ⚠' : ''}</span>`;
}

function accuracyCardHtml(acc) {
  if (!acc) return '';
  const min = acc.min_failed_for_confidence;
  if (!acc.models.length) {
    return `<div class="card"><div class="card-title">🎯 Real-world accuracy</div>
      <p class="card-desc">Nothing to judge yet. When a belt is recorded as <strong>failed</strong>, the model's earlier predictions for it are compared with what actually happened, and the results appear here.
      Belts that were replaced early are shown separately — they only prove a belt lasted <em>at least</em> that long.</p>
      ${acc.total_replaced_belts ? `<div class="hint">${acc.total_replaced_belts} replaced belt(s) recorded so far.</div>` : ''}</div>`;
  }
  const bucketRows = acc.models.map(a => a.buckets.map((b, k) => `<tr>
      ${k === 0 ? `<td class="strong" rowspan="${a.buckets.length}">v${a.model_id}</td>` : ''}
      <td>${esc(b.label)}</td><td>${b.n}</td><td>± ${hours(b.mae_hours)}</td><td>${biasText(b.bias_hours)}</td>
      <td>${b.coverage === null ? '—' : `${Math.round(b.coverage * 100)}%`}</td></tr>`).join('')).join('');
  const failedRows = acc.failed_belts.slice(0, 25).map(b => `<tr class="clickable" data-action="open-belt" data-id="${b.belt_id}">
      <td class="strong">${esc(b.tag)}</td><td>${esc(b.type)}</td><td>${hours(b.end_hour)}</td><td>${hours(b.last_inspection_hour)}</td>
      <td>${hours(b.predicted_rul)}${b.low != null ? ` <span class="hint">(${Math.round(b.low).toLocaleString()}–${Math.round(b.high).toLocaleString()})</span>` : ''}</td>
      <td>${hours(b.actual_rul)}</td><td>${biasText(b.error_hours)}</td>
      <td>${b.in_interval === null ? '—' : b.in_interval ? '<span class="hit-yes">✓</span>' : '<span class="hit-no">✗</span>'}</td><td class="muted">v${b.model_id}</td></tr>`).join('');
  const pessimistic = acc.replaced_belts.filter(b => b.too_pessimistic).length;
  return `<div class="card">
    <div class="card-title">🎯 Real-world accuracy</div>
    <p class="card-desc">Each model's <strong>stored prediction at every inspection</strong> of a belt that later failed, compared with the true remaining life (failure hour − inspection hour).
      Interval hit rate should be close to the 80% the model promises. Bias above 0 means optimistic — the risky direction.</p>
    ${acc.models.some(a => a.low_sample) ? `<div class="notice notice-warn">⚠ Fewer than ${min} failed belts for some versions (marked ⚠). Treat those figures as anecdotes, not accuracy.</div>` : ''}
    <div class="table-wrap"><table class="data-table">
      <thead><tr><th>Version</th><th>Failed belts</th><th>Predictions judged</th><th>Avg error</th><th>Bias</th><th>Interval hit rate</th></tr></thead>
      <tbody>${acc.models.map(a => `<tr><td class="strong">v${a.model_id}</td><td>${a.failed_belts}${a.low_sample ? ' ⚠' : ''}</td><td>${a.points}</td>
        <td>± ${hours(a.mae_hours)}</td><td>${biasText(a.bias_hours)}</td>
        <td>${a.interval_coverage === null ? '—' : `${Math.round(a.interval_coverage * 100)}%`}</td></tr>`).join('')}</tbody></table></div>
    <div class="sub-title">By how much life was really left</div>
    <div class="table-wrap"><table class="data-table">
      <thead><tr><th>Version</th><th>Actual time left</th><th>Predictions</th><th>Avg error</th><th>Bias</th><th>Interval hit rate</th></tr></thead>
      <tbody>${bucketRows}</tbody></table></div>
    <div class="sub-title">Failed belts: the model's last prediction vs. what happened (${acc.total_failed_belts})</div>
    <div class="table-wrap"><table class="data-table">
      <thead><tr><th>Belt</th><th>Type</th><th>Failed at</th><th>Last inspection</th><th>Predicted left</th><th>Actually left</th><th>Error</th><th>In range</th><th>Model</th></tr></thead>
      <tbody>${failedRows}</tbody></table></div>
    ${acc.total_replaced_belts ? `<p class="hint">${acc.total_replaced_belts} belt(s) were replaced before failing and are <strong>not</strong> counted above${pessimistic ? `; for ${pessimistic} of them the model's range ended before the belt's proven survival time (the model was too cautious)` : ''}.</p>` : ''}
  </div>`;
}

function biasText(b) {
  if (b === null || b === undefined) return '—';
  const cls = b > 0 ? 'bias-opt' : 'bias-pess';
  return `<span class="${cls}" title="${b > 0 ? 'Predicts MORE life than reality (optimistic)' : 'Predicts LESS life than reality (conservative)'}">${b > 0 ? '+' : ''}${Math.round(b).toLocaleString()} h</span>`;
}

async function activateModel(id) {
  const m = state.models.find(x => x.id === id);
  const names = m.info.belt_type_ids.map(typeName).join(', ');
  const flags = [];
  if (m.info.rul_estimated) flags.push('• It was trained partly on ESTIMATED belt lives.');
  if (m.info.thresholds_unconfirmed?.length) flags.push(`• Failure limits are unconfirmed for: ${m.info.thresholds_unconfirmed.join(', ')}.`);
  if (!confirm(`Activate v${id} for: ${names}?\n\nAll RUL predictions for these belt types will use it immediately.${flags.length ? '\n\n' + flags.join('\n') : ''}`)) return;
  await api(`/api/models/${id}/activate`, { method: 'POST', body: {} });
  toast(`v${id} is now active for ${names}.`, 'success');
  renderModels();
}

async function modelDetail(id) {
  const m = await api(`/api/models/${id}`);
  const i = m.info;
  const imp = Object.entries(i.feature_importance).slice(0, 10);
  const maxImp = imp.length ? imp[0][1] : 1;
  $('#model-detail-card').innerHTML = `
    <div class="card">
      <div class="card-title">Version v${m.id} — details <button class="btn-icon-only" data-action="close-detail" title="Close">✕</button></div>
      <div class="detail-grid">
        <div>
          <div class="sub-title">Training data</div>
          <ul class="plain-list">${i.datasets.map(d => `<li><span class="strong">${esc(d.filename)}</span> — ${d.rows.toLocaleString()} rows
            <div class="hint">${esc(d.rul_source.message)}</div></li>`).join('')}</ul>
          <div class="sub-title">Failure limits used</div>
          <ul class="plain-list">${Object.entries(i.thresholds_used).map(([tid, thr]) => `<li>${esc(typeName(Number(tid)))}: ${
            Object.entries(thr).map(([f, v]) => `${esc(fieldLabel(f))} ${fmt(v)}${esc(state.fields[f]?.unit || '')}`).join(', ')}</li>`).join('')}</ul>
          <div class="sub-title">Held-out accuracy</div>
          <p class="small">Average error ± ${hours(i.metrics.mae_hours)} · near-failure ± ${hours(i.metrics.near_failure_mae_hours)} · RMSE ${hours(i.metrics.rmse_hours)} · R² ${i.metrics.r2_score ?? '—'} · bias ${biasText(i.metrics.bias_hours)}</p>
          ${i.method === 'aft' ? `<p class="small">Survival model (XGBoost AFT): concordance index ${i.metrics.c_index ?? '—'} (belt ranking at first inspection ${i.metrics.c_index_belt ?? '—'}) ·
            80% interval covers the true RUL ${pct(i.metrics.interval_coverage)} of the time (target 80%) · mean interval width ${hours(i.metrics.interval_mean_width_hours)}.</p>
            <p class="small">${i.n_events} belts with a known end of life, ${i.n_censored_belts} still running / replaced early (censored).
            Error figures use belts with a known end of life only; ${i.metrics.n_censored_rows} censored test rows count in the concordance index only.</p>`
          : `<p class="small">Point-regression model (no interval, no concordance-calibrated range): belts still running were ignored in training.</p>`}
          <div class="sub-title">Test method</div>
          <p class="small">${i.test_split === 'by_belt' ? `${i.n_test_belts} of ${i.n_belts} belts held out entirely (never seen in training).` : 'No belt IDs in the data — rows were split randomly, so metrics may be optimistic.'}
            Near-failure error = belts with under ${hours(i.metrics.near_failure_cutoff_hours)} left.</p>
          ${m.notes ? `<div class="sub-title">Notes</div><p class="small">${esc(m.notes)}</p>` : ''}
        </div>
        <div>
          <div class="sub-title">Most influential inputs</div>
          ${imp.map(([f, v]) => `<div class="imp-row"><span class="imp-label">${esc(fieldLabel(f))}</span>
            <span class="imp-bar"><span style="width:${(v / maxImp * 100).toFixed(0)}%"></span></span></div>`).join('')}
          <div class="sub-title">Review note</div>
          <textarea id="review-note" rows="3" maxlength="1000" placeholder="Your assessment of this version (e.g. checked against Q3 failures)">${esc(m.review_note)}</textarea>
          <div class="action-row"><button class="btn btn-outline btn-sm" data-action="save-review" data-id="${m.id}">Save note</button></div>
        </div>
      </div>
    </div>`;
  $('#model-detail-card').scrollIntoView({ behavior: 'smooth' });
}

async function compareModels() {
  const [a, b] = [...state.compareSel].sort((x, y) => x - y);
  showOverlay('Comparing versions…', 'Scoring both on the same held-out belts.');
  const res = await guarded(() => api(`/api/models/compare?a=${a}&b=${b}`));
  hideOverlay();
  if (!res) return;
  const rows = [
    ['Average error', 'mae_hours', true, hours], ['Near-failure error', 'near_failure_mae_hours', true, hours],
    ['RMSE', 'rmse_hours', true, hours], ['R²', 'r2_score', false, v => v ?? '—'], ['Bias', 'bias_hours', null, biasText],
    ['Concordance index', 'c_index', false, v => v ?? '—'], ['80% interval coverage', 'interval_coverage', null, pct],
  ];
  const metricsOf = id => res.comparable ? res.models[id].shared_metrics : res.models[id].own_metrics;
  const ma = metricsOf(a), mb = metricsOf(b);
  const cell = (val, other, lowerBetter, f) => {
    let cls = '';
    if (lowerBetter !== null && val !== null && other !== null && val !== other) {
      const better = lowerBetter ? val < other : val > other;
      cls = better ? 'better' : '';
    }
    return `<td class="${cls}">${f(val)}</td>`;
  };
  $('#compare-card').innerHTML = `
    <div class="card">
      <div class="card-title">Comparison: v${a} vs v${b} <button class="btn-icon-only" data-action="close-compare" title="Close">✕</button></div>
      ${res.comparable ? `<div class="notice notice-info">${esc(res.note)} (${res.n_test_belts} shared test belts from ${esc(res.datasets.join(', '))})</div>`
                       : `<div class="notice notice-warn">${esc(res.reason)}</div>`}
      <table class="data-table compare-table">
        <thead><tr><th>${res.comparable ? 'Same test belts' : 'Own test belts'}</th><th>v${a}</th><th>v${b}</th></tr></thead>
        <tbody>${rows.map(([label, key, lower, f]) => `<tr><td>${label}</td>${cell(ma[key], mb[key], lower, f)}${cell(mb[key], ma[key], lower, f)}</tr>`).join('')}
          <tr><td>Warnings</td>${[a, b].map(id => `<td class="small">${res.models[id].flags.map(x => `<div>⚠ ${esc(x)}</div>`).join('') || '—'}</td>`).join('')}</tr>
        </tbody>
      </table>
      <p class="hint">Green = better on that metric. Bias &gt; 0 means the model predicts more life than the belt really had — the risky direction.</p>
    </div>`;
  $('#compare-card').scrollIntoView({ behavior: 'smooth' });
}

function retrainModel(id) {
  const m = state.models.find(x => x.id === id);
  state.trainPreselect = m.info.datasets.map(d => ({ filename: d.filename, belt_type_id: d.belt_type_id }));
  toast(`Pre-selected v${id}'s datasets — tick the new dataset and train.`, 'info');
  location.hash = 'data';
}

// ═══════════════════════════════════════════════════════════════════════════
// USERS (engineer)
// ═══════════════════════════════════════════════════════════════════════════
async function renderUsers() {
  const users = await api('/api/users');
  $('#view-users').innerHTML = `
    <div class="view-header"><div><h1 class="panel-title">Users</h1>
      <p class="panel-subtitle">Planners manage the fleet and inspections. Engineers can also manage data, models, belt types and users.</p></div></div>
    <div class="card">
      <div class="card-title">Add user</div>
      <div class="form-row cols-3">
        <div class="form-group"><label>Username</label><input type="text" id="uf-name" maxlength="50" autocomplete="off" /></div>
        <div class="form-group"><label>Role</label><select id="uf-role"><option value="planner">Planner</option><option value="engineer">Engineer</option></select></div>
        <div class="form-group"><label>Initial password (min 10 chars)</label><input type="password" id="uf-pass" autocomplete="new-password" /></div>
      </div>
      <div class="action-row"><button class="btn btn-primary" data-action="add-user">Add user</button></div>
    </div>
    <div class="card"><div class="table-wrap"><table class="data-table">
      <thead><tr><th>Username</th><th>Role</th><th>Status</th><th>Created</th><th></th></tr></thead>
      <tbody>${users.map(u => `<tr><td class="strong">${esc(u.username)}</td><td>${esc(u.role)}</td>
        <td>${u.active ? '<span class="tier tier-green">Active</span>' : '<span class="tier tier-unknown">Disabled</span>'}</td><td>${date(u.created_at)}</td>
        <td>${u.id === state.user.id ? '<span class="muted">you</span>' : `<button class="btn btn-ghost btn-sm" data-action="toggle-user" data-id="${u.id}" data-active="${u.active ? 0 : 1}">${u.active ? 'Disable' : 'Enable'}</button>`}</td></tr>`).join('')}
      </tbody></table></div></div>`;
}

async function addUser() {
  await api('/api/users', { method: 'POST', body: {
    username: $('#uf-name').value.trim(), role: $('#uf-role').value, password: $('#uf-pass').value } });
  toast('User added.', 'success');
  renderUsers();
}

// ═══════════════════════════════════════════════════════════════════════════
// Event delegation
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
// BULK IMPORT (fleet page)
// ═══════════════════════════════════════════════════════════════════════════
function openImport() {
  state.importReport = null;
  const typeOpts = state.types.map(t => `<option value="${t.id}">${esc(t.name)}</option>`).join('');
  $('#import-card').innerHTML = `
    <div class="card import-card">
      <div class="card-title">⬆ Import from CSV <button class="btn-icon-only" data-action="close-import" title="Close">✕</button></div>
      <p class="card-desc">Load many belts or inspections at once. The file is <strong>checked first</strong> and nothing is saved until you confirm. If any row has an error, nothing is imported — fix the file and check it again.</p>
      <div class="form-row cols-3">
        <div class="form-group"><label>What are you importing?</label>
          <select id="import-kind"><option value="inspections">Inspections (readings for registered belts)</option><option value="belts">Belts (register many belts)</option></select></div>
        <div class="form-group"><label>CSV file</label><input type="file" id="import-file" accept=".csv" /></div>
        <div class="form-group" id="import-type-group"><label>Default belt type</label>
          <select id="import-default-type"><option value="">Use the file's type column</option>${typeOpts}</select></div>
      </div>
      <div class="hint" id="import-hint"></div>
      <div class="action-row">
        <a class="btn btn-ghost" id="import-template" href="/api/import/template/inspections" download>⬇ Download template</a>
        <button class="btn btn-outline" data-action="import-check">Check file</button>
      </div>
      <div id="import-result"></div>
    </div>`;
  $('#import-kind').addEventListener('change', syncImportKind);
  $('#import-file').addEventListener('change', () => { state.importReport = null; $('#import-result').innerHTML = ''; });
  syncImportKind();
  $('#import-card').scrollIntoView({ behavior: 'smooth' });
}

function syncImportKind() {
  const kind = $('#import-kind').value;
  $('#import-template').href = `/api/import/template/${kind}`;
  $('#import-type-group').classList.toggle('hidden', kind !== 'belts');
  $('#import-hint').innerHTML = kind === 'inspections'
    ? 'One row per belt per inspection: <code>belt_tag</code>, <code>hour</code> and any measurements. Belts must already be registered and in service. Use the units in the template (°C, mm, %). Leave a cell empty if it was not measured — never 0.'
    : 'One row per belt: <code>belt_tag</code>, <code>belt_type</code> (name, section code or alias), optional location, installed date (2025-01-31) and drive details. Tags that already exist are left unchanged.';
  state.importReport = null;
  $('#import-result').innerHTML = '';
}

async function runImport(dryRun) {
  const file = $('#import-file').files[0];
  if (!file) { toast('Choose a CSV file first.', 'error'); return; }
  const kind = $('#import-kind').value;
  const form = new FormData();
  form.append('file', file);
  form.append('dry_run', dryRun ? 'true' : 'false');
  const def = $('#import-default-type').value;
  if (kind === 'belts' && def) form.append('default_belt_type_id', def);
  showOverlay(dryRun ? 'Checking file…' : 'Importing…', file.name);
  let report;
  try { report = await api(`/api/import/${kind}`, { method: 'POST', form }); }
  finally { hideOverlay(); }
  state.importReport = { kind, file: file.name, report };
  const r = report;
  if (r.committed) {
    $('#import-result').innerHTML = `<div class="notice notice-ok">✓ Imported ${r.to_create} new${r.to_update ? ` and updated ${r.to_update}` : ''} ${kind === 'belts' ? 'belt(s)' : `inspection(s) across ${r.belts_affected} belt(s)`}.</div>`;
    toast('Import complete.', 'success');
    state.fleet = await api('/api/belts');
    drawFleetSummary();
    drawFleetTable();
    return;
  }
  renderImportReport(r);
}

function renderImportReport(r) {
  const cols = r.columns.map(c => `<tr class="${c.field ? '' : 'col-ignored'}"><td class="strong">${esc(c.source)}</td>
    <td>${c.field ? `${esc(state.fields[c.field]?.label || c.field.replace(/^__|__$/g, '').replace(/_/g, ' '))}${c.unit ? ` <span class="unit">(${esc(c.unit)})</span>` : ''}` : '<span class="tier tier-amber">Not recognised — will not be used</span>'}</td></tr>`).join('');
  const unrecognised = r.ignored_columns.length;
  const errors = r.errors.map(e => `<tr><td>${e.row}</td><td>${esc(e.message)}</td></tr>`).join('');
  const preview = r.preview.length ? `<div class="sub-title">First rows that will be imported</div>
    <div class="table-wrap"><table class="data-table"><tbody>${r.preview.map(p => `<tr><td>row ${p.row}</td><td class="strong">${esc(p.tag)}</td>
      <td class="wrap small">${p.hour !== undefined ? `${hours(p.hour)} · ${Object.entries(p.readings).map(([k, v]) => `${esc(state.fields[k]?.label || k)} ${fmt(v)}`).join(', ')}` : `${esc(p.type)}${p.location ? ` · ${esc(p.location)}` : ''}`}</td></tr>`).join('')}</tbody></table></div>` : '';
  const ready = !r.error_count && (r.to_create + r.to_update) > 0;
  $('#import-result').innerHTML = `
    <div class="import-summary">
      <div class="summary-tile"><div class="summary-num">${r.rows.toLocaleString()}</div><div class="summary-label">Rows in file</div></div>
      <div class="summary-tile tile-green"><div class="summary-num">${r.to_create}</div><div class="summary-label">To add</div></div>
      <div class="summary-tile tile-amber"><div class="summary-num">${r.to_update}</div><div class="summary-label">To replace</div></div>
      <div class="summary-tile ${r.error_count ? 'tile-red' : ''}"><div class="summary-num">${r.error_count}</div><div class="summary-label">Errors</div></div>
    </div>
    ${r.warnings.map(w => `<div class="notice notice-warn">${esc(w)}</div>`).join('')}
    ${unrecognised ? `<div class="notice notice-warn"><strong>${unrecognised} column(s) were not recognised and will be ignored:</strong> ${r.ignored_columns.map(esc).join(', ')}. If one of them is a measurement you want, rename it to match the template.</div>` : ''}
    ${r.error_count ? `<div class="sub-title">Errors${r.error_count > r.errors.length ? ` (first ${r.errors.length} of ${r.error_count})` : ''}</div>
      <div class="table-wrap import-errors"><table class="data-table"><thead><tr><th>Row</th><th>Problem</th></tr></thead><tbody>${errors}</tbody></table></div>` : ''}
    <div class="sub-title">How your columns were read</div>
    <div class="table-wrap"><table class="data-table"><thead><tr><th>Your column</th><th>Used as</th></tr></thead><tbody>${cols}</tbody></table></div>
    ${preview}
    <div class="action-row">
      <button class="btn btn-primary btn-large" data-action="import-commit" ${ready ? '' : 'disabled'}>Import ${(r.to_create + r.to_update).toLocaleString()} ${state.importReport.kind === 'belts' ? 'belt(s)' : 'inspection(s)'}</button>
    </div>`;
}

const ACTIONS = {
  'open-import': openImport,
  'close-import': () => { $('#import-card').innerHTML = ''; state.importReport = null; },
  'import-check': () => runImport(true),
  'import-commit': async () => {
    const r = state.importReport?.report;
    if (!r || r.error_count) return;
    if (!confirm(`Import ${r.to_create + r.to_update} ${state.importReport.kind === 'belts' ? 'belt(s)' : 'inspection(s)'} from ${state.importReport.file}?${r.to_update ? `\n\n${r.to_update} existing inspection(s) will be replaced.` : ''}`)) return;
    await runImport(false);
  },
  'dataset-from-fleet': async () => {
    const res = await api('/api/datasets/from-fleet', { method: 'POST' });
    toast(`Created ${res.filename}: ${res.belts} belts (${res.failed_belts} failed, ${res.replaced_belts} replaced, ${res.in_service_belts} in service).`, 'success');
    await renderData();
  },
  'logout': async () => { await api('/api/auth/logout', { method: 'POST' }).catch(() => {}); location.hash = ''; showLogin(); },
  'open-belt': el => { location.hash = `belt/${el.dataset.id}`; },
  'new-belt': () => { state.editBelt = null; $('#belt-form-card').innerHTML = beltFormHtml(null); $('#bf-tag').focus(); },
  'edit-belt': () => { state.editBelt = state.belt.belt; $('#belt-form-card').innerHTML = beltFormHtml(state.belt.belt); $('#belt-form-card').scrollIntoView({ behavior: 'smooth' }); },
  'cancel-belt-form': () => { state.editBelt = null; $('#belt-form-card').innerHTML = ''; },
  'save-belt': saveBelt,
  'add-inspection': addInspection,
  'delete-inspection': async el => {
    if (!confirm('Delete this inspection?')) return;
    await api(`/api/belts/${state.belt.belt.id}/inspections/${el.dataset.id}`, { method: 'DELETE' });
    renderBelt(state.belt.belt.id);
  },
  'end-of-life': () => endOfLife(false),
  'undo-end-of-life': () => endOfLife(true),
  'new-type': () => { state.editType = null; $('#type-form-card').innerHTML = typeFormHtml(null); },
  'edit-type': el => { state.editType = state.types.find(t => t.id === Number(el.dataset.id)); $('#type-form-card').innerHTML = typeFormHtml(state.editType); $('#type-form-card').scrollIntoView({ behavior: 'smooth' }); },
  'cancel-type-form': () => { state.editType = null; $('#type-form-card').innerHTML = ''; },
  'save-type': saveType,
  'open-mapping': el => openMapping(el.dataset.name),
  'close-mapping': () => { $('#mapping-card').innerHTML = ''; },
  'approve-mapping': approveMapping,
  'delete-dataset': async el => {
    if (!confirm(`Delete ${el.dataset.name} and its approved mapping? Model versions trained on it keep working, but can no longer be retrained or compared on this file.`)) return;
    await api(`/api/datasets/${encodeURIComponent(el.dataset.name)}`, { method: 'DELETE' });
    toast('Dataset deleted.', 'success');
    renderData();
  },
  'train': startTraining,
  'generate': generateData,
  'toggle-compare': el => {
    const id = Number(el.dataset.id);
    if (el.checked) {
      if (state.compareSel.size >= 2) { el.checked = false; toast('Select exactly two versions.', 'info'); return; }
      state.compareSel.add(id);
    } else state.compareSel.delete(id);
    const btn = $('[data-action="compare"]');
    btn.disabled = state.compareSel.size !== 2;
    btn.textContent = `Compare selected (${state.compareSel.size}/2)`;
  },
  'compare': compareModels,
  'close-compare': () => { $('#compare-card').innerHTML = ''; },
  'activate-model': el => activateModel(Number(el.dataset.id)),
  'model-detail': el => modelDetail(Number(el.dataset.id)),
  'close-detail': () => { $('#model-detail-card').innerHTML = ''; },
  'save-review': async el => {
    await api(`/api/models/${el.dataset.id}/review`, { method: 'PUT', body: { review_note: $('#review-note').value } });
    toast('Note saved.', 'success');
  },
  'retrain-model': el => retrainModel(Number(el.dataset.id)),
  'delete-model': async el => {
    if (!confirm(`Delete version v${el.dataset.id}? This cannot be undone.`)) return;
    await api(`/api/models/${el.dataset.id}`, { method: 'DELETE' });
    toast('Version deleted.', 'success');
    renderModels();
  },
  'add-user': addUser,
  'toggle-user': async el => {
    await api(`/api/users/${el.dataset.id}/active`, { method: 'PUT', body: { active: el.dataset.active === '1' } });
    renderUsers();
  },
};

function bindEvents() {
  $('#login-form').addEventListener('submit', login);
  document.addEventListener('click', e => {
    const el = e.target.closest('[data-action]');
    if (!el || el.disabled) return;
    const fn = ACTIONS[el.dataset.action];
    if (!fn) return;
    if (el.tagName !== 'INPUT') e.preventDefault();
    guarded(() => fn(el, e));
  });
}
