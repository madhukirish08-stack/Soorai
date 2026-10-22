/**
 * Belt RUL — guided tours and glossary tooltips
 * =============================================
 * Tour.tip('rul')        -> HTML for an (i) icon whose hover/focus tooltip explains the term (GLOSSARY).
 * Tour.auto('belt')      -> start the page tour once per user (tracked server-side, /api/auth/tours).
 * Tour.start('belt')     -> replay a tour on demand (Help menu).
 * Steps point at CSS selectors; steps whose element is missing (e.g. no inspections yet) are skipped.
 *
 * CSP: no inline handlers; styles are applied through the CSSOM, which the CSP allows.
 */
'use strict';

const Tour = (() => {
  const GLOSSARY = {
    rul: 'Remaining useful life: how many more operating hours the belt is expected to run before it reaches a failure limit. It is the model\'s median estimate.',
    interval: 'The 80% interval is the range the true remaining life should fall in 80 times out of 100. A wide range means the model is less sure.',
    health: 'Health score from 100% (new) to 0% (at a failure limit), based on the most worn measurement. It does not need a model.',
    tier: 'Healthy, Watch or Critical, based on health and remaining life. Critical belts are listed first in the fleet.',
    limit: 'The measurement value at which this belt type counts as failed. Set per belt type by an engineer. Gauges show how close each reading is to it.',
    gauge: 'Each ring is one measurement as a share of its failure limit. 100% means the limit has been reached.',
    elongation: 'How much longer the belt has stretched compared with new. It only increases.',
    wear: 'How much of the belt material has worn away. It only increases.',
    crack: 'Depth of the deepest crack. It only increases, so you will be warned if you enter less than a value already recorded.',
    hours: 'Total operating hours since the belt was installed, not calendar time.',
    hit_rate: 'Interval hit rate: how often real failures landed inside the predicted range on belts the model never trained on. Close to 80% is well calibrated.',
    mae: 'Mean absolute error: on average, how many hours the prediction was off by on belts the model never trained on.',
    active_model: 'The model version used for predictions of a belt type. Activating another version changes RUL for every belt of that type.',
    estimated: 'The model was trained partly on belt lives that were estimated, not observed. Record real failures to replace them.',
    retrain: 'Train a new version using the same settings plus newly recorded data. The old version stays available for comparison.',
    bias: 'Average over- or under-prediction. Above 0 means the model predicts more life than the belt really had (optimistic, the risky direction).',
    near_failure: 'Average error counting only the last part of belt life, which is the part that matters for replacement decisions.',
    censored: 'A belt that had not failed yet when the data ended. Its true life is at least as long as recorded.',
    outcome: 'Failed means the belt broke. Replaced means it was taken out of service early, so it is not counted as a failure when judging accuracy.',
  };

  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const tip = key => GLOSSARY[key]
    ? ` <span class="tip" tabindex="0" role="note" aria-label="${esc(GLOSSARY[key])}" data-tip="${esc(GLOSSARY[key])}">i</span>` : '';

  // ── Tooltip bubble: one floating element (table wrappers clip absolutely-positioned children) ──
  let bubble = null;
  function showTip(el) {
    if (!bubble) { bubble = document.createElement('div'); bubble.className = 'tip-bubble'; bubble.setAttribute('role', 'tooltip'); document.body.appendChild(bubble); }
    bubble.textContent = el.dataset.tip;
    bubble.style.display = 'block';
    place(bubble, el.getBoundingClientRect(), 'top');
  }
  function hideTip() { if (bubble) bubble.style.display = 'none'; }

  // Position `box` next to `rect`, preferring `side`, flipping if it would leave the viewport.
  function place(box, rect, side) {
    const m = 10, w = box.offsetWidth, h = box.offsetHeight;
    let top = side === 'top' ? rect.top - h - m : rect.bottom + m;
    if (top < 8) top = rect.bottom + m;
    if (top + h > innerHeight - 8) top = Math.max(8, rect.top - h - m);
    if (top < 8 || top + h > innerHeight - 8) top = Math.max(8, innerHeight - h - 8);   // element is taller than the viewport
    const left = Math.min(Math.max(8, rect.left + rect.width / 2 - w / 2), innerWidth - w - 8);
    box.style.top = `${top}px`;
    box.style.left = `${left}px`;
  }

  document.addEventListener('mouseover', e => { const t = e.target.closest?.('[data-tip]'); if (t) showTip(t); });
  document.addEventListener('mouseout', e => { if (e.target.closest?.('[data-tip]')) hideTip(); });
  document.addEventListener('focusin', e => { const t = e.target.closest?.('[data-tip]'); if (t) showTip(t); });
  document.addEventListener('focusout', hideTip);
  addEventListener('scroll', hideTip, true);

  // ── Tours ────────────────────────────────────────────────────────────────
  // role: 'engineer' limits a step to engineers. `before` can prepare the page (e.g. open a form).
  const TOURS = {
    welcome: { title: 'Welcome to Belt RUL', steps: [
      { title: 'Welcome', text: 'Belt RUL tracks the condition of your belts and predicts how many hours each one has left. This 1-minute tour shows where things are. You can replay it any time from <strong>? Help</strong>.' },
      { sel: '#main-nav', title: 'Navigation', text: 'Fleet is the home page. Engineers also see Belt Types, Data &amp; Training, Models and Users.' },
      { sel: '#fleet-summary', title: 'Fleet status', text: 'How many belts are Critical, on Watch or Healthy. Check this first each shift.' },
      { sel: '#fleet-search', title: 'Find a belt', text: 'Search by tag or location, or filter by belt type. Most urgent belts are always listed first.' },
      { sel: '#fleet-table', title: 'Open a belt', text: 'Click any row to open the belt, record an inspection and see its predicted remaining life.' },
      { sel: '[data-action="new-belt"]', title: 'Register a belt', text: 'New belts start here. Only a tag and a belt type are required.' },
      { sel: '[data-action="open-import"]', title: 'Bulk import', text: 'Have many belts or old inspections in a spreadsheet? Import them from CSV. A check step shows problems before anything is saved.' },
      { sel: '[data-action="help"]', title: 'Help is always here', text: 'Replay this tour or the tour of any page. Hover the small <strong>i</strong> icons for plain-language explanations of terms.' },
    ] },
    belt: { title: 'Belt page tour', steps: [
      { sel: '#view-belt .rul-card, #view-belt .results-empty', title: 'Remaining life', text: 'The headline number: expected hours left, with an 80% range and a health score. It updates every time you save an inspection.' },
      { sel: '#insp-hour', title: 'Record an inspection', text: 'Enter the operating hours at the time of inspection. Only hours are required, and the form suggests 250 h after the last inspection.' },
      { sel: '#view-belt .sub-title + .form-row', title: 'Degradation measurements', text: 'Leave a field empty if it was not measured, never enter 0. Under each field you see the value already on file, and you are warned if you enter less, because wear and cracks only grow.' },
      { sel: '[data-action="add-inspection"]', title: 'Save', text: 'Saving stores the reading and immediately shows the new prediction.' },
      { sel: '#view-belt .gauge-row', title: 'Gauges', text: 'Each ring shows a measurement as a share of this belt type\'s failure limit.' },
      { sel: '#belt-chart', title: 'History', text: 'RUL (blue, with its range) and health (green) across all inspections. A sudden jump usually means a wrong reading, so check the log below.' },
      { sel: '#eol-status', title: 'When the belt is done', text: 'Record a failure or a planned replacement here. This is the most valuable data for making future predictions better.' },
    ] },
    types: { role: 'engineer', title: 'Belt Types tour', steps: [
      { sel: '#view-types .data-table', title: 'Failure limits', text: 'Each belt type has its own limits for elongation, wear and crack depth. Health and RUL are measured against these, so they matter most.' },
      { sel: '[data-action="new-type"]', title: 'Add a type', text: 'Aliases let dataset values (section codes, product names) match this type automatically.' },
      { sel: '#view-types .data-table', title: 'Confirm limits', text: 'Types with unconfirmed placeholder limits show a warning on every belt page until an engineer confirms them.' },
    ] },
    data: { role: 'engineer', title: 'Data &amp; Training tour', steps: [
      { sel: '#upload-drop', title: '1. Upload data', text: 'Drop a CSV of inspections and failures. The app proposes how your columns map to its fields.' },
      { sel: '[data-action="dataset-from-fleet"]', title: 'Or use your own records', text: 'Turns the inspections and failures recorded in this app into a training dataset.' },
      { sel: '[data-action="train"]', title: '2. Train', text: 'Pick datasets and train a new model version. Nothing changes for operators until you activate it on the Models page.' },
    ] },
    models: { role: 'engineer', title: 'Models tour', steps: [
      { sel: '#view-models .data-table', title: 'Model versions', text: 'Every training run is kept as a version, with its held-out accuracy. The active version is used for predictions.' },
      { sel: '[data-action="activate-model"]', title: 'Activate', text: 'Switches predictions for that belt type to this version immediately.' },
      { sel: '[data-action="toggle-compare"]', title: 'Compare', text: 'Tick two versions and compare them side by side before activating.' },
      { sel: '#view-models .card-title', title: 'Real-world accuracy', text: 'Once belts fail, this shows how close the predictions made at the time really were.' },
    ] },
    users: { role: 'engineer', title: 'Users tour', steps: [
      { sel: '#uf-role', title: 'Roles', text: 'Planners can register belts and record inspections. Engineers can also manage types, data, models and users.' },
    ] },
  };
  const TOUR_FOR_VIEW = { fleet: 'welcome', belt: 'belt', types: 'types', data: 'data', models: 'models', users: 'users' };

  let session = null;      // { key, steps, i, ring, card }
  const seen = new Set();

  function stepsFor(key) {
    const tour = TOURS[key];
    if (!tour) return [];
    // `state` is app.js's top-level const: visible as a global binding, but not as window.state.
    const engineer = typeof state !== 'undefined' && state.user?.role === 'engineer';
    return tour.steps.filter(s => (!s.role || engineer) && (!s.sel || document.querySelector(s.sel)));
  }

  function end(completed) {
    if (!session) return;
    const { key } = session;
    session.block.remove(); session.ring.remove(); session.card.remove();
    removeEventListener('resize', reposition);
    removeEventListener('keydown', onKey, true);
    session = null;
    // Mark seen even when skipped, otherwise the tour would nag on every login.
    if (!seen.has(key)) { seen.add(key); fetch(`/api/auth/tours/${key}`, { method: 'PUT', credentials: 'same-origin' }).catch(() => {}); }
    if (completed && key === 'welcome') window.toast?.('Tour finished. Open a belt to see the belt page tour.', 'info');
  }

  function onKey(e) {
    if (!session) return;
    if (e.key === 'Escape') end(false);
    else if (e.key === 'ArrowRight' || e.key === 'Enter') { e.preventDefault(); go(1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
  }

  function go(delta) {
    const next = session.i + delta;
    if (next < 0) return;
    if (next >= session.steps.length) { end(true); return; }
    session.i = next;
    render();
  }

  function reposition() { if (session) render(true); }

  function render(keepScroll) {
    const { steps, i, ring, card } = session;
    const s = steps[i];
    const el = s.sel ? document.querySelector(s.sel) : null;
    if (el && !keepScroll) el.scrollIntoView({ block: 'center', behavior: 'instant' });
    const last = i === steps.length - 1;
    card.innerHTML = `
      <div class="tour-count">${i + 1} / ${steps.length}</div>
      <div class="tour-title">${s.title}</div>
      <div class="tour-text">${s.text}</div>
      <div class="tour-actions">
        <button class="btn btn-ghost btn-sm" data-tour-act="skip">${last ? 'Close' : 'Skip tour'}</button>
        <span class="tour-spacer"></span>
        ${i > 0 ? '<button class="btn btn-ghost btn-sm" data-tour-act="back">Back</button>' : ''}
        <button class="btn btn-primary btn-sm" data-tour-act="next">${last ? 'Done' : 'Next'}</button>
      </div>`;
    if (el) {
      const r = el.getBoundingClientRect(), pad = 6;
      Object.assign(ring.style, { display: 'block', top: `${r.top - pad}px`, left: `${r.left - pad}px`, width: `${r.width + pad * 2}px`, height: `${r.height + pad * 2}px` });
      place(card, r, 'bottom');
    } else {
      // Intro step: a zero-size ring in the middle whose shadow dims the whole page.
      Object.assign(ring.style, { display: 'block', top: `${innerHeight / 2}px`, left: `${innerWidth / 2}px`, width: '0px', height: '0px' });
      card.style.top = `${Math.max(16, (innerHeight - card.offsetHeight) / 3)}px`;
      card.style.left = `${Math.max(8, (innerWidth - card.offsetWidth) / 2)}px`;
    }
    card.querySelector('[data-tour-act="next"]').focus({ preventScroll: true });
  }

  function start(key) {
    if (session) end(false);
    const steps = stepsFor(key);
    if (!steps.length) return false;
    hideTip();
    // Transparent full-screen layer so clicks cannot reach the page (and navigate away) mid-tour.
    const block = document.createElement('div'); block.className = 'tour-block';
    const ring = document.createElement('div'); ring.className = 'tour-ring';
    const card = document.createElement('div'); card.className = 'tour-card'; card.setAttribute('role', 'dialog'); card.setAttribute('aria-live', 'polite');
    document.body.append(block, ring, card);
    card.addEventListener('click', e => {
      const act = e.target.closest('[data-tour-act]')?.dataset.tourAct;
      if (act === 'next') go(1); else if (act === 'back') go(-1); else if (act === 'skip') end(false);
    });
    session = { key, steps, i: 0, block, ring, card };
    addEventListener('resize', reposition);
    addEventListener('keydown', onKey, true);
    render();
    return true;
  }

  // Start the tour for a view once per user (`seen` is filled from /api/auth/me via load()).
  function auto(view) {
    const key = TOUR_FOR_VIEW[view];
    if (!key || session || seen.has(key)) return;
    // Let freshly rendered cards settle first. A timer, not requestAnimationFrame: rAF never fires in a background tab.
    setTimeout(() => { if (!session) start(key); }, 80);
  }

  function hasTour(view) { return Boolean(TOUR_FOR_VIEW[view]) && stepsFor(TOUR_FOR_VIEW[view]).length > 0; }

  // Called at sign-in: the server-side flags are per account, so replace whatever was cached.
  function load(list) { seen.clear(); (list || []).forEach(k => seen.add(k)); }

  return { tip, start, auto, hasTour, load, stop: () => session && end(false), keyFor: v => TOUR_FOR_VIEW[v] };
})();
