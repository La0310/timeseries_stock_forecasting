/**
 * forecast_card.js
 * Screen 2 — SKU Detail + AI Forecast
 *
 * Flow:
 *   1. Read ?sku=SKU-xxx from URL
 *   2. Call POST /api/forecast immediately to populate the hero header
 *   3. "Generate AI Forecast" button re-calls the same endpoint and renders
 *      all four analysis sections using ONLY fields from the API response.
 *
 * Guardrails (matching backend spec):
 *   - Status badge rendered verbatim from replenishment.status — never derived
 *   - ACF values: null → "n/a" (never 0 or blank)
 *   - Model scores: null → "Insufficient Data"
 *   - No field is computed or transformed client-side
 */

'use strict';

/* ── Constants ─────────────────────────────────────────────────────────── */
const API_BASE = 'http://127.0.0.1:8000';
const FORECAST_ENDPOINT = `${API_BASE}/api/forecast`;

/* ── Read SKU from URL ──────────────────────────────────────────────────── */
const urlParams = new URLSearchParams(window.location.search);
const SKU_ID = urlParams.get('sku') || '';

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const mainContent = document.getElementById('main-content');
let _forecastData = null;
let _sectionsVisible = false;

/* ── Escape helper ──────────────────────────────────────────────────────── */
function esc(v) {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ── Format helpers ─────────────────────────────────────────────────────── */
const fmt = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d));
const fmtInt = (v) => (v == null ? '—' : Number(v).toLocaleString());
const fmtAcf = (v) => (v == null ? 'n/a' : fmt(v, 2));

/* ── Status badge class ─────────────────────────────────────────────────── */
function statusClass(status) {
  if (!status) return 'unknown';
  switch (status.toLowerCase()) {
    case 'critical': return 'critical';
    case 'low': return 'low';
    case 'healthy': return 'healthy';
    default: return 'unknown';
  }
}

function heroBadge(status) {
  const cls = statusClass(status);
  return `<span class="hero-badge ${cls}" aria-label="Status: ${esc(status || '—')}">
            <span class="hero-badge-dot" aria-hidden="true"></span>${esc(status || '—')}
          </span>`;
}

/* ── Demand class pill ──────────────────────────────────────────────────── */
function demandClassPill(dc) {
  const map = {
    smooth: 'demand-smooth',
    intermittent: 'demand-intermittent',
    erratic: 'demand-erratic',
    lumpy: 'demand-lumpy',
    'zero demand': 'demand-zero',
  };
  const cls = map[(dc ?? '').toLowerCase()] ?? 'demand-zero';
  return `<span class="demand-class-pill ${cls}">${esc(dc ?? '—')}</span>`;
}

/* ── Error page ─────────────────────────────────────────────────────────── */
function renderError(title, msg) {
  mainContent.innerHTML = `
    <div class="error-box" role="alert">
      <div class="e-icon">⚠️</div>
      <div class="e-title">${esc(title)}</div>
      <div class="e-msg">${esc(msg)}</div>
      <a href="index.html" class="e-back">← Back to Inventory</a>
    </div>`;
}

/* ── Fetch wrapper ──────────────────────────────────────────────────────── */
async function callForecast(skuId) {
  // Let network-level TypeErrors propagate as-is so callers can detect them.
  const res = await fetch(FORECAST_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sku_id: skuId, horizon_days: 28 }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    const err = new Error(`HTTP ${res.status}: ${detail}`);
    err.httpStatus = res.status;          // tag for callers to branch on
    throw err;
  }
  return res.json();
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION RENDERERS — each takes the full API response object
   ══════════════════════════════════════════════════════════════════════════ */

/* ── 1. Demand Pattern Diagnosis ────────────────────────────────────────── */
function renderDiagnosis(data) {
  const da = data.demand_analysis ?? {};
  const s = da.seasonality ?? {};

  return `
  <div class="section-card" id="section-diagnosis">
    <div class="section-header">
      <div class="section-icon icon-blue" aria-hidden="true">🔬</div>
      <div class="section-title">Demand Pattern Diagnosis</div>
      <div class="section-subtitle">Syntetos-Boylan classification</div>
    </div>
    <div class="section-body">
      <div class="stat-grid">
        <div class="stat-item">
          <div class="stat-label">Demand Class</div>
          <div class="stat-value" style="font-size:15px;margin-top:2px">
            ${demandClassPill(da.demand_class)}
          </div>
        </div>
        <div class="stat-item">
          <div class="stat-label">ADI</div>
          <div class="stat-value">${fmt(da.adi, 2)}</div>
          <div class="stat-sub">Avg inter-demand interval</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">CV²</div>
          <div class="stat-value">${fmt(da.cv_squared, 2)}</div>
          <div class="stat-sub">Squared coefficient of variation</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Zero Demand Ratio</div>
          <div class="stat-value">${da.zero_demand_ratio != null ? (da.zero_demand_ratio * 100).toFixed(0) + '%' : '—'}</div>
          <div class="stat-sub">Days with no sales</div>
        </div>
      </div>

      <div class="seasonality-row">
        <div class="seas-card">
          <div class="seas-title">Weekly Seasonality</div>
          <div class="seas-label">${esc(s.weekly_label ?? '—')}</div>
          <div class="seas-acf">ACF (lag 7): ${fmtAcf(s.weekly_acf)}</div>
        </div>
        <div class="seas-card">
          <div class="seas-title">Quarterly Seasonality</div>
          <div class="seas-label">${esc(s.quarterly_label ?? '—')}</div>
          <div class="seas-acf">ACF (lag 91): ${fmtAcf(s.quarterly_acf)}</div>
        </div>
      </div>
    </div>
  </div>`;
}

/* ── 2. Model Routing ────────────────────────────────────────────────────── */
function renderRouting(data) {
  const r = data.routing ?? {};
  const scores = r.model_scores ?? {};
  const selModel = r.selected_model ?? null;

  // find the selected model's score
  const selScore = selModel ? scores[selModel] : null;
  const selScoreText = selScore == null ? 'Insufficient Data' : `${selScore.toFixed(1)} / 100`;

  // build model score rows
  const maxScore = Math.max(0, ...Object.values(scores).filter(s => s != null));
  const rows = Object.entries(scores).map(([model, score]) => {
    const isSelected = model === selModel;
    const barWidth = (score != null && maxScore > 0) ? (score / maxScore * 100) : 0;
    return `
      <tr class="${isSelected ? 'selected-row' : ''}">
        <td class="model-name-cell">
          ${esc(model)}${isSelected ? '<span class="selected-chip">Selected</span>' : ''}
        </td>
        <td>
          ${score == null
        ? `<span class="score-null">Insufficient Data</span>`
        : `<div class="score-bar-wrap">
                 <span style="min-width:38px;text-align:right;font-variant-numeric:tabular-nums">${score.toFixed(1)}</span>
                 <div class="score-bar-bg">
                   <div class="score-bar-fill ${isSelected ? 'selected-fill' : ''}"
                        style="width:${barWidth}%"></div>
                 </div>
               </div>`
      }
        </td>
      </tr>`;
  }).join('');

  return `
  <div class="section-card" id="section-routing">
    <div class="section-header">
      <div class="section-icon icon-purple" aria-hidden="true">🤖</div>
      <div class="section-title">Model Routing</div>
      <div class="section-subtitle">WAPE-score-driven selection</div>
    </div>
    <div class="section-body">
      <div class="selected-model-banner">
        <div>
          <div class="selected-model-name">${esc(selModel ?? 'Moving Average (fallback)')}</div>
          <div class="selected-model-score">Score: ${esc(selScoreText)}</div>
        </div>
        <div class="selected-model-justification">${esc(r.justification ?? '—')}</div>
      </div>

      <table class="model-table" aria-label="Model scores">
        <thead>
          <tr>
            <th>Model</th>
            <th>Backtest Score (0–100)</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

/* ── 3. Probabilistic Forecast ───────────────────────────────────────────── */
function renderForecast(data) {
  const fc = data.forecast ?? {};
  const sum = fc.summary ?? {};
  const q = fc.quantiles ?? {};

  const p10 = q.P10 ?? [];
  const p50 = q.P50 ?? [];
  const p90 = q.P90 ?? [];

  const days = Array.from({ length: p50.length }, (_, i) => `Day ${i + 1}`);

  const html = `
  <div class="section-card" id="section-forecast">
    <div class="section-header">
      <div class="section-icon icon-green" aria-hidden="true">📈</div>
      <div class="section-title">Probabilistic Forecast</div>
      <div class="section-subtitle">28-day horizon · P10 / P50 / P90</div>
    </div>
    <div class="section-body">
      <div class="forecast-summary" role="list">
        <div class="forecast-sum-card" role="listitem">
          <div class="forecast-sum-label">7-Day Total</div>
          <div class="forecast-sum-value">${fmtInt(sum['7_days'])}</div>
          <div class="forecast-sum-unit">units (P50)</div>
        </div>
        <div class="forecast-sum-card" role="listitem">
          <div class="forecast-sum-label">14-Day Total</div>
          <div class="forecast-sum-value">${fmtInt(sum['14_days'])}</div>
          <div class="forecast-sum-unit">units (P50)</div>
        </div>
        <div class="forecast-sum-card" role="listitem">
          <div class="forecast-sum-label">28-Day Total</div>
          <div class="forecast-sum-value">${fmtInt(sum['28_days'])}</div>
          <div class="forecast-sum-unit">units (P50)</div>
        </div>
      </div>

      <div id="forecast-chart" aria-label="Forecast chart: P10, P50, P90 daily units" role="img"></div>
    </div>
  </div>`;

  // inject first, then draw Plotly after DOM is ready
  setTimeout(() => drawForecastChart(days, p10, p50, p90), 0);

  return html;
}

function drawForecastChart(days, p10, p50, p90) {
  const el = document.getElementById('forecast-chart');
  if (!el || typeof Plotly === 'undefined') return;

  // Ribbon: P10→P90 filled area
  const ribbon = {
    x: [...days, ...days.slice().reverse()],
    y: [...p90, ...p10.slice().reverse()],
    fill: 'toself',
    fillcolor: 'rgba(88, 166, 255, 0.12)',
    line: { color: 'transparent' },
    name: 'P10–P90 band',
    type: 'scatter',
    hoverinfo: 'skip',
    showlegend: true,
  };
  const traceP10 = {
    x: days, y: p10,
    mode: 'lines',
    line: { color: 'rgba(88,166,255,0.35)', width: 1.2, dash: 'dot' },
    name: 'P10',
    type: 'scatter',
  };
  const traceP90 = {
    x: days, y: p90,
    mode: 'lines',
    line: { color: 'rgba(88,166,255,0.35)', width: 1.2, dash: 'dot' },
    name: 'P90',
    type: 'scatter',
  };
  const traceP50 = {
    x: days, y: p50,
    mode: 'lines+markers',
    line: { color: '#58a6ff', width: 2.5 },
    marker: { color: '#58a6ff', size: 4 },
    name: 'P50 (point forecast)',
    type: 'scatter',
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 10, r: 16, b: 40, l: 48 },
    font: { family: 'Inter, system-ui', color: '#8b949e', size: 11 },
    xaxis: {
      showgrid: false,
      tickfont: { size: 10, color: '#484f58' },
      tickangle: -35,
      nticks: 14,
      linecolor: '#30363d',
      zeroline: false,
    },
    yaxis: {
      showgrid: true,
      gridcolor: '#21262d',
      gridwidth: 1,
      tickfont: { size: 10, color: '#484f58' },
      zeroline: false,
      linecolor: 'transparent',
    },
    legend: {
      orientation: 'h',
      x: 0, y: -0.18,
      font: { size: 11, color: '#8b949e' },
      bgcolor: 'transparent',
    },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#1c2333',
      bordercolor: '#30363d',
      font: { color: '#e6edf3', size: 12 },
    },
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  Plotly.newPlot(el, [ribbon, traceP10, traceP90, traceP50], layout, config);
}

/* ── 4. Replenishment Recommendation ────────────────────────────────────── */
function renderReplenishment(data) {
  const rep = data.replenishment ?? {};
  const sku = data.sku_id ?? SKU_ID;
  const qty = rep.recommended_order ?? 0;

  // Build PO link
  const poHref = `po_confirmation.html?sku=${encodeURIComponent(sku)}&qty=${encodeURIComponent(qty)}`;

  const statCard = (label, value, sub = '') => `
    <div class="stat-item">
      <div class="stat-label">${esc(label)}</div>
      <div class="stat-value">${esc(String(value ?? '—'))}</div>
      ${sub ? `<div class="stat-sub">${esc(sub)}</div>` : ''}
    </div>`;

  const statusCls = statusClass(rep.status);

  return `
  <div class="section-card" id="section-replenishment">
    <div class="section-header">
      <div class="section-icon icon-orange" aria-hidden="true">📦</div>
      <div class="section-title">Replenishment Recommendation</div>
      <div class="section-subtitle">IP-gated · pack-size rounded</div>
    </div>
    <div class="section-body">

      <!-- Recommended order highlight -->
      <div class="order-highlight" aria-label="Recommended order: ${qty} units">
        <div>
          <div class="order-highlight-label">Recommended Order</div>
          <div class="order-highlight-qty">${fmtInt(rep.recommended_order)}</div>
          <div class="order-highlight-unit">units</div>
        </div>
        <div>
          <div class="order-highlight-label">Recommended Cases</div>
          <div class="order-highlight-qty">${fmtInt(rep.recommended_cases)}</div>
          <div class="order-highlight-unit">cases (pack size: ${fmtInt(rep.pack_size)})</div>
        </div>
        <div style="margin-left:auto">
          ${heroBadge(rep.status)}
        </div>
      </div>

      <hr class="divider" />

      <!-- Replenishment metrics grid -->
      <div class="replenishment-grid">
        ${statCard('Current Stock', fmtInt(rep.current_stock), 'units on hand')}
        ${statCard('On Order', fmtInt(rep.on_order_stock), 'open PO units')}
        ${statCard('Inventory Position', fmtInt(rep.inventory_position), 'on hand + on order')}
        ${statCard('Lead Time Demand', fmt(rep.lead_time_demand, 1), 'units during lead time')}
        ${statCard('Service Level Z', fmt(rep.service_level_z, 2), 'z-score for category')}
        ${statCard('Safety Stock', fmtInt(rep.safety_stock), 'buffer units')}
        ${statCard('Reorder Point', fmtInt(rep.reorder_point), 'trigger threshold')}
        ${statCard('Pack Size', fmtInt(rep.pack_size), 'units per case')}
      </div>

      <!-- PO button -->
      <a
        id="po-btn"
        href="${esc(poHref)}"
        aria-label="Create purchase order for ${esc(qty.toString())} units of ${esc(sku)}"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z"/>
          <line x1="3" y1="6" x2="21" y2="6"/>
          <path d="M16 10a4 4 0 01-8 0"/>
        </svg>
        Create Purchase Order
      </a>
      <div class="po-btn-sub">
        Will navigate to PO confirmation for ${esc(sku)} · ${fmtInt(qty)} units
      </div>
    </div>
  </div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   PAGE BOOTSTRAP
   ══════════════════════════════════════════════════════════════════════════ */

/* ── Guard: no SKU in URL ────────────────────────────────────────────────── */
if (!SKU_ID) {
  renderError('No SKU specified', 'Please navigate to this page from the inventory table.');
} else {
  // Update breadcrumb
  const bc = document.getElementById('breadcrumb-sku');
  if (bc) bc.textContent = SKU_ID;

  // Update page title
  document.title = `${SKU_ID} — StockSense`;

  bootstrapPage();
}

/* ── Initial page load ──────────────────────────────────────────────────── */
async function bootstrapPage() {
  // Show skeleton hero while we make the initial fetch
  try {
    const data = await callForecast(SKU_ID);
    renderHero(data);
  } catch (err) {
    console.error('[forecast_card] Initial load failed:', err);
    if (err instanceof TypeError) {
      // fetch() throws TypeError when the network is unreachable
      renderError(
        'Could not reach the server',
        'Make sure the backend is running and try again.'
      );
    } else if (err.httpStatus === 404) {
      renderError(
        'SKU not found',
        `"${SKU_ID}" does not exist in the system. Please select a valid SKU from the inventory table.`
      );
    } else {
      renderError(
        `Failed to load ${SKU_ID}`,
        `The server returned an unexpected error.\n\nDetail: ${err.message}`
      );
    }
  }
}

/* ── Render hero + generate button ─────────────────────────────────────── */
function renderHero(data) {
  const rep = data.replenishment ?? {};
  const name = data.product_name ?? SKU_ID;
  const cat = data.category ?? '—';

  // Update breadcrumb / title with full name
  const bc = document.getElementById('breadcrumb-sku');
  if (bc) bc.textContent = `${SKU_ID} — ${name}`;
  document.title = `${name} — StockSense`;

  mainContent.innerHTML = `
    <!-- ── SKU Hero ──────────────────────────────────────────────────── -->
    <div class="sku-hero" id="sku-hero">
      <div class="sku-hero-left">
        <div class="sku-tag" aria-label="SKU ID">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/>
            <line x1="1" y1="10" x2="23" y2="10"/>
          </svg>
          ${esc(SKU_ID)}
        </div>
        <h1>${esc(name)}</h1>
        <div class="sku-hero-category">${esc(cat)}</div>
        <div class="hero-meta" role="list">
          <div class="hero-meta-item" role="listitem">
            <span class="hero-meta-label">Current Stock</span>
            <span class="hero-meta-value">${fmtInt(rep.current_stock)}</span>
          </div>
          <div class="hero-meta-item" role="listitem">
            <span class="hero-meta-label">Lead Time</span>
            <span class="hero-meta-value">${rep.lead_time_days != null ? rep.lead_time_days + 'd' : '—'}</span>
          </div>
          <div class="hero-meta-item" role="listitem">
            <span class="hero-meta-label">Pack Size</span>
            <span class="hero-meta-value">${fmtInt(rep.pack_size)}</span>
          </div>
          <div class="hero-meta-item" role="listitem">
            <span class="hero-meta-label">On Order</span>
            <span class="hero-meta-value">${fmtInt(rep.on_order_stock)}</span>
          </div>
        </div>
      </div>
      <div class="sku-hero-right">
        ${heroBadge(rep.status)}
        <div style="font-size:12px;color:var(--text-muted);text-align:right">
          Inventory Position: ${fmtInt(rep.inventory_position)}
        </div>
      </div>
    </div>

    <!-- ── Limited history banner (shown if needed) ──────────────────── -->
    <div id="limited-history-banner" style="display:none"
         class="banner banner-warn" role="alert" aria-live="polite">
      <span class="banner-icon">⚠️</span>
      <span>
        <strong>Limited history:</strong>
        Defaulting to moving-average baseline. Results may be less accurate.
      </span>
    </div>

    <!-- ── Generate button + analyzing state ─────────────────────────── -->
    <button id="generate-btn" aria-label="Generate AI demand forecast for ${esc(SKU_ID)}">
      <div class="btn-spinner" aria-hidden="true"></div>
      <svg class="btn-icon" width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
           stroke-linejoin="round" aria-hidden="true">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
      </svg>
      <span id="generate-btn-label">Generate AI Forecast</span>
    </button>

    <div id="analyzing-msg" role="status" aria-live="polite">
      <div class="pulse-dot" aria-hidden="true"></div>
      Analyzing demand patterns and evaluating candidate models…
    </div>

    <!-- ── Analysis sections rendered here ───────────────────────────── -->
    <div id="analysis-sections"></div>
  `;

  // Wire the generate button (toggles: generate → hide → show again, no re-fetch)
  const btn = document.getElementById('generate-btn');
  btn.addEventListener('click', () => toggleForecast());


}


function setButtonLabel(text) {
  const label = document.getElementById('generate-btn-label');
  if (label) label.textContent = text;
}

async function toggleForecast() {
  const sections = document.getElementById('analysis-sections');
  if (!sections) return;

  if (_sectionsVisible) {
    // Fold back — just hide, no re-fetch
    sections.style.display = 'none';
    _sectionsVisible = false;
    setButtonLabel('Generate AI Forecast');
    return;
  }

  if (_forecastData) {
    // Already have the result from a previous click — show it instantly
    sections.style.display = '';
    _sectionsVisible = true;
    setButtonLabel('Hide Forecast');
    return;
  }

  // First click ever — actually fetch and render
  const btn = document.getElementById('generate-btn');
  const analyzingMsg = document.getElementById('analyzing-msg');
  const limitBanner = document.getElementById('limited-history-banner');

  if (btn) { btn.classList.add('loading'); btn.disabled = true; }
  if (analyzingMsg) analyzingMsg.classList.add('visible');

  try {
    const data = await callForecast(SKU_ID);
    _forecastData = data;
    _sectionsVisible = true;

    if (limitBanner) {
      limitBanner.style.display = data.limited_history ? 'flex' : 'none';
    }

    sections.style.display = '';
    sections.innerHTML =
      renderDiagnosis(data) +
      renderRouting(data) +
      renderForecast(data) +
      renderReplenishment(data);

    setButtonLabel('Hide Forecast');

  } catch (err) {
    console.error('[forecast_card] Forecast failed:', err);
    sections.style.display = '';

    let errTitle, errMsg;
    if (err instanceof TypeError) {
      errTitle = 'Could not reach the server';
      errMsg   = 'Make sure the backend is running and try again.';
    } else if (err.httpStatus === 404) {
      errTitle = 'SKU not found';
      errMsg   = `"${SKU_ID}" does not exist in the system. Please select a valid SKU from the inventory table.`;
    } else {
      errTitle = 'Forecast failed';
      errMsg   = `The server returned an unexpected error.\n\nDetail: ${esc(err.message)}`;
    }

    sections.innerHTML = `
      <div class="error-box" role="alert">
        <div class="e-icon">⚠️</div>
        <div class="e-title">${esc(errTitle)}</div>
        <div class="e-msg">${esc(errMsg)}</div>
        <a href="index.html" class="e-back">← Back to Inventory</a>
      </div>`;
  } finally {
    if (btn) { btn.classList.remove('loading'); btn.disabled = false; }
    if (analyzingMsg) analyzingMsg.classList.remove('visible');
  }
}

