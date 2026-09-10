/**
 * inventory_table.js
 * Screen 1 — Inventory Table
 *
 * On load:
 *   1. Shows skeleton rows while fetching
 *   2. POSTs /api/forecast once per SKU (6 total) — no invented endpoints
 *   3. Renders rows with sortable columns, search, and status filter
 *   4. Clicking a row navigates to product_detail.html?sku=<sku_id>
 *
 * All status badges are rendered verbatim from replenishment.status —
 * the frontend never derives status from stock numbers.
 */

/* ── Constants ──────────────────────────────────────────────────────────── */
const API_BASE = 'http://127.0.0.1:8000';
const FORECAST_ENDPOINT = `${API_BASE}/api/forecast`;

/**
 * Canonical SKU list — sourced directly from backend/data/demo_inventory.csv
 * (Product IDs: SKU-101, SKU-202, SKU-303, SKU-404, SKU-505, SKU-606)
 */
const SKU_IDS = ['SKU-101', 'SKU-202', 'SKU-303', 'SKU-404', 'SKU-505', 'SKU-606'];

/* ── State ──────────────────────────────────────────────────────────────── */
let allRows = [];           // full dataset after load
let sortCol = null;         // currently sorted column key
let sortDir = 'asc';        // 'asc' | 'desc'
let searchQuery = '';
let statusFilter = '';

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const tableBody   = document.getElementById('table-body');
const footerStatus = document.getElementById('footer-status');
const footerTs    = document.getElementById('footer-ts');
const rowCountEl  = document.getElementById('row-count');
const searchInput = document.getElementById('search-input');
const filterSelect = document.getElementById('filter-status');
const refreshBtn  = document.getElementById('refresh-btn');

/* ── Status badge helper ────────────────────────────────────────────────── */
/**
 * Map the raw status string the API returns → badge CSS class + display label.
 * We do NOT reclassify — we just normalise casing for the CSS key.
 */
function badgeClass(status) {
  if (!status) return 'badge-unknown';
  const s = status.toLowerCase();
  if (s === 'critical') return 'badge-critical';
  if (s === 'low')      return 'badge-low';
  if (s === 'healthy')  return 'badge-healthy';
  return 'badge-unknown';
}

function renderBadge(status) {
  const cls  = badgeClass(status);
  const label = status || '—';
  return `<span class="badge ${cls}" role="img" aria-label="Status: ${label}">
            <span class="badge-dot" aria-hidden="true"></span>${label}
          </span>`;
}

/* ── Skeleton injector ──────────────────────────────────────────────────── */
function showSkeletons(count = 6) {
  const rows = Array.from({ length: count }, () => `
    <tr class="skeleton-row" aria-busy="true" aria-label="Loading…">
      <td class="cell-product">
        <div class="skel skel-sm" style="margin-bottom:5px"></div>
        <div class="skel skel-lg"></div>
      </td>
      <td><div class="skel skel-md"></div></td>
      <td><div class="skel skel-num"></div></td>
      <td><div class="skel skel-num"></div></td>
      <td><div class="skel skel-num"></div></td>
      <td><div class="skel skel-badge"></div></td>
      <td class="cell-arrow"></td>
    </tr>`
  ).join('');
  tableBody.innerHTML = rows;
  footerStatus.textContent = 'Fetching inventory data…';
}

/* ── KPI updater ────────────────────────────────────────────────────────── */
function updateKPIs(rows) {
  const total    = rows.length;
  const critical = rows.filter(r => r.status?.toLowerCase() === 'critical').length;
  const low      = rows.filter(r => r.status?.toLowerCase() === 'low').length;
  const healthy  = rows.filter(r => r.status?.toLowerCase() === 'healthy').length;

  const set = (id, val, sub) => {
    const el = document.getElementById(id);
    if (el) { el.textContent = val; el.classList.remove('skeleton'); }
    const subEl = document.getElementById(`${id}-sub`);
    if (subEl) { subEl.textContent = sub; subEl.classList.remove('skeleton'); }
  };

  set('kpi-total',   total,    'tracked products');
  set('kpi-critical', critical, critical === 1 ? 'needs action' : 'need action');
  set('kpi-low',     low,      low === 1 ? 'SKU below reorder' : 'SKUs below reorder');
  set('kpi-healthy', healthy,  healthy === 1 ? 'SKU in stock' : 'SKUs in stock');
}

/* ── Fetch one SKU ──────────────────────────────────────────────────────── */
async function fetchSku(skuId) {
  const response = await fetch(FORECAST_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sku_id: skuId, horizon_days: 28 }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`${skuId}: HTTP ${response.status} — ${detail}`);
  }

  const data = await response.json();

  /* Extract columns per spec §3.1 response shape */
  return {
    sku_id:          data.sku_id,
    product_name:    data.product_name,
    category:        data.category,
    current_stock:   data.replenishment?.current_stock   ?? null,
    sales_week:      data.forecast?.summary?.['7_days']  ?? null,
    lead_time_days:  data.replenishment?.lead_time_days  ?? null,
    status:          data.replenishment?.status          ?? null,
    _raw:            data,   // keep for future detail page pass-through
  };
}

/* ── Fetch all SKUs concurrently ────────────────────────────────────────── */
async function fetchAll() {
  const results = await Promise.allSettled(SKU_IDS.map(fetchSku));

  const rows   = [];
  const errors = [];

  results.forEach((r, i) => {
    if (r.status === 'fulfilled') {
      rows.push(r.value);
    } else {
      errors.push({ skuId: SKU_IDS[i], reason: r.reason?.message ?? String(r.reason) });
      console.error(`[inventory_table] Failed to load ${SKU_IDS[i]}:`, r.reason);
    }
  });

  return { rows, errors };
}

/* ── Sort helper ────────────────────────────────────────────────────────── */
function sortedRows(rows) {
  if (!sortCol) return rows;
  return [...rows].sort((a, b) => {
    let va = a[sortCol];
    let vb = b[sortCol];

    // numeric columns
    if (typeof va === 'number' || typeof vb === 'number') {
      va = va ?? -Infinity;
      vb = vb ?? -Infinity;
      return sortDir === 'asc' ? va - vb : vb - va;
    }

    // string columns
    va = (va ?? '').toString().toLowerCase();
    vb = (vb ?? '').toString().toLowerCase();
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ?  1 : -1;
    return 0;
  });
}

/* ── Filter helper ──────────────────────────────────────────────────────── */
function filteredRows(rows) {
  let out = rows;

  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    out = out.filter(r =>
      r.sku_id.toLowerCase().includes(q) ||
      (r.product_name ?? '').toLowerCase().includes(q) ||
      (r.category ?? '').toLowerCase().includes(q)
    );
  }

  if (statusFilter) {
    out = out.filter(r =>
      (r.status ?? '').toLowerCase() === statusFilter.toLowerCase()
    );
  }

  return out;
}

/* ── Render table rows ──────────────────────────────────────────────────── */
function renderRows() {
  const visible = filteredRows(sortedRows(allRows));

  if (visible.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="7">
          <div class="state-box">
            <div class="state-icon">🔍</div>
            <div class="state-title">No results</div>
            <div class="state-msg">No SKUs match your current search or filter.</div>
          </div>
        </td>
      </tr>`;
    rowCountEl.textContent = '0 results';
    return;
  }

  const totalShown = visible.length;
  const totalAll   = allRows.length;
  rowCountEl.textContent = totalShown === totalAll
    ? `${totalAll} products`
    : `${totalShown} of ${totalAll} products`;

  tableBody.innerHTML = visible.map(row => {
    const stockCls = row.status?.toLowerCase() === 'critical' ? ' stock-critical' : '';
    const stockVal = row.current_stock != null ? row.current_stock.toLocaleString() : '—';
    const salesVal = row.sales_week    != null ? row.sales_week.toLocaleString()    : '—';
    const leadVal  = row.lead_time_days != null ? `${row.lead_time_days} d`         : '—';

    return `
      <tr
        id="row-${row.sku_id}"
        data-sku="${row.sku_id}"
        tabindex="0"
        role="link"
        aria-label="View details for ${row.product_name ?? row.sku_id}"
        title="Click to view ${row.sku_id} details"
      >
        <td class="cell-product">
          <div class="product-id">${escHtml(row.sku_id)}</div>
          <div class="product-name">${escHtml(row.product_name ?? '—')}</div>
        </td>
        <td class="cell-category">
          <span>${escHtml(row.category ?? '—')}</span>
        </td>
        <td class="cell-number${stockCls}">${stockVal}</td>
        <td class="cell-number">${salesVal}</td>
        <td class="cell-number">${leadVal}</td>
        <td>${renderBadge(row.status)}</td>
        <td class="cell-arrow" aria-hidden="true">›</td>
      </tr>`;
  }).join('');

  /* ── Row click / keyboard nav ── */
  tableBody.querySelectorAll('tr[data-sku]').forEach(tr => {
    const navigate = () => {
      const sku = tr.dataset.sku;
      window.location.href = `product_detail.html?sku=${encodeURIComponent(sku)}`;
    };
    tr.addEventListener('click', navigate);
    tr.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(); }
    });
  });
}

/* ── Sort column headers ────────────────────────────────────────────────── */
function initSortHeaders() {
  document.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortCol = col;
        sortDir = 'asc';
      }
      // update aria-sort attributes
      document.querySelectorAll('th').forEach(h => {
        h.classList.remove('sort-asc', 'sort-desc');
        h.removeAttribute('aria-sort');
      });
      th.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
      th.setAttribute('aria-sort', sortDir === 'asc' ? 'ascending' : 'descending');
      renderRows();
    });
  });
}

/* ── Footer timestamp ───────────────────────────────────────────────────── */
function stampFooter() {
  const now = new Date();
  footerTs.textContent = `Updated ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

/* ── Escape HTML ────────────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── Error state renderer ───────────────────────────────────────────────── */
function showError(errors) {
  const msgs = errors.map(e => `<li><strong>${escHtml(e.skuId)}</strong>: ${escHtml(e.reason)}</li>`).join('');
  tableBody.innerHTML = `
    <tr>
      <td colspan="7">
        <div class="state-box">
          <div class="state-icon">⚠️</div>
          <div class="state-title">Failed to load inventory</div>
          <div class="state-msg">
            Could not reach the backend at <code>${API_BASE}</code>.
            Make sure the server is running on port 8000.<br/>
            <ul style="margin-top:8px;text-align:left;list-style:disc;padding-left:20px">${msgs}</ul>
          </div>
          <button id="retry-btn">Retry</button>
        </div>
      </td>
    </tr>`;
  document.getElementById('retry-btn')?.addEventListener('click', loadInventory);
  footerStatus.textContent = `${errors.length} of ${SKU_IDS.length} SKU(s) failed to load`;
}

/* ── Spinner animation on refresh button ────────────────────────────────── */
function setRefreshBusy(busy) {
  refreshBtn.disabled = busy;
  const svg = refreshBtn.querySelector('svg');
  if (busy) {
    svg.style.animation = 'spin 0.8s linear infinite';
  } else {
    svg.style.animation = '';
  }
}

// inject spin keyframes for the refresh icon
const spinStyle = document.createElement('style');
spinStyle.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
document.head.appendChild(spinStyle);

/* ── Main loader ────────────────────────────────────────────────────────── */
async function loadInventory() {
  showSkeletons(SKU_IDS.length);
  setRefreshBusy(true);
  allRows = [];

  try {
    const { rows, errors } = await fetchAll();
    allRows = rows;

    if (rows.length === 0 && errors.length > 0) {
      showError(errors);
      return;
    }

    updateKPIs(rows);
    renderRows();
    stampFooter();

    if (errors.length > 0) {
      footerStatus.textContent =
        `Loaded ${rows.length} SKU(s) — ${errors.length} failed (see console)`;
    } else {
      footerStatus.textContent = `All ${rows.length} SKUs loaded successfully`;
    }
  } catch (err) {
    console.error('[inventory_table] Unexpected error:', err);
    showError([{ skuId: 'ALL', reason: err.message }]);
  } finally {
    setRefreshBusy(false);
  }
}

/* ── Wire controls ──────────────────────────────────────────────────────── */
searchInput.addEventListener('input', () => {
  searchQuery = searchInput.value.trim();
  renderRows();
});

filterSelect.addEventListener('change', () => {
  statusFilter = filterSelect.value;
  renderRows();
});

refreshBtn.addEventListener('click', loadInventory);

initSortHeaders();

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
loadInventory();
