/**
 * po_modal.js
 * Screen 3 — Purchase Order Confirmation
 *
 * Flow:
 *   1. Read sku_id and qty from URL query string
 *   2. POST /api/purchase-order on page load
 *   3. Show "Recording purchase order…" while in-flight
 *   4. Render the API response verbatim — no field is computed or assumed.
 *
 * Guardrails:
 *   - new_status rendered exactly as returned (never assumed to be "Healthy")
 *   - No before/after status transition logic
 *   - All displayed values taken directly from the API response object
 */

'use strict';

/* ── Constants ─────────────────────────────────────────────────────────── */
const API_BASE = 'http://127.0.0.1:8000';
const PO_ENDPOINT = `${API_BASE}/api/purchase-order`;

/* ── Read params from URL ───────────────────────────────────────────────── */
const urlParams = new URLSearchParams(window.location.search);
const SKU_ID    = urlParams.get('sku') ?? '';
const QTY_RAW   = urlParams.get('qty') ?? '';
const QTY       = parseInt(QTY_RAW, 10);

/* ── DOM refs ─────────────────────────────────────────────────── */
const poContent    = document.getElementById('po-content');
const bcDetail     = document.getElementById('bc-detail');
const navBackSku   = document.getElementById('nav-back-sku');
const navBackLabel = document.getElementById('nav-back-sku-label');

/* ── Breadcrumb + header nav links ───────────────────────────────── */
if (SKU_ID) {
  const skuHref = `product_detail.html?sku=${encodeURIComponent(SKU_ID)}`;
  if (bcDetail)     { bcDetail.textContent = SKU_ID; bcDetail.href = skuHref; }
  if (navBackSku)   { navBackSku.href = skuHref; }
  if (navBackLabel) { navBackLabel.textContent = `← Back to ${SKU_ID} Detail`; }
}

/* ── Escape helper ──────────────────────────────────────────────────────── */
function esc(v) {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ── Format helpers ─────────────────────────────────────────────────────── */
const fmtInt = (v) => (v == null ? '—' : Number(v).toLocaleString());

/* ── Status badge (identical rules to index.html + product_detail.html) ── */
function badgeClass(status) {
  switch ((status ?? '').toLowerCase()) {
    case 'critical': return 'badge-critical';
    case 'low':      return 'badge-low';
    case 'healthy':  return 'badge-healthy';
    default:         return 'badge-unknown';
  }
}

function renderBadge(status) {
  const cls   = badgeClass(status);
  const label = status ?? '—';
  return `<span class="badge ${cls}" role="img" aria-label="Post-PO status: ${esc(label)}">
            <span class="badge-dot" aria-hidden="true"></span>${esc(label)}
          </span>`;
}

/* ── Loading state ──────────────────────────────────────────────────────── */
function showLoading() {
  poContent.innerHTML = `
    <div id="loading-state" role="status" aria-live="polite">
      <div class="loading-ring" aria-hidden="true"></div>
      <div class="loading-label">Recording purchase order…</div>
      <div class="loading-sublabel">${esc(SKU_ID)} · ${fmtInt(QTY)} units</div>
    </div>`;
}

/* ── Error state ──────────────────────────────────────────────────────── */
/**
 * @param {string} title
 * @param {string} detail
 * @param {string} [backHref] - Defaults to 'index.html' when omitted or when SKU_ID is empty.
 */
function showError(title, detail, backHref) {
  // If no explicit back-link is provided, always fall back to index.html
  // rather than building a broken product_detail.html?sku= link.
  const href = backHref || 'index.html';
  const linkLabel = href === 'index.html' ? '← Back to Inventory' : '← Back to SKU Detail';
  poContent.innerHTML = `
    <div class="error-box" role="alert">
      <div class="e-icon">⚠️</div>
      <div class="e-title">${esc(title)}</div>
      <div class="e-msg">${esc(detail)}</div>
      <a href="${esc(href)}">${linkLabel}</a>
    </div>`;
}

/* ── Guard: missing or invalid params ─────────────────────────────── */
if (!SKU_ID || !QTY_RAW || isNaN(QTY) || QTY <= 0) {
  // Don't attempt the POST — params are missing or malformed.
  // Always link back to index.html (SKU_ID may be empty, so a product_detail link would be broken).
  showError(
    'Missing order details',
    'The required SKU and quantity were not provided.\n' +
    'Please navigate here from the SKU detail page, not directly.',
    'index.html'
  );
} else {
  showLoading();
  recordPurchaseOrder();
}

/* ── API call ───────────────────────────────────────────────────────────── */
async function recordPurchaseOrder() {
  try {
    const res = await fetch(PO_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sku_id: SKU_ID, quantity: QTY }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${detail}`);
    }

    const data = await res.json();
    renderConfirmation(data);

  } catch (err) {
    console.error('[po_modal] Purchase order failed:', err);
    const backHref = SKU_ID
      ? `product_detail.html?sku=${encodeURIComponent(SKU_ID)}`
      : 'index.html';

    if (err instanceof TypeError) {
      // fetch() throws TypeError when the server is unreachable
      showError(
        'Could not reach the server',
        'Make sure the backend is running and try again.',
        backHref
      );
    } else {
      showError(
        'Purchase order failed',
        `The server returned an unexpected error.\n\nDetail: ${err.message}`,
        backHref
      );
    }
  }
}

/* ── Render confirmation receipt ────────────────────────────────────────── */
function renderConfirmation(data) {
  // Extract every field verbatim from the API response
  const poId             = data.po_id;
  const skuId            = data.sku_id;
  const quantityOrdered  = data.quantity_ordered;
  const currentStock     = data.current_stock;
  const onOrderStock     = data.on_order_stock;
  const inventoryPosition = data.inventory_position;
  const reorderPoint     = data.reorder_point;
  const newStatus        = data.new_status;

  // Update page title with the PO ID
  document.title = `${esc(poId ?? 'PO Confirmed')} — StockSense`;

  poContent.innerHTML = `

    <!-- ── Success header ─────────────────────────────────────────────── -->
    <div class="po-success-header" role="status" aria-live="polite">
      <div class="po-check-ring" aria-hidden="true">✓</div>
      <div class="po-success-title">Purchase Order Recorded</div>
      <div class="po-id-chip" id="po-id-display" aria-label="Purchase order ID: ${esc(poId ?? '—')}">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        ${esc(poId ?? '—')}
      </div>
    </div>

    <!-- ── Order quantity hero ────────────────────────────────────────── -->
    <div class="order-hero" aria-label="Order summary for ${esc(skuId ?? SKU_ID)}">
      <div class="order-hero-qty-block">
        <div class="order-hero-qty-label">Units Ordered</div>
        <div class="order-hero-qty" id="qty-display">${fmtInt(quantityOrdered)}</div>
        <div class="order-hero-qty-unit">units</div>
      </div>
      <div class="order-hero-divider" aria-hidden="true"></div>
      <div class="order-hero-meta">
        <div class="order-hero-row">
          <span class="order-hero-key">SKU</span>
          <span class="order-hero-val" id="sku-display">${esc(skuId ?? SKU_ID)}</span>
        </div>
        <div class="order-hero-row">
          <span class="order-hero-key">PO Reference</span>
          <span class="order-hero-val" style="font-family:'SF Mono','Fira Mono',monospace;font-size:12px">
            ${esc(poId ?? '—')}
          </span>
        </div>
        <div class="order-hero-row">
          <span class="order-hero-key">Post-PO Status</span>
          <span>${renderBadge(newStatus)}</span>
        </div>
      </div>
    </div>

    <!-- ── Inventory position section ────────────────────────────────── -->
    <div class="section-card">
      <div class="section-header">
        <div class="section-icon icon-blue" aria-hidden="true">📊</div>
        <div class="section-title">Post-Order Inventory Position</div>
      </div>
      <div class="section-body">
        <div class="metrics-row">
          <div class="metric-item">
            <div class="metric-label">On-Hand Stock</div>
            <div class="metric-value" id="current-stock-display">${fmtInt(currentStock)}</div>
            <div class="metric-sub">units on shelf</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">On Order</div>
            <div class="metric-value" id="on-order-display">${fmtInt(onOrderStock)}</div>
            <div class="metric-sub">open PO units (incl. this order)</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">Inventory Position</div>
            <div class="metric-value" id="ip-display">${fmtInt(inventoryPosition)}</div>
            <div class="metric-formula">= On-Hand + On-Order</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">Reorder Point</div>
            <div class="metric-value" id="rop-display">${fmtInt(reorderPoint)}</div>
            <div class="metric-sub">replenishment trigger</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── New status section ─────────────────────────────────────────── -->
    <div class="section-card">
      <div class="section-header">
        <div class="section-icon icon-orange" aria-hidden="true">🔄</div>
        <div class="section-title">Updated Replenishment Status</div>
      </div>
      <div class="section-body">
        <div class="status-result-row">
          <div class="status-result-label">
            Post-PO status for <strong>${esc(skuId ?? SKU_ID)}</strong>
          </div>
          <div id="new-status-display">
            ${renderBadge(newStatus)}
          </div>
        </div>
        <hr class="divider" />
        <p style="font-size:13px;color:var(--text-secondary);line-height:1.7">
          Status is derived from <strong style="color:var(--text-primary)">
          Inventory Position (${fmtInt(inventoryPosition)})</strong> vs
          <strong style="color:var(--text-primary)">
          Reorder Point (${fmtInt(reorderPoint)})</strong>.
          The updated figure reflects this order's contribution to on-order stock.
        </p>
      </div>
    </div>

    <!-- ── Action footer removed: links are now in the sub-header ── -->
  `;
}
