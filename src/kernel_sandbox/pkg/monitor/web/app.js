const state = {
  snapshot: null,
  cluster: 'all',
  query: '',
  selectedKey: '',
  timer: null,
};

const elements = {
  updatedAt: document.querySelector('#updatedAt'),
  targetHealth: document.querySelector('#targetHealth'),
  refreshButton: document.querySelector('#refreshButton'),
  summaryScope: document.querySelector('#summaryScope'),
  summaryScopeHint: document.querySelector('#summaryScopeHint'),
  summaryScopeType: document.querySelector('#summaryScopeType'),
  summaryUtilizationLabel: document.querySelector('#summaryUtilizationLabel'),
  summaryUtilization: document.querySelector('#summaryUtilization'),
  summaryPoolsLabel: document.querySelector('#summaryPoolsLabel'),
  summaryPools: document.querySelector('#summaryPools'),
  summaryWorkers: document.querySelector('#summaryWorkers'),
  summaryPending: document.querySelector('#summaryPending'),
  summaryQueued: document.querySelector('#summaryQueued'),
  summaryQueueAvg: document.querySelector('#summaryQueueAvg'),
  summaryQueueMax: document.querySelector('#summaryQueueMax'),
  summaryQueueRate: document.querySelector('#summaryQueueRate'),
  summarySessions: document.querySelector('#summarySessions'),
  clusterTabs: document.querySelector('#clusterTabs'),
  searchInput: document.querySelector('#searchInput'),
  poolRows: document.querySelector('#poolRows'),
  emptyState: document.querySelector('#emptyState'),
  detailPool: document.querySelector('#detailPool'),
  detailCluster: document.querySelector('#detailCluster'),
  detailUtilization: document.querySelector('#detailUtilization'),
  detailLifetime: document.querySelector('#detailLifetime'),
  detailBalance: document.querySelector('#detailBalance'),
  detailSessions: document.querySelector('#detailSessions'),
  detailSessionStatus: document.querySelector('#detailSessionStatus'),
  sessionPrefixTotal: document.querySelector('#sessionPrefixTotal'),
  sessionPrefixRows: document.querySelector('#sessionPrefixRows'),
  historyRange: document.querySelector('#historyRange'),
  detailBusy: document.querySelector('#detailBusy'),
  detailIdle: document.querySelector('#detailIdle'),
  detailPending: document.querySelector('#detailPending'),
  detailQueueLength: document.querySelector('#detailQueueLength'),
  detailQueueCount: document.querySelector('#detailQueueCount'),
  detailQueueAvg: document.querySelector('#detailQueueAvg'),
  detailQueueMax: document.querySelector('#detailQueueMax'),
  detailQueueRate: document.querySelector('#detailQueueRate'),
  workerTotal: document.querySelector('#workerTotal'),
  busyBar: document.querySelector('#busyBar'),
  idleBar: document.querySelector('#idleBar'),
  pendingBar: document.querySelector('#pendingBar'),
  detailImage: document.querySelector('#detailImage'),
  historyChart: document.querySelector('#historyChart'),
};

function percent(value) {
  return value == null ? '--' : `${(value * 100).toFixed(1)}%`;
}

function duration(value) {
  if (value == null) {
    return '--';
  }
  if (value < 1000) {
    return `${value.toFixed(0)} ms`;
  }
  if (value < 60000) {
    return `${(value / 1000).toFixed(1)} s`;
  }
  return `${(value / 60000).toFixed(1)} min`;
}

function rate(value) {
  return value == null ? '--' : value.toFixed(1);
}

function sessionStatus(sessions) {
  if (!sessions) {
    return '--';
  }
  const parts = [];
  if (sessions.ready) {
    parts.push(`${sessions.ready} ready`);
  }
  if (sessions.creating) {
    parts.push(`${sessions.creating} creating`);
  }
  if (sessions.failed) {
    parts.push(`${sessions.failed} failed`);
  }
  return parts.length ? parts.join(' · ') : '0 active';
}
function renderSessionPrefixes(prefixes) {
  const entries = Object.entries(prefixes || {}).sort((left, right) => {
    if (right[1] !== left[1]) {
      return right[1] - left[1];
    }
    return left[0].localeCompare(right[0]);
  });
  elements.sessionPrefixTotal.textContent = `${entries.length} ${entries.length === 1 ? 'prefix' : 'prefixes'}`;
  elements.sessionPrefixRows.innerHTML = entries.length
    ? entries.map(([prefix, count]) => `
      <div class="prefix-row">
        <code title="${escapeHTML(prefix)}">${escapeHTML(prefix)}</code>
        <strong>${count}</strong>
      </div>
    `).join('')
    : '<span class="prefix-empty">No active session prefixes</span>';
}

function escapeHTML(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function visiblePools() {
  if (!state.snapshot) {
    return [];
  }
  const query = state.query.trim().toLowerCase();
  return state.snapshot.pools.filter((pool) => {
    const clusterMatch = state.cluster === 'all' || pool.cluster === state.cluster;
    const queryMatch = !query || `${pool.gpu_pool_id} ${pool.image} ${pool.cluster}`.toLowerCase().includes(query);
    return clusterMatch && queryMatch;
  });
}

function selectedPool() {
  if (!state.snapshot || !state.selectedKey) {
    return null;
  }
  const pool = state.snapshot.pools.find((item) => item.key === state.selectedKey);
  if (!pool || (state.cluster !== 'all' && pool.cluster !== state.cluster)) {
    return null;
  }
  return pool;
}

function ensureSelection() {
  if (state.cluster === 'all') {
    if (state.selectedKey && !selectedPool()) {
      state.selectedKey = '';
    }
    return;
  }
  const pool = selectedPool();
  if (pool) {
    return;
  }
  state.selectedKey = state.snapshot.pools.find((item) => item.cluster === state.cluster)?.key || '';
}
function renderSummary() {
  const { summary, generated_at: generatedAt } = state.snapshot;
  const pool = selectedPool();
  if (pool) {
    const queueSupported = pool.queue_supported;
    elements.summaryScope.textContent = `${pool.cluster} / ${pool.gpu_pool_id}`;
    elements.summaryScopeHint.textContent = 'Every value below is isolated to this GPU pool';
    elements.summaryScopeType.textContent = 'Pool';
    elements.summaryUtilizationLabel.textContent = 'Pool utilization';
    elements.summaryPoolsLabel.textContent = 'Replicas';
    elements.summaryUtilization.textContent = percent(pool.utilization);
    elements.summaryPools.textContent = pool.replicas;
    elements.summaryWorkers.textContent = `${pool.workers.busy} / ${pool.workers.busy + pool.workers.idle}`;
    elements.summaryPending.textContent = pool.workers.pending;
    elements.summaryQueued.textContent = queueSupported ? pool.queue_length : 'N/A';
    elements.summaryQueueAvg.textContent = queueSupported ? duration(pool.queue_avg_ms) : 'N/A';
    elements.summaryQueueMax.textContent = queueSupported ? duration(pool.queue_max_ms || null) : 'N/A';
    elements.summaryQueueRate.textContent = queueSupported ? rate(pool.queue_rate_per_minute) : 'N/A';
    elements.summarySessions.textContent = `${pool.sessions.total} (${pool.sessions.ready} ready)`;
  } else {
    const queueSupported = summary.queue_supported;
    elements.summaryScope.textContent = 'All clusters';
    elements.summaryScopeHint.textContent = 'Aggregated across every GPU pool';
    elements.summaryScopeType.textContent = 'Global';
    elements.summaryUtilizationLabel.textContent = 'Cluster utilization';
    elements.summaryPoolsLabel.textContent = 'GPU pools';
    elements.summaryUtilization.textContent = percent(summary.utilization);
    elements.summaryPools.textContent = summary.pools;
    elements.summaryWorkers.textContent = `${summary.workers_busy} / ${summary.workers_busy + summary.workers_idle}`;
    elements.summaryPending.textContent = summary.workers_pending;
    elements.summaryQueued.textContent = queueSupported ? summary.queued_commands : 'N/A';
    elements.summaryQueueAvg.textContent = queueSupported ? duration(summary.queue_avg_ms) : 'N/A';
    elements.summaryQueueMax.textContent = queueSupported ? duration(summary.queue_max_ms || null) : 'N/A';
    elements.summaryQueueRate.textContent = queueSupported ? rate(summary.queue_rate_per_minute) : 'N/A';
    elements.summarySessions.textContent = `${summary.sessions} (${summary.sessions_ready} ready)`;
  }
  elements.updatedAt.textContent =
    `Updated ${new Date(generatedAt).toLocaleTimeString()} · ${state.snapshot.poll_interval_seconds}s polling`;
}

function renderTargets() {
  elements.targetHealth.innerHTML = state.snapshot.targets.map((target) => `
    <span class="target-pill ${target.healthy ? '' : 'error'}" title="${escapeHTML(target.error || target.url)}">
      ${escapeHTML(target.name)} · ${target.healthy ? target.pool_count : 'error'}
    </span>
  `).join('');
}

function renderTabs() {
  const clusters = [...new Set(state.snapshot.pools.map((pool) => pool.cluster))].sort();
  if (state.cluster !== 'all' && !clusters.includes(state.cluster)) {
    state.cluster = 'all';
  }
  elements.clusterTabs.innerHTML = ['all', ...clusters].map((cluster) => {
    const count = cluster === 'all'
      ? state.snapshot.pools.length
      : state.snapshot.pools.filter((pool) => pool.cluster === cluster).length;
    const label = cluster === 'all' ? `All clusters (${count})` : `${cluster} (${count})`;
    return `<button class="tab ${state.cluster === cluster ? 'active' : ''}" type="button"
      role="tab" aria-selected="${state.cluster === cluster}"
      data-cluster="${escapeHTML(cluster)}">${escapeHTML(label)}</button>`;
  }).join('');
}
function renderPools() {
  const pools = visiblePools();
  elements.poolRows.innerHTML = pools.map((pool) => {
    const total = pool.workers.busy + pool.workers.idle + pool.workers.pending;
    return `
      <tr class="${pool.key === state.selectedKey ? 'selected' : ''}" data-key="${escapeHTML(pool.key)}" tabindex="0">
        <td><span class="pool-name" title="${escapeHTML(pool.gpu_pool_id)}">${escapeHTML(pool.gpu_pool_id)}</span></td>
        <td>${escapeHTML(pool.cluster)}</td>
        <td>
          <div class="util-cell">
            <span>${percent(pool.utilization)}</span>
            <span class="util-track">
              <span class="util-fill" style="width:${pool.utilization == null ? 0 : pool.utilization * 100}%"></span>
            </span>
          </div>
        </td>
        <td>
          <span class="workers" title="${total} workers">
            <span class="busy-text">${pool.workers.busy}B</span>
            <span class="idle-text">${pool.workers.idle}I</span>
            <span class="pending-text">${pool.workers.pending}P</span>
          </span>
        </td>
        <td>${pool.replicas}</td>
        <td>${percent(pool.balance)}</td>
        <td>
          <span class="queue-value">
            <strong>${pool.queue_supported ? pool.queue_length : 'N/A'}</strong>
            <span>${pool.queue_supported ? duration(pool.queue_avg_ms) : 'unsupported'}</span>
          </span>
        </td>
        <td>${pool.queue_supported ? duration(pool.queue_max_ms || null) : 'N/A'}</td>
        <td>
          <span class="session-count">
            <strong>${pool.sessions.total}</strong>
            <span>${sessionStatus(pool.sessions)}</span>
          </span>
        </td>
        <td><span class="image-name" title="${escapeHTML(pool.image)}">${escapeHTML(pool.image)}</span></td>
      </tr>
    `;
  }).join('');
  elements.emptyState.hidden = pools.length !== 0;
}
function renderDetail() {
  const pool = selectedPool();
  if (!pool) {
    elements.detailPool.textContent = 'No pool selected';
    elements.detailCluster.textContent = '--';
    elements.detailUtilization.textContent = '--';
    elements.detailLifetime.textContent = '--';
    elements.detailBalance.textContent = '--';
    elements.detailSessions.textContent = '0';
    elements.detailSessionStatus.textContent = '--';
    elements.historyRange.textContent = '--';
    renderSessionPrefixes({});
    elements.detailBusy.textContent = '0';
    elements.detailIdle.textContent = '0';
    elements.detailPending.textContent = '0';
    elements.detailQueueLength.textContent = '--';
    elements.detailQueueCount.textContent = '--';
    elements.detailQueueAvg.textContent = '--';
    elements.detailQueueMax.textContent = '--';
    elements.detailQueueRate.textContent = '--';
    elements.workerTotal.textContent = '0 total';
    elements.busyBar.style.width = '0';
    elements.idleBar.style.width = '0';
    elements.pendingBar.style.width = '0';
    elements.detailImage.textContent = '--';
    drawChart([]);
    return;
  }
  elements.detailPool.textContent = pool.gpu_pool_id;
  elements.detailCluster.textContent = pool.cluster;
  elements.detailUtilization.textContent = percent(pool.utilization);
  elements.detailLifetime.textContent = percent(pool.lifetime_utilization);
  elements.detailBalance.textContent = percent(pool.balance);
  elements.detailSessions.textContent = pool.sessions.total;
  elements.detailSessionStatus.textContent = sessionStatus(pool.sessions);
  renderSessionPrefixes(pool.session_prefixes);
  elements.detailBusy.textContent = pool.workers.busy;
  elements.detailIdle.textContent = pool.workers.idle;
  elements.detailPending.textContent = pool.workers.pending;
  elements.detailQueueLength.textContent = pool.queue_supported ? pool.queue_length : 'N/A';
  elements.detailQueueCount.textContent = pool.queue_supported ? pool.queue_count : 'N/A';
  elements.detailQueueAvg.textContent = pool.queue_supported ? duration(pool.queue_avg_ms) : 'N/A';
  elements.detailQueueMax.textContent = pool.queue_supported ? duration(pool.queue_max_ms || null) : 'N/A';
  elements.detailQueueRate.textContent = pool.queue_supported
    ? `${rate(pool.queue_rate_per_minute)} / min`
    : 'unsupported';
  elements.detailImage.textContent = pool.image;
  if (pool.history?.length) {
    const first = new Date(pool.history[0].timestamp);
    const last = new Date(pool.history.at(-1).timestamp);
    elements.historyRange.textContent =
      `${first.toLocaleString()} - ${last.toLocaleString()} · ${pool.history.length} samples`;
  } else {
    elements.historyRange.textContent = 'No interval samples';
  }
  const total = pool.workers.busy + pool.workers.idle + pool.workers.pending;
  elements.workerTotal.textContent = `${total} total`;
  elements.busyBar.style.width = `${total ? pool.workers.busy * 100 / total : 0}%`;
  elements.idleBar.style.width = `${total ? pool.workers.idle * 100 / total : 0}%`;
  elements.pendingBar.style.width = `${total ? pool.workers.pending * 100 / total : 0}%`;
  drawChart(pool.history || []);
}
function drawChart(history) {
  const canvas = elements.historyChart;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const padding = { top: 12, right: 8, bottom: 24, left: 34 };
  context.clearRect(0, 0, width, height);
  context.font = '11px Inter, sans-serif';
  context.strokeStyle = '#dfe6e3';
  context.fillStyle = '#7a8784';
  context.lineWidth = 1;
  [0, 0.5, 1].forEach((value) => {
    const y = padding.top + (1 - value) * (height - padding.top - padding.bottom);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText(`${value * 100}%`, 2, y + 4);
  });
  if (history.length < 1) {
    return;
  }
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  if (history.length === 1) {
    const point = history[0];
    const x = padding.left + chartWidth / 2;
    const y = padding.top + (1 - point.utilization) * chartHeight;
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fillStyle = '#159a7c';
    context.fill();
  } else {
    context.beginPath();
    history.forEach((point, index) => {
      const x = padding.left + chartWidth * index / (history.length - 1);
      const y = padding.top + (1 - point.utilization) * chartHeight;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.strokeStyle = '#159a7c';
    context.lineWidth = 2;
    context.stroke();
    const gradient = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(21, 154, 124, 0.22)');
    gradient.addColorStop(1, 'rgba(21, 154, 124, 0.02)');
    context.lineTo(width - padding.right, height - padding.bottom);
    context.lineTo(padding.left, height - padding.bottom);
    context.closePath();
    context.fillStyle = gradient;
    context.fill();
  }
  const first = new Date(history[0].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const last = new Date(history.at(-1).timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  context.fillStyle = '#7a8784';
  context.fillText(first, padding.left, height - 6);
  const lastWidth = context.measureText(last).width;
  context.fillText(last, width - padding.right - lastWidth, height - 6);
}
function render() {
  renderTargets();
  renderTabs();
  ensureSelection();
  renderSummary();
  renderPools();
  renderDetail();
}

async function refresh() {
  elements.refreshButton.classList.add('loading');
  try {
    const response = await fetch('/api/v1/snapshot', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.snapshot = await response.json();
    render();
    const interval = Math.max(2000, state.snapshot.poll_interval_seconds * 1000);
    clearTimeout(state.timer);
    state.timer = setTimeout(refresh, interval);
  } catch (error) {
    elements.updatedAt.textContent = `Refresh failed · ${error.message}`;
    clearTimeout(state.timer);
    state.timer = setTimeout(refresh, 5000);
  } finally {
    elements.refreshButton.classList.remove('loading');
  }
}

elements.refreshButton.addEventListener('click', refresh);
elements.searchInput.addEventListener('input', (event) => {
  state.query = event.target.value;
  renderPools();
});
elements.clusterTabs.addEventListener('click', (event) => {
  const button = event.target.closest('[data-cluster]');
  if (!button) {
    return;
  }
  state.cluster = button.dataset.cluster;
  state.selectedKey = state.cluster === 'all'
    ? ''
    : state.snapshot.pools.find((pool) => pool.cluster === state.cluster)?.key || '';
  renderTabs();
  renderSummary();
  renderPools();
  renderDetail();
});
elements.poolRows.addEventListener('click', (event) => {
  const row = event.target.closest('[data-key]');
  if (!row) {
    return;
  }
  state.selectedKey = row.dataset.key;
  renderSummary();
  renderPools();
  renderDetail();
});
elements.poolRows.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' && event.key !== ' ') {
    return;
  }
  const row = event.target.closest('[data-key]');
  if (!row) {
    return;
  }
  event.preventDefault();
  state.selectedKey = row.dataset.key;
  renderSummary();
  renderPools();
  renderDetail();
});
window.addEventListener('resize', () => {
  if (state.snapshot) {
    renderDetail();
  }
});

refresh();




