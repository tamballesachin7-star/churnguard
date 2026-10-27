// ChurnGuard AI - Frontend Orchestration & Interactive Visualizations

let allCustomers = [];
let filteredCustomers = [];
let activeFilter = 'all';
let currentSort = { col: 'churn', asc: false };
let healthChartInstance = null;
let featuresChartInstance = null;
let trendChartInstance = null;
let activeModalCustomer = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchDashboardData();
  setupDragAndDrop();
  runLiveSimulation();
});

// Fetch current dataset and summary from backend
async function fetchDashboardData() {
  try {
    const res = await fetch('/api/data');
    if (!res.ok) throw new Error('Failed to load data');
    const data = await res.json();

    allCustomers = data.customers || [];
    updateSummaryKPIs(data.summary, data.mapping);
    updateCharts(data.summary);
    filterAndRender();
  } catch (err) {
    console.error('Error loading dashboard data:', err);
  }
}

// Update KPI cards and active dataset banner
function updateSummaryKPIs(summary, mapping) {
  if (!summary) return;

  document.getElementById('stat-total-accounts').innerText = summary.total_accounts.toLocaleString();
  document.getElementById('stat-healthy-pct').innerText = `${Math.round((summary.healthy_count / summary.total_accounts) * 100)}% healthy`;
  
  document.getElementById('stat-total-mrr').innerText = `$${Math.round(summary.total_mrr).toLocaleString()}`;
  document.getElementById('stat-total-arr').innerText = `$${Math.round(summary.total_mrr * 12).toLocaleString()}`;
  
  document.getElementById('stat-at-risk-mrr').innerText = `$${Math.round(summary.at_risk_mrr).toLocaleString()}`;
  document.getElementById('stat-risk-mrr-pct').innerText = `${summary.mrr_risk_pct}%`;
  
  document.getElementById('stat-critical-count').innerText = summary.critical_count;
  document.getElementById('stat-warning-count').innerText = `${summary.at_risk_count} Accounts`;

  document.getElementById('donut-crit-val').innerText = summary.critical_count;
  document.getElementById('donut-risk-val').innerText = summary.at_risk_count;
  document.getElementById('donut-healthy-val').innerText = summary.healthy_count;

  if (summary.model_accuracy) {
    document.getElementById('stat-model-acc').innerText = `${summary.model_accuracy}% (AUC: ${summary.model_auc})`;
  }

  if (mapping) {
    document.getElementById('schema-id-col').innerText = mapping.id_col || 'Account';
    document.getElementById('schema-target-col').innerText = mapping.target_col ? `${mapping.target_col} (Trained)` : 'Dynamic Risk Score';
    document.getElementById('schema-rev-col').innerText = mapping.revenue_col || 'MRR';
  }
}

// Render or update Chart.js visualizations
function updateCharts(summary) {
  // 1. Health Distribution Donut
  const donutCtx = document.getElementById('healthDonutChart').getContext('2d');
  if (healthChartInstance) healthChartInstance.destroy();

  healthChartInstance = new Chart(donutCtx, {
    type: 'doughnut',
    data: {
      labels: ['Critical Risk', 'At Risk', 'Healthy'],
      datasets: [{
        data: [summary.critical_count, summary.at_risk_count, summary.healthy_count],
        backgroundColor: ['#f43f5e', '#f59e0b', '#10b981'],
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      cutout: '72%'
    }
  });

  // 2. Feature Importance Horizontal Bar Chart
  const featCtx = document.getElementById('featuresBarChart').getContext('2d');
  if (featuresChartInstance) featuresChartInstance.destroy();

  const featLabels = summary.top_features && summary.top_features.length > 0 
    ? summary.top_features 
    : ['Seat Utilization Drop', 'Inactivity Days', 'Open Tickets', 'CSAT Decline', 'Renewal Horizon'];
  
  const featValues = [0.88, 0.76, 0.65, 0.54, 0.42, 0.35].slice(0, featLabels.length);

  featuresChartInstance = new Chart(featCtx, {
    type: 'bar',
    data: {
      labels: featLabels,
      datasets: [{
        label: 'Predictive Feature Weight',
        data: featValues,
        backgroundColor: '#6366f1',
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { size: 11, weight: '500' } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Filter accounts by search term & risk category
function setFilter(filterType) {
  activeFilter = filterType;
  document.querySelectorAll('.filter-tab').forEach(btn => {
    btn.classList.remove('bg-indigo-600', 'text-white');
    btn.classList.add('text-slate-400');
  });

  const activeBtn = document.getElementById(`filter-${filterType.toLowerCase().replace(' ', '-')}`);
  if (activeBtn) {
    activeBtn.classList.add('bg-indigo-600', 'text-white');
    activeBtn.classList.remove('text-slate-400');
  }

  filterAndRender();
}

function filterAccounts() {
  filterAndRender();
}

function filterAndRender() {
  const searchTerm = (document.getElementById('search-input').value || '').toLowerCase();

  filteredCustomers = allCustomers.filter(c => {
    const matchesSearch = c.name.toLowerCase().includes(searchTerm) || c.id.toLowerCase().includes(searchTerm);
    const matchesFilter = (activeFilter === 'all') || (c.risk_level.toLowerCase() === activeFilter.toLowerCase());
    return matchesSearch && matchesFilter;
  });

  // Sort
  filteredCustomers.sort((a, b) => {
    let valA = a.churn_probability;
    let valB = b.churn_probability;
    if (currentSort.col === 'mrr') {
      valA = a.mrr;
      valB = b.mrr;
    } else if (currentSort.col === 'health') {
      valA = a.health_score;
      valB = b.health_score;
    }
    return currentSort.asc ? (valA - valB) : (valB - valA);
  });

  renderTable();
}

function toggleSort(col) {
  if (currentSort.col === col) {
    currentSort.asc = !currentSort.asc;
  } else {
    currentSort.col = col;
    currentSort.asc = false;
  }
  document.getElementById('sort-mrr-icon').innerText = currentSort.col === 'mrr' ? (currentSort.asc ? '↑' : '↓') : '↕';
  document.getElementById('sort-health-icon').innerText = currentSort.col === 'health' ? (currentSort.asc ? '↑' : '↓') : '↕';
  document.getElementById('sort-churn-icon').innerText = currentSort.col === 'churn' ? (currentSort.asc ? '↑' : '↓') : '↕';
  filterAndRender();
}

// Render Table Rows
function renderTable() {
  const tbody = document.getElementById('accounts-table-body');
  tbody.innerHTML = '';

  document.getElementById('showing-count-text').innerText = `Showing ${filteredCustomers.length} of ${allCustomers.length} accounts`;

  if (filteredCustomers.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-10 text-slate-500">No matching accounts found.</td></tr>`;
    return;
  }

  filteredCustomers.forEach(c => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-800/40 transition group';

    // Risk badge styling
    let badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
    let progressBg = 'bg-emerald-500';
    if (c.risk_level === 'Critical') {
      badgeClass = 'bg-rose-500/15 text-rose-400 border-rose-500/30 critical-pulse';
      progressBg = 'bg-rose-500';
    } else if (c.risk_level === 'At Risk') {
      badgeClass = 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      progressBg = 'bg-amber-500';
    }

    const firstReason = c.risk_reasons && c.risk_reasons.length > 0 ? c.risk_reasons[0] : 'Telemetry normal';

    tr.innerHTML = `
      <td class="py-3.5 px-4 font-semibold text-white">
        <div class="flex items-center space-x-2.5">
          <div class="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-xs font-mono text-indigo-300">
            ${c.name.substring(0, 2).toUpperCase()}
          </div>
          <div>
            <div>${c.name}</div>
            <div class="text-[10px] text-slate-400 font-mono">ID: ${c.id}</div>
          </div>
        </div>
      </td>
      <td class="py-3.5 px-4 font-mono font-medium text-slate-200">
        $${Math.round(c.mrr).toLocaleString()}/mo
      </td>
      <td class="py-3.5 px-4">
        <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold font-mono ${badgeClass} border">
          ${c.health_score} / 100
        </span>
      </td>
      <td class="py-3.5 px-4">
        <div class="w-32">
          <div class="flex justify-between text-[11px] font-mono mb-1">
            <span class="${c.risk_level === 'Critical' ? 'text-rose-400 font-bold' : (c.risk_level === 'At Risk' ? 'text-amber-400' : 'text-slate-400')}">${c.churn_probability}%</span>
            <span class="text-[10px] text-slate-400 uppercase">${c.risk_level}</span>
          </div>
          <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div class="${progressBg} h-full rounded-full transition-all duration-500" style="width: ${c.churn_probability}%"></div>
          </div>
        </div>
      </td>
      <td class="py-3.5 px-4 max-w-xs truncate text-slate-300" title="${firstReason}">
        <span class="text-[11px] text-slate-300">${firstReason}</span>
      </td>
      <td class="py-3.5 px-4 text-right">
        <button onclick="openCustomerModal('${c.id}')" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-indigo-600 hover:text-white text-indigo-300 border border-slate-700 hover:border-indigo-500 transition">
          Diagnose &rarr;
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Deep Dive Modal / Drawer
function openCustomerModal(customerId) {
  const customer = allCustomers.find(c => String(c.id) === String(customerId));
  if (!customer) return;

  activeModalCustomer = customer;

  document.getElementById('modal-cust-name').innerText = customer.name;
  document.getElementById('modal-cust-mrr').innerText = `$${Math.round(customer.mrr).toLocaleString()}/mo`;
  document.getElementById('modal-cust-health').innerText = `${customer.health_score} / 100`;

  const badge = document.getElementById('modal-cust-risk-badge');
  badge.innerText = `${customer.risk_level} (${customer.churn_probability}% Churn Risk)`;
  badge.className = `text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider ${
    customer.risk_level === 'Critical' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
    customer.risk_level === 'At Risk' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
    'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
  }`;

  // Populate AI Reasons
  const reasonsDiv = document.getElementById('modal-cust-reasons');
  reasonsDiv.innerHTML = '';
  customer.risk_reasons.forEach(r => {
    const item = document.createElement('div');
    item.className = 'p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 text-xs text-slate-300 flex items-start space-x-2';
    item.innerHTML = `
      <span class="text-rose-400 mt-0.5">•</span>
      <span>${r}</span>
    `;
    reasonsDiv.appendChild(item);
  });

  // Populate Retention Playbooks
  const playbooksDiv = document.getElementById('modal-cust-playbooks');
  playbooksDiv.innerHTML = '';
  customer.playbooks.forEach((p, idx) => {
    const card = document.createElement('div');
    card.className = 'p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between text-xs gap-3';
    card.innerHTML = `
      <div>
        <div class="font-bold text-white flex items-center space-x-2">
          <span>${p.title}</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">${p.type}</span>
        </div>
        <p class="text-slate-400 text-[11px] mt-0.5">${p.action}</p>
      </div>
      <button onclick="triggerPlaybook('${customer.id}', '${p.title}', this)" class="shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition flex items-center space-x-1">
        <span>Execute</span>
      </button>
    `;
    playbooksDiv.appendChild(card);
  });

  // Populate Raw Metrics
  const rawDiv = document.getElementById('modal-raw-metrics');
  rawDiv.innerHTML = '';
  if (customer.raw_metrics) {
    for (const [k, v] of Object.entries(customer.raw_metrics)) {
      const field = document.createElement('div');
      field.className = 'p-2 rounded bg-slate-900 border border-slate-800 truncate';
      field.innerHTML = `<span class="text-slate-400 block text-[10px]">${k}:</span> <strong class="text-white">${v}</strong>`;
      rawDiv.appendChild(field);
    }
  }

  // Render 14-day velocity sparkline
  renderTrendChart(customer.sparkline || [40, 42, 38, 30, 25, 20, 15, 12, 10, 8]);

  document.getElementById('customer-modal').classList.remove('hidden');
}

function closeCustomerModal() {
  document.getElementById('customer-modal').classList.add('hidden');
}

function renderTrendChart(dataPoints) {
  const ctx = document.getElementById('customerTrendChart').getContext('2d');
  if (trendChartInstance) trendChartInstance.destroy();

  const labels = dataPoints.map((_, i) => `Day -${dataPoints.length - i}`);

  trendChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Daily Active Events',
        data: dataPoints,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: '#818cf8'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 9 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 9 } } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Trigger Retention Playbook
async function triggerPlaybook(customerId, actionTitle, buttonEl) {
  try {
    buttonEl.innerHTML = `<span>Dispatching...</span>`;
    buttonEl.disabled = true;

    const res = await fetch('/api/playbook/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: customerId, action_title: actionTitle })
    });

    const data = await res.json();
    if (data.success) {
      buttonEl.className = 'shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white flex items-center space-x-1';
      buttonEl.innerHTML = `<span>✓ Dispatched</span>`;
    }
  } catch (err) {
    console.error('Failed to trigger playbook:', err);
    buttonEl.disabled = false;
    buttonEl.innerText = 'Retry';
  }
}

// Upload Modal Handlers
function openUploadModal() {
  document.getElementById('upload-modal').classList.remove('hidden');
}

function closeUploadModal() {
  document.getElementById('upload-modal').classList.add('hidden');
  document.getElementById('upload-status').classList.add('hidden');
}

function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('border-indigo-500', 'bg-indigo-950/20');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-indigo-500', 'bg-indigo-950/20');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });
}

// Handle File Upload to /api/upload
async function handleFileSelected(file) {
  if (!file) return;

  const statusBox = document.getElementById('upload-status');
  const statusText = document.getElementById('upload-status-text');
  statusBox.classList.remove('hidden');
  statusText.innerText = `Ingesting & training model on ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();
    if (!res.ok) throw new Error(result.error || 'Failed to upload dataset');

    document.getElementById('active-dataset-name').innerText = file.name;
    statusText.innerText = `Success! Scored ${result.total_customers} accounts with AutoML.`;

    setTimeout(() => {
      closeUploadModal();
      fetchDashboardData();
    }, 800);
  } catch (err) {
    statusText.innerText = `Error: ${err.message}`;
    console.error(err);
  }
}

// Sample Switcher
async function loadSampleDataset(sampleKey) {
  const b2bBtn = document.getElementById('sample-b2b-btn');
  const telcoBtn = document.getElementById('sample-telco-btn');

  if (sampleKey === 'b2b_saas') {
    b2bBtn.className = 'px-3 py-1.5 rounded-md font-medium bg-indigo-600 text-white shadow';
    telcoBtn.className = 'px-3 py-1.5 rounded-md font-medium text-slate-400 hover:text-white transition-all';
    document.getElementById('active-dataset-name').innerText = 'B2B SaaS Cloud Subscriptions';
  } else {
    telcoBtn.className = 'px-3 py-1.5 rounded-md font-medium bg-indigo-600 text-white shadow';
    b2bBtn.className = 'px-3 py-1.5 rounded-md font-medium text-slate-400 hover:text-white transition-all';
    document.getElementById('active-dataset-name').innerText = 'Telco & Consumer Subscriptions';
  }

  try {
    const res = await fetch('/api/load-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample: sampleKey })
    });
    if (res.ok) {
      fetchDashboardData();
    }
  } catch (err) {
    console.error('Error switching sample:', err);
  }
}

// Live "What-If" Simulator
function openSimulatorModal() {
  document.getElementById('simulator-modal').classList.remove('hidden');
  runLiveSimulation();
}

function closeSimulatorModal() {
  document.getElementById('simulator-modal').classList.add('hidden');
}

async function runLiveSimulation() {
  const seatUtil = parseFloat(document.getElementById('sim-seat').value);
  const daysLogin = parseFloat(document.getElementById('sim-login').value);
  const decay = parseFloat(document.getElementById('sim-decay').value);
  const renewal = parseFloat(document.getElementById('sim-renewal').value);
  const tickets = parseFloat(document.getElementById('sim-tickets').value);

  // Update slider label texts
  document.getElementById('val-sim-seat').innerText = `${seatUtil}%`;
  document.getElementById('val-sim-login').innerText = `${daysLogin} days`;
  document.getElementById('val-sim-decay').innerText = `${decay > 0 ? '+' : ''}${decay}%`;
  document.getElementById('val-sim-renewal').innerText = `${renewal} days`;
  document.getElementById('val-sim-tickets').innerText = `${tickets} ticket${tickets === 1 ? '' : 's'}`;

  try {
    const res = await fetch('/api/simulator/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        seat_utilization: seatUtil,
        days_since_last_login: daysLogin,
        activity_decay_30d: decay,
        contract_days_left: renewal,
        open_support_tickets: tickets,
        csat_score: tickets >= 3 ? 2 : 4,
        exec_sponsor_engaged: daysLogin > 15 ? 0 : 1
      })
    });

    const pred = await res.json();
    document.getElementById('sim-prob-output').innerText = `${pred.churn_probability}%`;
    document.getElementById('sim-health-output').innerText = `${pred.health_score} / 100`;

    const badge = document.getElementById('sim-risk-badge');
    badge.innerText = pred.risk_level;
    if (pred.risk_level === 'Critical') {
      badge.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30';
      document.getElementById('sim-prob-output').className = 'text-3xl font-extrabold text-rose-400 mt-1';
      document.getElementById('sim-health-output').className = 'text-3xl font-extrabold text-rose-400 mt-1';
    } else if (pred.risk_level === 'At Risk') {
      badge.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30';
      document.getElementById('sim-prob-output').className = 'text-3xl font-extrabold text-amber-400 mt-1';
      document.getElementById('sim-health-output').className = 'text-3xl font-extrabold text-amber-400 mt-1';
    } else {
      badge.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
      document.getElementById('sim-prob-output').className = 'text-3xl font-extrabold text-white mt-1';
      document.getElementById('sim-health-output').className = 'text-3xl font-extrabold text-emerald-400 mt-1';
    }
  } catch (err) {
    console.error('Simulation error:', err);
  }
}
