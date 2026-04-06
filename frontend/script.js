// FraudSense AI - Enterprise Risk Platform
// Security Operations Center Dashboard

const API_BASE = 'http://localhost:8000';

// Feature name mapping for SHAP/Top Factors display
const FEATURE_NAME_MAP = {
    "V14": "Beneficiary Trust Score",
    "V4": "IP Geolocation Mismatch",
    "V12": "Device Fingerprint Risk",
    "Amount": "Transaction Amount"
};

// State
let currentPrediction = null;
let analyticsInterval = null;
let transactionHistory = [];

// DOM Elements
const elements = {
    // Header
    systemTime: document.getElementById('systemTime'),
    modelVersion: document.getElementById('modelVersion'),
    driftIndicator: document.getElementById('driftIndicator'),
    
    // Model Health
    modelStatus: document.getElementById('modelStatus'),
    rocAuc: document.getElementById('rocAuc'),
    precision: document.getElementById('precision'),
    recall: document.getElementById('recall'),
    threshold: document.getElementById('threshold'),
    
    // Risk Gauge
    gaugeFill: document.getElementById('gaugeFill'),
    gaugeNeedle: document.getElementById('gaugeNeedle'),
    riskScoreValue: document.getElementById('riskScoreValue'),
    riskBand: document.getElementById('riskBand'),
    
    // Live Stats
    totalTransactions: document.getElementById('totalTransactions'),
    fraudRate: document.getElementById('fraudRate'),
    highRiskCount: document.getElementById('highRiskCount'),
    avgProbability: document.getElementById('avgProbability'),
    
    // Latency
    latencyValue: document.getElementById('latencyValue'),
    latencyFill: document.getElementById('latencyFill'),
    minLatency: document.getElementById('minLatency'),
    maxLatency: document.getElementById('maxLatency'),
    
    // Transaction Input
    amountInput: document.getElementById('amountInput'),
    timeInput: document.getElementById('timeInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    simulateBtn: document.getElementById('simulateBtn'),
    clearBtn: document.getElementById('clearBtn'),
    
    // Results
    transactionId: document.getElementById('transactionId'),
    fraudProbBar: document.getElementById('fraudProbBar'),
    anomalyBar: document.getElementById('anomalyBar'),
    riskBar: document.getElementById('riskBar'),
    fraudProbability: document.getElementById('fraudProbability'),
    anomalyScore: document.getElementById('anomalyScore'),
    ensembleRisk: document.getElementById('ensembleRisk'),
    riskFactors: document.getElementById('riskFactors'),
    actionBox: document.getElementById('actionBox'),
    actionIcon: document.getElementById('actionIcon'),
    actionText: document.getElementById('actionText'),
    actionReasoning: document.getElementById('actionReasoning'),
    explanation: document.getElementById('explanation'),
    actionBadge: document.getElementById('actionBadge'),
    
    // Distribution
    distLow: document.getElementById('distLow'),
    distMedium: document.getElementById('distMedium'),
    distHigh: document.getElementById('distHigh'),
    distCritical: document.getElementById('distCritical'),
    distLowVal: document.getElementById('distLowVal'),
    distMediumVal: document.getElementById('distMediumVal'),
    distHighVal: document.getElementById('distHighVal'),
    distCriticalVal: document.getElementById('distCriticalVal'),
    
    // Action Distribution
    blockedCount: document.getElementById('blockedCount'),
    reviewCount: document.getElementById('reviewCount'),
    approvedCount: document.getElementById('approvedCount'),
    
    // Audit Modal
    auditModal: document.getElementById('auditModal'),
    auditLogBtn: document.getElementById('auditLogBtn'),
    closeAuditModal: document.getElementById('closeAuditModal'),
    auditTableBody: document.getElementById('auditTableBody'),
    auditSearch: document.getElementById('auditSearch'),
    
    // History
    historyTableBody: document.getElementById('historyTableBody'),
    historySearch: document.getElementById('historySearch'),
    historyFilter: document.getElementById('historyFilter'),
    
    // Settings
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    autoRefresh: document.getElementById('autoRefresh'),
    refreshInterval: document.getElementById('refreshInterval'),
    apiEndpoint: document.getElementById('apiEndpoint')
};

// Clock
function initClock() {
    updateTime();
    setInterval(updateTime, 1000);
}

function updateTime() {
    if (elements.systemTime) {
        const now = new Date();
        elements.systemTime.textContent = now.toLocaleTimeString('en-US', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// API Functions
async function fetchAPI(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
}

// Load Dashboard Data
async function loadDashboardData() {
    try {
        const modelInfo = await fetchAPI('/model-info');
        updateModelInfo(modelInfo);
        
        const driftStatus = await fetchAPI('/drift-status');
        updateDriftStatus(driftStatus);
        
        const analytics = await fetchAPI('/analytics/realtime');
        updateLiveStats(analytics);
        
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Update Model Info
function updateModelInfo(info) {
    if (elements.rocAuc) elements.rocAuc.textContent = info.metrics.ROC_AUC.toFixed(3);
    if (elements.precision) elements.precision.textContent = info.metrics.precision.toFixed(3);
    if (elements.recall) elements.recall.textContent = info.metrics.recall.toFixed(3);
    if (elements.threshold) elements.threshold.textContent = info.threshold.toFixed(2);
    if (elements.modelVersion) {
        const versionEl = elements.modelVersion.querySelector('.version-number');
        if (versionEl) versionEl.textContent = 'v' + info.version;
    }
}

// Update Drift Status
function updateDriftStatus(status) {
    if (elements.driftIndicator) {
        elements.driftIndicator.className = 'drift-indicator ' + status.status_color;
    }
}

// Update Live Stats
function updateLiveStats(analytics) {
    if (!analytics.summary) return;
    
    const summary = analytics.summary;
    
    if (elements.totalTransactions) {
        animateCounter(elements.totalTransactions, summary.total_transactions);
        animateStatValue(elements.totalTransactions);
    }
    if (elements.fraudRate) {
        elements.fraudRate.textContent = summary.fraud_rate_percentage.toFixed(2) + '%';
    }
    if (elements.highRiskCount) {
        elements.highRiskCount.textContent = (summary.risk_distribution.High || 0) + (summary.risk_distribution.Critical || 0);
        animateStatValue(elements.highRiskCount);
    }
    if (elements.avgProbability) {
        elements.avgProbability.textContent = summary.averages.fraud_probability.toFixed(3);
    }
    
    // Update distribution bars
    const total = summary.total_transactions || 1;
    if (elements.distLow) elements.distLow.style.height = ((summary.risk_distribution.Low || 0) / total * 100) + '%';
    if (elements.distMedium) elements.distMedium.style.height = ((summary.risk_distribution.Medium || 0) / total * 100) + '%';
    if (elements.distHigh) elements.distHigh.style.height = ((summary.risk_distribution.High || 0) / total * 100) + '%';
    if (elements.distCritical) elements.distCritical.style.height = ((summary.risk_distribution.Critical || 0) / total * 100) + '%';
    
    if (elements.distLowVal) elements.distLowVal.textContent = summary.risk_distribution.Low || 0;
    if (elements.distMediumVal) elements.distMediumVal.textContent = summary.risk_distribution.Medium || 0;
    if (elements.distHighVal) elements.distHighVal.textContent = summary.risk_distribution.High || 0;
    if (elements.distCriticalVal) elements.distCriticalVal.textContent = summary.risk_distribution.Critical || 0;
    
    // Update action distribution
    if (elements.blockedCount) elements.blockedCount.textContent = summary.actions.blocked || 0;
    if (elements.reviewCount) elements.reviewCount.textContent = summary.actions.review_required || 0;
    if (elements.approvedCount) elements.approvedCount.textContent = summary.actions.approved || 0;
}

// Animate Counter
function animateCounter(element, target) {
    const current = parseInt(element.textContent) || 0;
    const diff = target - current;
    const steps = 20;
    const stepValue = diff / steps;
    let step = 0;
    
    const interval = setInterval(() => {
        step++;
        element.textContent = Math.round(current + (stepValue * step));
        
        if (step >= steps) {
            element.textContent = target;
            clearInterval(interval);
        }
    }, 30);
}

// Analytics Polling
function startAnalyticsPolling() {
    const interval = parseInt(elements.refreshInterval?.value || 10) * 1000;
    analyticsInterval = setInterval(refreshAnalytics, interval);
}

function stopAnalyticsPolling() {
    if (analyticsInterval) {
        clearInterval(analyticsInterval);
        analyticsInterval = null;
    }
}

async function refreshAnalytics() {
    try {
        const analytics = await fetchAPI('/analytics/realtime');
        updateLiveStats(analytics);
        
        // Update charts with real-time data
        if (analytics.summary) {
            const fraudRate = analytics.summary.fraud_rate_percentage || 0;
            const transactionCount = analytics.summary.total_transactions || 0;
            updateChartData(fraudRate / 100, transactionCount);
        }
    } catch (error) {
        console.error('Error refreshing analytics:', error);
    }
}

// ===== ANALYSIS PAGE FUNCTIONS =====

function initAnalysisEvents() {
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', analyzeTransaction);
    }
    if (elements.simulateBtn) {
        elements.simulateBtn.addEventListener('click', simulateTransaction);
    }
    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', clearForm);
    }
}

// Analyze Transaction
async function analyzeTransaction() {
    const features = collectFeatures();
    
    if (!features) return;
    
    setLoading(elements.analyzeBtn, true);
    
    try {
        const startTime = performance.now();
        console.log('Sending features to API:', features);
        const result = await fetchAPI('/predict', {
            method: 'POST',
            body: JSON.stringify(features)
        });
        console.log('API Response:', result);
        const endTime = performance.now();
        
        const latency = Math.round(endTime - startTime);
        displayResult(result, latency);
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error analyzing transaction: ' + error.message);
    } finally {
        setLoading(elements.analyzeBtn, false);
    }
}

// Simulate Transaction
async function simulateTransaction() {
    setLoading(elements.simulateBtn, true);
    
    try {
        const startTime = performance.now();
        console.log('Calling simulate endpoint...');
        const result = await fetchAPI('/simulate');
        console.log('Simulate Response:', result);
        const endTime = performance.now();
        
        const latency = Math.round(endTime - startTime);
        displayResult(result, latency);
        
    } catch (error) {
        console.error('Simulation error:', error);
        alert('Error simulating transaction: ' + error.message);
    } finally {
        setLoading(elements.simulateBtn, false);
    }
}

// Collect Features from Form
function collectFeatures() {
    const features = {};
    const vFields = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
                   'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
                   'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28'];
    
    let hasVFeatures = false;
    for (const field of vFields) {
        const input = document.getElementById(field);
        if (input && input.value) {
            features[field.toUpperCase()] = parseFloat(input.value);
            hasVFeatures = true;
        } else {
            features[field.toUpperCase()] = 0.0;
        }
    }
    
    const amount = parseFloat(elements.amountInput?.value) || 0;
    const time = parseFloat(elements.timeInput?.value) || 0;
    
    features['Amount'] = amount;
    features['Time'] = time;
    
    if (!hasVFeatures && amount === 0 && time === 0) {
        alert('Please enter transaction features');
        return null;
    }
    
    return features;
}

// Display Result
function displayResult(result, latency) {
    console.log('displayResult called with:', result);
    currentPrediction = result;
    
    if (elements.transactionId) {
        const txId = result.transaction_id || result.transactionId || 'N/A';
        console.log('Setting transaction ID:', txId);
        elements.transactionId.textContent = txId;
    }
    
    if (result.final_risk_score !== undefined) {
        updateRiskGauge(result.final_risk_score, result.risk_band);
    }
    
    if (elements.fraudProbBar) {
        const fraudProb = result.fraud_probability ?? result.fraud_prob ?? 0;
        console.log('Fraud probability:', fraudProb);
        updateResultBar(elements.fraudProbBar, elements.fraudProbability, fraudProb);
    }
    if (elements.anomalyBar) {
        const anomalyScore = result.anomaly_score ?? 0;
        console.log('Anomaly score:', anomalyScore);
        updateResultBar(elements.anomalyBar, elements.anomalyScore, anomalyScore);
    }
    if (elements.riskBar) {
        const riskScore = result.final_risk_score ?? 0;
        console.log('Risk score:', riskScore);
        updateResultBar(elements.riskBar, elements.ensembleRisk, riskScore);
    }
    
    if (elements.latencyValue) {
        elements.latencyValue.textContent = latency;
        const percent = Math.min((latency / 200) * 100, 100);
        if (elements.latencyFill) elements.latencyFill.style.width = `${percent}%`;
        
        const currentMin = parseInt(elements.minLatency?.textContent) || Infinity;
        const currentMax = parseInt(elements.maxLatency?.textContent) || 0;
        
        if (latency < currentMin && elements.minLatency) elements.minLatency.textContent = latency;
        if (latency > currentMax && elements.maxLatency) elements.maxLatency.textContent = latency;
    }
    
    if (elements.riskFactors) updateRiskFactors(result.top_factors);
    if (elements.actionText) {
        updateActionRecommendation(result.action_recommendation, result.action_reasoning);
        playActionSound(result.action_recommendation);
    }
    if (elements.explanation) elements.explanation.textContent = result.explanation_summary;
    
    // Add to history
    addToHistory(result);
    
    // Refresh history page if open
    loadHistoryData();
    
    refreshAnalytics();
}

// Update Risk Gauge
function updateRiskGauge(score, band) {
    if (elements.riskScoreValue) elements.riskScoreValue.textContent = score.toFixed(2);
    
    if (elements.gaugeFill) {
        const offset = 173 - (score * 173);
        elements.gaugeFill.style.strokeDashoffset = offset;
    }
    
    if (elements.gaugeNeedle) {
        const rotation = -90 + (score * 180);
        elements.gaugeNeedle.style.transform = `rotate(${rotation}deg)`;
    }
    
    if (elements.riskBand) {
        elements.riskBand.textContent = band.toUpperCase();
        elements.riskBand.className = 'gauge-band ' + band.toLowerCase();
    }
}

// Update Result Bar
function updateResultBar(bar, valueEl, value) {
    bar.style.width = `${value * 100}%`;
    valueEl.textContent = value.toFixed(3);
    
    if (value < 0.2) {
        valueEl.style.color = 'var(--accent-success)';
    } else if (value < 0.5) {
        valueEl.style.color = 'var(--accent-warning)';
    } else {
        valueEl.style.color = 'var(--accent-danger)';
    }
}

// Update Risk Factors
function updateRiskFactors(factors) {
    if (!factors || factors.length === 0) {
        elements.riskFactors.innerHTML = '<div class="factor-item">No significant risk factors</div>';
        return;
    }
    
    elements.riskFactors.innerHTML = factors.map(factor => {
        const displayName = FEATURE_NAME_MAP[factor.feature] || factor.feature;
        return `
        <div class="factor-item">
            <span>${displayName}</span>
            <span class="factor-impact ${factor.impact > 0 ? 'positive' : 'negative'}">
                ${factor.impact > 0 ? '+' : ''}${factor.impact.toFixed(4)}
            </span>
        </div>
        `;
    }).join('');
}

// Update Action Recommendation
function updateActionRecommendation(action, reasoning) {
    elements.actionText.textContent = action;
    elements.actionReasoning.textContent = reasoning;
    elements.actionBadge.textContent = action;
    
    elements.actionBox.className = 'action-box';
    elements.actionIcon.className = 'action-icon';
    elements.actionIcon.innerHTML = '';
    
    if (action.includes('Block')) {
        elements.actionBox.classList.add('block');
        elements.actionIcon.classList.add('blocked');
        elements.actionIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>';
    } else if (action.includes('Review')) {
        elements.actionBox.classList.add('review');
        elements.actionIcon.classList.add('review');
        elements.actionIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
    } else {
        elements.actionBox.classList.add('approve');
        elements.actionIcon.classList.add('approved');
        elements.actionIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    }
}

// Clear Form
function clearForm() {
    for (let i = 1; i <= 28; i++) {
        const input = document.getElementById('v' + i);
        if (input) input.value = '';
    }
    
    if (elements.amountInput) elements.amountInput.value = '';
    if (elements.timeInput) elements.timeInput.value = '';
    
    if (elements.transactionId) elements.transactionId.textContent = '--';
    if (elements.fraudProbBar) elements.fraudProbBar.style.width = '0%';
    if (elements.anomalyBar) elements.anomalyBar.style.width = '0%';
    if (elements.riskBar) elements.riskBar.style.width = '0%';
    if (elements.fraudProbability) elements.fraudProbability.textContent = '0.000';
    if (elements.anomalyScore) elements.anomalyScore.textContent = '0.000';
    if (elements.ensembleRisk) elements.ensembleRisk.textContent = '0.000';
    if (elements.riskFactors) elements.riskFactors.innerHTML = '<div class="factor-item">--</div>';
    if (elements.actionText) elements.actionText.textContent = 'Ready to analyze';
    if (elements.actionReasoning) elements.actionReasoning.textContent = 'Enter transaction data or simulate a live transaction';
    if (elements.actionBox) {
        elements.actionBox.className = 'action-box';
    }
    if (elements.actionIcon) {
        elements.actionIcon.className = 'action-icon approved';
        elements.actionIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    }
    if (elements.actionBadge) elements.actionBadge.textContent = 'READY';
    if (elements.explanation) elements.explanation.textContent = 'The analysis will provide a human-readable explanation of the risk factors.';
    
    updateRiskGauge(0, 'Low');
    
    currentPrediction = null;
}

// Button Loading
function setLoading(button, loading) {
    if (!button) return;
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = `<svg class="btn-spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
            <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round">
                <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.8s" repeatCount="indefinite"/>
            </path>
        </svg> Loading...`;
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalText;
    }
}

// ===== HISTORY PAGE FUNCTIONS =====

function initHistoryEvents() {
    if (elements.historySearch) {
        elements.historySearch.addEventListener('input', updateHistoryTable);
    }
    if (elements.historyFilter) {
        elements.historyFilter.addEventListener('change', updateHistoryTable);
    }
}

async function loadHistoryData() {
    console.log('Loading history data...');
    try {
        // Load from audit-log endpoint for persistent transaction history
        const logs = await fetchAPI('/audit-log?limit=100');
        console.log('Audit logs response:', logs);
        if (logs && logs.logs) {
            transactionHistory = logs.logs.map(t => ({
                id: t.transaction_id,
                time: new Date(t.timestamp).toLocaleString(),
                fraudProb: t.fraud_probability,
                riskLevel: t.risk_level,
                action: t.action_recommendation
            }));
            console.log('Transaction history loaded:', transactionHistory.length, 'transactions');
            updateHistoryTable();
        }
    } catch (error) {
        console.error('Error loading history:', error);
        // Fallback: try analytics/realtime if audit-log fails
        try {
            console.log('Trying fallback to analytics/realtime...');
            const analytics = await fetchAPI('/analytics/realtime');
            console.log('Analytics response:', analytics);
            if (analytics.recent_transactions) {
                transactionHistory = analytics.recent_transactions.map(t => ({
                    id: t.transaction_id,
                    time: new Date(t.timestamp * 1000).toLocaleTimeString(),
                    fraudProb: t.fraud_probability,
                    riskLevel: t.risk_band,
                    action: t.action_recommendation
                }));
                updateHistoryTable();
            }
        } catch (fallbackError) {
            console.error('Error loading fallback history:', fallbackError);
        }
    }
}

function addToHistory(result) {
    const entry = {
        id: result.transaction_id,
        time: new Date().toLocaleTimeString(),
        fraudProb: result.fraud_probability,
        riskLevel: result.risk_band,
        action: result.action_recommendation
    };
    
    transactionHistory.unshift(entry);
    
    if (transactionHistory.length > 100) {
        transactionHistory.pop();
    }
    
    updateHistoryTable();
}

function updateHistoryTable() {
    if (!elements.historyTableBody) return;
    
    const searchTerm = elements.historySearch?.value?.toLowerCase() || '';
    const filter = elements.historyFilter?.value || 'all';
    
    let filtered = transactionHistory.filter(item => {
        const matchesSearch = !searchTerm || item.id.toLowerCase().includes(searchTerm);
        const matchesFilter = filter === 'all' || 
            (filter === 'blocked' && item.action.includes('Block')) ||
            (filter === 'review' && item.action.includes('Review')) ||
            (filter === 'approved' && item.action.includes('Approve'));
        return matchesSearch && matchesFilter;
    });
    
    if (filtered.length === 0) {
        elements.historyTableBody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: #64748b; padding: 40px;">No transactions found</td>
            </tr>
        `;
        return;
    }
    
    elements.historyTableBody.innerHTML = filtered.map(item => `
        <tr>
            <td>${item.time}</td>
            <td>${item.id}</td>
            <td>${item.fraudProb?.toFixed(3) || '0.000'}</td>
            <td>${item.riskLevel || 'Low'}</td>
            <td>${item.action || 'Approved'}</td>
        </tr>
    `).join('');
}

// ===== SETTINGS PAGE FUNCTIONS =====

function initSettingsEvents() {
    if (elements.thresholdSlider) {
        elements.thresholdSlider.addEventListener('input', (e) => {
            if (elements.thresholdValue) {
                elements.thresholdValue.textContent = (e.target.value / 100).toFixed(2);
            }
        });
    }
    
    if (elements.autoRefresh) {
        elements.autoRefresh.addEventListener('change', (e) => {
            if (e.target.checked) {
                startAnalyticsPolling();
            } else {
                stopAnalyticsPolling();
            }
        });
    }
}

// ===== AUDIT MODAL FUNCTIONS =====

function initAuditModal() {
    if (elements.auditLogBtn) {
        elements.auditLogBtn.addEventListener('click', openAuditModal);
    }
    if (elements.closeAuditModal) {
        elements.closeAuditModal.addEventListener('click', closeAuditModal);
    }
    if (elements.auditModal) {
        elements.auditModal.addEventListener('click', (e) => {
            if (e.target === elements.auditModal) closeAuditModal();
        });
    }
    if (elements.auditSearch) {
        elements.auditSearch.addEventListener('input', filterAuditLogs);
    }
}

function openAuditModal() {
    if (elements.auditModal) {
        elements.auditModal.classList.add('active');
        loadAuditLogs();
    }
}

function closeAuditModal() {
    if (elements.auditModal) {
        elements.auditModal.classList.remove('active');
    }
}

async function loadAuditLogs() {
    try {
        const logs = await fetchAPI('/audit-log');
        renderAuditLogs(logs);
    } catch (error) {
        console.error('Error loading audit logs:', error);
    }
}

function renderAuditLogs(logs) {
    if (!elements.auditTableBody) return;
    
    // Handle response format from /audit-log endpoint
    const logEntries = logs?.logs || logs || [];
    
    if (!logEntries || logEntries.length === 0) {
        elements.auditTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No logs found</td></tr>';
        return;
    }
    
    elements.auditTableBody.innerHTML = logEntries.map(log => `
        <tr>
            <td>${new Date(log.timestamp).toLocaleString()}</td>
            <td>${log.transaction_id}</td>
            <td>${log.fraud_probability?.toFixed(3) || '0.000'}</td>
            <td>${log.risk_level || 'Low'}</td>
            <td>${log.action_recommendation || 'Approved'}</td>
        </tr>
    `).join('');
}

function filterAuditLogs() {
    loadAuditLogs();
}

// ===== SIDEBAR =====
function initSidebar() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const appContainer = document.getElementById('appContainer');
    
    if (!sidebar || !sidebarToggle || !appContainer) return;
    
    // Load saved sidebar state
    const isCollapsed = localStorage.getItem('sidebar_collapsed');
    if (isCollapsed === 'true') {
        sidebar.classList.add('collapsed');
        appContainer.classList.add('sidebar-collapsed');
    }
    
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        appContainer.classList.toggle('sidebar-collapsed');
        
        const collapsed = sidebar.classList.contains('collapsed');
        localStorage.setItem('sidebar_collapsed', collapsed);
    });
}



// ===== INTERACTIVE FEATURES =====

// Real-time Chart System
class RealtimeChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.maxPoints = options.maxPoints || 60;
        this.data = [];
        this.animationId = null;
        this.options = options;
        this.init();
    }

    init() {
        if (!this.container) return;
        
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'chart-canvas';
        this.container.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        setTimeout(() => this.resize(), 100);
        window.addEventListener('resize', () => this.resize());
        
        this.startAnimation();
        
        if (this.data.length === 0) {
            this.addDemoData();
        }
    }
    
    addDemoData() {
        const isFraudChart = this.container.id === 'fraudChartContainer';
        for (let i = 0; i < 15; i++) {
            const value = isFraudChart 
                ? Math.random() * 15 + 5 
                : Math.random() * 500 + 100;
            this.addPoint(value);
        }
    }

    resize() {
        if (!this.container) return;
        this.canvas.width = this.container.offsetWidth || this.container.getBoundingClientRect().width;
        this.canvas.height = this.container.offsetHeight || this.container.getBoundingClientRect().height;
    }

    addPoint(value) {
        const now = Date.now();
        this.data.push({ value, time: now });
        
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }
    }

    startAnimation() {
        const animate = () => {
            this.draw();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }

    draw() {
        if (!this.ctx || this.data.length < 2) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = 30;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        this.ctx.strokeStyle = 'rgba(45, 58, 79, 0.5)';
        this.ctx.lineWidth = 1;
        
        for (let i = 0; i <= 4; i++) {
            const y = padding + (height - 2 * padding) * (i / 4);
            this.ctx.beginPath();
            this.ctx.moveTo(padding, y);
            this.ctx.lineTo(width - padding, y);
            this.ctx.stroke();
        }
        
        // Calculate scale
        const values = this.data.map(d => d.value);
        const maxVal = Math.max(...values, 1);
        const minVal = Math.min(...values, 0);
        const range = maxVal - minVal || 1;
        
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Draw line
        this.ctx.beginPath();
        this.ctx.strokeStyle = this.options.color || '#f59e0b';
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        this.data.forEach((point, i) => {
            const x = padding + (i / (this.data.length - 1)) * chartWidth;
            const y = padding + chartHeight - ((point.value - minVal) / range) * chartHeight;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });
        
        this.ctx.stroke();
        
        // Draw glow
        this.ctx.shadowColor = this.options.color || '#f59e0b';
        this.ctx.shadowBlur = 10;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
        
        // Draw points
        this.data.forEach((point, i) => {
            const x = padding + (i / (this.data.length - 1)) * chartWidth;
            const y = padding + chartHeight - ((point.value - minVal) / range) * chartHeight;
            
            this.ctx.beginPath();
            this.ctx.fillStyle = this.options.color || '#f59e0b';
            this.ctx.arc(x, y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }
}

// Custom Cursor
class CustomCursor {
    constructor() {
        this.enabled = localStorage.getItem('cursor_enabled') === 'true';
        this.element = null;
        
        if (this.enabled) {
            this.init();
        }
    }

    init() {
        this.element = document.createElement('div');
        this.element.className = 'custom-cursor';
        this.element.innerHTML = `
            <div class="custom-cursor-dot"></div>
            <div class="custom-cursor-ring"></div>
        `;
        document.body.appendChild(this.element);
        
        document.addEventListener('mousemove', (e) => this.move(e));
        
        // Add hover effect for interactive elements
        const interactiveElements = document.querySelectorAll('a, button, .card, .draggable');
        interactiveElements.forEach(el => {
            el.addEventListener('mouseenter', () => this.element.classList.add('hovering'));
            el.addEventListener('mouseleave', () => this.element.classList.remove('hovering'));
        });
    }

    move(e) {
        if (!this.element) return;
        this.element.style.left = e.clientX + 'px';
        this.element.style.top = e.clientY + 'px';
    }

    enable() {
        this.enabled = true;
        localStorage.setItem('cursor_enabled', 'true');
        document.body.classList.add('cursor-custom');
        this.init();
    }

    disable() {
        this.enabled = false;
        localStorage.setItem('cursor_enabled', 'false');
        document.body.classList.remove('cursor-custom');
        if (this.element) {
            this.element.remove();
            this.element = null;
        }
    }
}

// Sound Effects System
class SoundEffects {
    constructor() {
        this.enabled = localStorage.getItem('sound_enabled') === 'true';
        this.audioContext = null;
        this.sounds = {};
        
        if (this.enabled) {
            this.init();
        }
    }

    init() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn('Audio context not supported');
        }
    }

    playTone(frequency, duration, type = 'sine') {
        if (!this.audioContext) return;
        
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.type = type;
        oscillator.frequency.value = frequency;
        
        gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.start();
        oscillator.stop(this.audioContext.currentTime + duration);
    }

    playClick() {
        this.playTone(800, 0.1, 'sine');
    }

    playSuccess() {
        this.playTone(523, 0.1, 'sine');
        setTimeout(() => this.playTone(659, 0.1, 'sine'), 100);
        setTimeout(() => this.playTone(784, 0.15, 'sine'), 200);
    }

    playWarning() {
        this.playTone(440, 0.15, 'square');
        setTimeout(() => this.playTone(440, 0.15, 'square'), 150);
    }

    playBlock() {
        this.playTone(200, 0.3, 'sawtooth');
    }

    enable() {
        this.enabled = true;
        localStorage.setItem('sound_enabled', 'true');
        this.init();
    }

    disable() {
        this.enabled = false;
        localStorage.setItem('sound_enabled', 'false');
    }
}

// Initialize interactive features
let customCursor = null;
let soundEffects = null;
let fraudChart = null;
let transactionsChart = null;

function initInteractiveFeatures() {
    // Custom Cursor
    customCursor = new CustomCursor();
    
    // Sound Effects
    soundEffects = new SoundEffects();
    
    // Create sound toggle button
    createSoundToggle();
    
    // Add fraud pattern 3D visualization
    add3DVisualization();
    
    // Initialize charts
    setTimeout(() => {
        const fraudChartContainer = document.getElementById('fraudChartContainer');
        const transactionsChartContainer = document.getElementById('transactionsChartContainer');
        
        if (fraudChartContainer) {
            fraudChart = new RealtimeChart('fraudChartContainer', { 
                color: '#ef4444', 
                maxPoints: 30 
            });
        }
        
        if (transactionsChartContainer) {
            transactionsChart = new RealtimeChart('transactionsChartContainer', { 
                color: '#3b82f6', 
                maxPoints: 30 
            });
        }
        
        // Initialize Chart.js visualizations
        initCharts();
    }, 500);
}

function createSoundToggle() {
    if (document.getElementById('soundToggle')) return;
    
    const toggle = document.createElement('button');
    toggle.id = 'soundToggle';
    toggle.className = 'sound-toggle' + (soundEffects?.enabled ? ' active' : '');
    toggle.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${soundEffects?.enabled 
                ? '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>'
                : '<path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>'
            }
        </svg>
        <span>Sound</span>
    `;
    
    toggle.addEventListener('click', () => {
        if (soundEffects.enabled) {
            soundEffects.disable();
            toggle.classList.remove('active');
            toggle.querySelector('svg').innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>';
        } else {
            soundEffects.enable();
            toggle.classList.add('active');
            toggle.querySelector('svg').innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>';
            soundEffects.playClick();
        }
    });
    
    document.body.appendChild(toggle);
}

function add3DVisualization() {
    const container = document.getElementById('fraud3DContainer');
    if (!container) return;
    
    container.innerHTML = `
        <div class="fraud-3d-scene">
            <div class="fraud-3d-node low"></div>
            <div class="fraud-3d-node medium"></div>
            <div class="fraud-3d-node high"></div>
            <div class="fraud-3d-node critical"></div>
        </div>
        <div class="fraud-particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
    `;
}

// Play sound on action
function playActionSound(action) {
    if (!soundEffects?.enabled) return;
    
    switch(action) {
        case 'Block':
            soundEffects.playBlock();
            break;
        case 'Review':
            soundEffects.playWarning();
            break;
        case 'Approved':
        case 'Approve':
            soundEffects.playSuccess();
            break;
        default:
            soundEffects.playClick();
    }
}

// Update chart with real-time data
function updateChartData(fraudRate, transactionCount) {
    if (fraudChart) {
        fraudChart.addPoint(fraudRate);
    }
    if (transactionsChart) {
        transactionsChart.addPoint(transactionCount);
    }
}

// Animate stat value change
function animateStatValue(element) {
    if (!element) return;
    element.classList.add('animated');
    setTimeout(() => element.classList.remove('animated'), 500);
}

// ===== CHART.JS VISUALIZATIONS =====

let roiChart = null;
let anomalyScatterChart = null;
let trendChart = null;

const chartColors = {
    primary: '#f59e0b',
    success: '#10b981',
    warning: '#f97316',
    danger: '#ef4444',
    info: '#3b82f6',
    purple: '#8b5cf6',
    grid: 'rgba(45, 58, 79, 0.5)',
    text: '#94a3b8'
};

function initCharts() {
    initROIChart();
    initAnomalyScatterChart();
    initTrendChart();
    setupChartControls();
}

function initROIChart() {
    const ctx = document.getElementById('roiChart');
    if (!ctx) return;

    roiChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [
                {
                    label: 'Fraud Prevented',
                    data: generateMonthlyData(50000, 150000),
                    backgroundColor: chartColors.success + '80',
                    borderColor: chartColors.success,
                    borderWidth: 2,
                    borderRadius: 6,
                    stack: 'stack0'
                },
                {
                    label: 'Cost Incurred',
                    data: generateMonthlyData(10000, 30000),
                    backgroundColor: chartColors.danger + '80',
                    borderColor: chartColors.danger,
                    borderWidth: 2,
                    borderRadius: 6,
                    stack: 'stack0'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: chartColors.text,
                        padding: 15,
                        font: { size: 11, family: "'DM Sans', sans-serif" }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 34, 52, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(245, 158, 11, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (context) => `${context.dataset.label}: $${context.raw.toLocaleString()}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: chartColors.text, font: { size: 11 } }
                },
                y: {
                    grid: { color: chartColors.grid },
                    ticks: {
                        color: chartColors.text,
                        font: { size: 11 },
                        callback: (value) => '$' + (value / 1000).toFixed(0) + 'k'
                    }
                }
            }
        }
    });
    
    updateROISummary();
}

function initAnomalyScatterChart() {
    const ctx = document.getElementById('anomalyScatterChart');
    if (!ctx) return;

    const anomalyData = generateAnomalyData(100);

    anomalyScatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'High Risk',
                    data: anomalyData.high,
                    backgroundColor: chartColors.danger + '80',
                    borderColor: chartColors.danger,
                    pointRadius: 8,
                    pointHoverRadius: 12
                },
                {
                    label: 'Medium Risk',
                    data: anomalyData.medium,
                    backgroundColor: chartColors.warning + '80',
                    borderColor: chartColors.warning,
                    pointRadius: 6,
                    pointHoverRadius: 10
                },
                {
                    label: 'Low Risk',
                    data: anomalyData.low,
                    backgroundColor: chartColors.success + '80',
                    borderColor: chartColors.success,
                    pointRadius: 4,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(26, 34, 52, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(245, 158, 11, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (context) => {
                            const point = context.raw;
                            return `Anomaly Score: ${point.anomalyScore.toFixed(3)}, Prob: ${point.probability.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fraud Probability',
                        color: chartColors.text
                    },
                    min: 0,
                    max: 1,
                    grid: { color: chartColors.grid },
                    ticks: { color: chartColors.text }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Anomaly Score',
                        color: chartColors.text
                    },
                    min: 0,
                    max: 1,
                    grid: { color: chartColors.grid },
                    ticks: { color: chartColors.text }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const datasetIndex = elements[0].datasetIndex;
                    const index = elements[0].index;
                    const point = anomalyScatterChart.data.datasets[datasetIndex].data[index];
                    showAnomalyDetails(point);
                }
            }
        }
    });
}

function initTrendChart() {
    const ctx = document.getElementById('trendChart');
    if (!ctx) return;

    const labels = generateTrendLabels(30);
    const fraudData = generateTrendData(30, 0.5, 3);

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual',
                    data: fraudData.actual,
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primary + '20',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    pointBackgroundColor: chartColors.primary
                },
                {
                    label: 'Trend',
                    data: fraudData.trend,
                    borderColor: chartColors.info,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Forecast',
                    data: fraudData.forecast,
                    borderColor: chartColors.purple,
                    borderDash: [2, 2],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointStyle: 'dash'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: chartColors.text,
                        padding: 15,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 34, 52, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(245, 158, 11, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.raw?.toFixed(2) ?? '--'}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { 
                        color: chartColors.text, 
                        font: { size: 10 },
                        maxRotation: 45
                    }
                },
                y: {
                    grid: { color: chartColors.grid },
                    ticks: { 
                        color: chartColors.text,
                        font: { size: 11 },
                        callback: (value) => value.toFixed(1) + '%'
                    }
                }
            }
        }
    });
    
    updateTrendForecast();
}

function setupChartControls() {
    const campaignSelect = document.getElementById('campaignSelect');
    if (campaignSelect) {
        campaignSelect.addEventListener('change', (e) => {
            updateROIForCampaign(e.target.value);
        });
    }
    
    const trendPeriodSelect = document.getElementById('trendPeriodSelect');
    if (trendPeriodSelect) {
        trendPeriodSelect.addEventListener('change', (e) => {
            updateTrendChart(e.target.value, document.getElementById('trendTypeSelect')?.value || 'fraud');
        });
    }
    
    const trendTypeSelect = document.getElementById('trendTypeSelect');
    if (trendTypeSelect) {
        trendTypeSelect.addEventListener('change', (e) => {
            updateTrendChart(document.getElementById('trendPeriodSelect')?.value || '7d', e.target.value);
        });
    }
    
    document.querySelectorAll('.anomaly-filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.anomaly-filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            filterAnomalyChart(e.target.dataset.filter);
        });
    });
}

function updateROIForCampaign(campaign) {
    if (!roiChart) return;
    
    const dataMap = {
        'all': [generateMonthlyData(50000, 150000), generateMonthlyData(10000, 30000)],
        'q1': [generateMonthlyData(40000, 120000, 3), generateMonthlyData(8000, 25000, 3)],
        'q2': [generateMonthlyData(55000, 130000, 3), generateMonthlyData(12000, 28000, 3)],
        'q3': [generateMonthlyData(60000, 140000, 3), generateMonthlyData(10000, 22000, 3)],
        'q4': [generateMonthlyData(70000, 160000, 3), generateMonthlyData(15000, 35000, 3)]
    };
    
    const selected = dataMap[campaign] || dataMap['all'];
    roiChart.data.datasets[0].data = selected[0];
    roiChart.data.datasets[1].data = selected[1];
    roiChart.update();
    
    updateROISummary();
}

function updateROISummary() {
    const prevented = Math.floor(Math.random() * 500000) + 300000;
    const costs = prevented * 0.15;
    const roi = ((prevented - costs) / costs * 100);
    
    const preventedEl = document.getElementById('roiPrevented');
    const savingsEl = document.getElementById('roiSavings');
    const roiEl = document.getElementById('roiPercent');
    
    if (preventedEl) preventedEl.textContent = '$' + prevented.toLocaleString();
    if (savingsEl) savingsEl.textContent = '$' + Math.floor(costs).toLocaleString();
    if (roiEl) roiEl.textContent = roi.toFixed(0) + '%';
}

function filterAnomalyChart(filter) {
    if (!anomalyScatterChart) return;
    
    anomalyScatterChart.data.datasets.forEach((dataset, i) => {
        if (filter === 'all') {
            dataset.hidden = false;
        } else if (filter === 'high') {
            dataset.hidden = i !== 0;
        } else if (filter === 'medium') {
            dataset.hidden = i !== 1;
        }
    });
    anomalyScatterChart.update();
}

function showAnomalyDetails(point) {
    console.log('Anomaly details:', point);
}

function updateTrendChart(period, type) {
    if (!trendChart) return;
    
    const days = { '7d': 7, '30d': 30, '90d': 90, '1y': 365 }[period] || 30;
    const labels = generateTrendLabels(days);
    const data = generateTrendData(days, 0.5, 3);
    
    trendChart.data.labels = labels;
    trendChart.data.datasets[0].data = data.actual;
    trendChart.data.datasets[1].data = data.trend;
    trendChart.data.datasets[2].data = data.forecast;
    trendChart.update();
    
    updateTrendForecast();
}

function updateTrendForecast() {
    const seasonalEl = document.getElementById('seasonalPattern');
    const longTermEl = document.getElementById('longTermTrend');
    const predictedEl = document.getElementById('predictedFraud');
    
    const patterns = ['Weekly Peak', 'Monthly Spike', 'Seasonal', 'Irregular'];
    const trends = ['Increasing', 'Decreasing', 'Stable', 'Volatile'];
    
    if (seasonalEl) seasonalEl.textContent = patterns[Math.floor(Math.random() * patterns.length)];
    if (longTermEl) longTermEl.textContent = trends[Math.floor(Math.random() * trends.length)];
    if (predictedEl) predictedEl.textContent = (Math.random() * 2 + 1).toFixed(2) + '%';
}

function generateMonthlyData(min, max, count = 12) {
    return Array.from({ length: count }, () => Math.floor(Math.random() * (max - min) + min));
}

function generateAnomalyData(count) {
    const data = { high: [], medium: [], low: [] };
    
    for (let i = 0; i < count; i++) {
        const probability = Math.random();
        const anomalyScore = probability + (Math.random() * 0.3 - 0.15);
        
        const point = {
            x: probability,
            y: Math.min(1, Math.max(0, anomalyScore)),
            probability: probability,
            anomalyScore: Math.min(1, Math.max(0, anomalyScore))
        };
        
        if (probability > 0.6 || anomalyScore > 0.7) {
            data.high.push(point);
        } else if (probability > 0.3 || anomalyScore > 0.4) {
            data.medium.push(point);
        } else {
            data.low.push(point);
        }
    }
    
    return data;
}

function generateTrendLabels(days) {
    const labels = [];
    const now = new Date();
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    return labels;
}

function generateTrendData(days, base, variance) {
    const actual = [];
    const trend = [];
    const forecast = [];
    
    let prevValue = base;
    for (let i = 0; i < days; i++) {
        const noise = (Math.random() - 0.5) * variance;
        const seasonal = Math.sin(i / 7 * Math.PI) * 0.5;
        const value = Math.max(0, Math.min(5, base + noise + seasonal));
        
        actual.push(value);
        trend.push(prevValue + (value - prevValue) * 0.3);
        prevValue = trend[trend.length - 1];
    }
    
    for (let i = 0; i < 7; i++) {
        const lastTrend = trend[trend.length - 1];
        forecast.push(lastTrend + (Math.random() - 0.5) * 0.5);
    }
    
    return { actual, trend, forecast };
}
