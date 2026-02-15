// FraudSense AI - Enterprise Risk Platform
// Security Operations Center Dashboard

const API_BASE = 'http://localhost:8001';

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
    
    // Stress Test
    stressTestBtn: document.getElementById('stressTestBtn'),
    stressResults: document.getElementById('stressResults'),
    stressProcessed: document.getElementById('stressProcessed'),
    stressAvgTime: document.getElementById('stressAvgTime'),
    stressTps: document.getElementById('stressTps'),
    
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
    }
    if (elements.fraudRate) {
        elements.fraudRate.textContent = summary.fraud_rate_percentage.toFixed(2) + '%';
    }
    if (elements.highRiskCount) {
        elements.highRiskCount.textContent = (summary.risk_distribution.High || 0) + (summary.risk_distribution.Critical || 0);
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
        const result = await fetchAPI('/predict', {
            method: 'POST',
            body: JSON.stringify(features)
        });
        const endTime = performance.now();
        
        const latency = Math.round(endTime - startTime);
        displayResult(result, latency);
        
    } catch (error) {
        console.error('Analysis error:', error);
    } finally {
        setLoading(elements.analyzeBtn, false);
    }
}

// Simulate Transaction
async function simulateTransaction() {
    setLoading(elements.simulateBtn, true);
    
    try {
        const startTime = performance.now();
        const result = await fetchAPI('/simulate');
        const endTime = performance.now();
        
        const latency = Math.round(endTime - startTime);
        displayResult(result, latency);
        
    } catch (error) {
        console.error('Simulation error:', error);
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
    currentPrediction = result;
    
    if (elements.transactionId) elements.transactionId.textContent = result.transaction_id;
    
    if (result.final_risk_score !== undefined) {
        updateRiskGauge(result.final_risk_score, result.risk_band);
    }
    
    if (elements.fraudProbBar) updateResultBar(elements.fraudProbBar, elements.fraudProbability, result.fraud_probability);
    if (elements.anomalyBar) updateResultBar(elements.anomalyBar, elements.anomalyScore, result.anomaly_score);
    if (elements.riskBar) updateResultBar(elements.riskBar, elements.ensembleRisk, result.final_risk_score);
    
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
    if (elements.actionText) updateActionRecommendation(result.action_recommendation, result.action_reasoning);
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
    
    if (action.includes('Block')) {
        elements.actionBox.classList.add('block');
        elements.actionIcon.textContent = '⛔';
    } else if (action.includes('Review')) {
        elements.actionBox.classList.add('review');
        elements.actionIcon.textContent = '👁️';
    } else {
        elements.actionBox.classList.add('approve');
        elements.actionIcon.textContent = '✓';
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
        elements.actionIcon.textContent = '✓';
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
        button.dataset.originalText = button.textContent;
        button.textContent = 'Loading...';
    } else {
        button.disabled = false;
        button.textContent = button.dataset.originalText;
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

// ===== STRESS TEST =====
async function runStressTest() {
    if (!elements.stressTestBtn) return;
    
    setLoading(elements.stressTestBtn, true);
    elements.stressResults.style.display = 'none';
    
    try {
        const result = await fetchAPI('/stress-test');
        
        elements.stressProcessed.textContent = result.total_processed;
        elements.stressAvgTime.textContent = result.average_inference_time_ms + 'ms';
        elements.stressTps.textContent = result.transactions_per_second + ' TPS';
        elements.stressResults.style.display = 'flex';
        
    } catch (error) {
        console.error('Stress test error:', error);
    } finally {
        setLoading(elements.stressTestBtn, false);
    }
}
