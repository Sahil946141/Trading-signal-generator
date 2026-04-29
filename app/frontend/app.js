// ============================================================================
// STATE MANAGEMENT
// ============================================================================
const state = {
  symbols: [],
  predictions: [],
  isDarkMode: localStorage.getItem('darkMode') === 'true',
  currentFile: null,
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initializeTheme();
  loadSupportedStocks();
  setupEventListeners();
  setupDragAndDrop();
});

// ============================================================================
// THEME MANAGEMENT
// ============================================================================
function initializeTheme() {
  const themeToggle = document.getElementById('themeToggle');
  if (state.isDarkMode) {
    document.documentElement.classList.add('dark-mode');
  }

  themeToggle.addEventListener('click', toggleTheme);
}

function toggleTheme() {
  state.isDarkMode = !state.isDarkMode;
  const html = document.documentElement;

  if (state.isDarkMode) {
    html.classList.add('dark-mode');
  } else {
    html.classList.remove('dark-mode');
  }

  localStorage.setItem('darkMode', state.isDarkMode);
}

// ============================================================================
// STOCK SELECTION
// ============================================================================
async function loadSupportedStocks() {
  const select = document.getElementById('predict-symbol');

  try {
    const response = await fetch('/api/stocks');
    const payload = await response.json();
    
    if (!response.ok) {
      throw new Error(payload.detail || 'Unable to load available stock models.');
    }

    state.symbols = payload.symbols || [];
    // If backend returns an empty list, fall back to a sensible default set.
    if (!state.symbols.length) {
      state.symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS'];
    }
    populateStockSelect(state.symbols, payload.default_symbol);
    renderStockSuggestions(state.symbols);
  } catch (error) {
    console.error('Error loading stocks:', error);
    state.symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS'];
    populateStockSelect(state.symbols, 'RELIANCE.NS');
    renderStockSuggestions(state.symbols);
  }
}

function populateStockSelect(symbols, defaultSymbol) {
  const select = document.getElementById('predict-symbol');
  select.innerHTML = symbols.map(symbol => `
    <option value="${symbol}" ${symbol === defaultSymbol ? 'selected' : ''}>${symbol}</option>
  `).join('');
}

function renderStockSuggestions(symbols) {
  const suggestionsContainer = document.getElementById('stockSuggestions');
  suggestionsContainer.innerHTML = symbols.map(symbol => `
    <div class="stock-chip" data-symbol="${symbol}">${symbol}</div>
  `).join('');

  document.querySelectorAll('.stock-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const symbol = chip.getAttribute('data-symbol');
      document.getElementById('predict-symbol').value = symbol;
      showToast(`Selected: ${symbol}`, 'info');
    });
  });
}

// Stock search functionality
document.getElementById('stockSearch')?.addEventListener('input', (e) => {
  const query = e.target.value.toLowerCase();
  const chips = document.querySelectorAll('.stock-chip');

  chips.forEach(chip => {
    const symbol = chip.getAttribute('data-symbol').toLowerCase();
    chip.style.display = symbol.includes(query) ? 'block' : 'none';
  });
});

// ============================================================================
// FILE UPLOAD HANDLING
// ============================================================================
function setupDragAndDrop() {
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('predict-file');

  // Click to upload
  uploadArea.addEventListener('click', () => fileInput.click());

  // Drag and drop
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
  });
}

function handleFiles(files) {
  if (files.length === 0) return;

  const file = files[0];
  
  // Validate file type
  const validTypes = ['image/png', 'image/jpeg', 'image/gif'];
  if (!validTypes.includes(file.type)) {
    showToast('Please upload a valid image file (PNG, JPG, GIF)', 'error');
    return;
  }

  state.currentFile = file;
  displayFilePreview(file);
}

function displayFilePreview(file) {
  const filePreview = document.getElementById('filePreview');
  const fileSize = (file.size / 1024 / 1024).toFixed(2);

  filePreview.innerHTML = `
    <div class="file-preview-item">
      <i class="fas fa-image"></i>
      <div class="file-info">
        <div class="file-name">${file.name}</div>
        <div class="file-size">${fileSize} MB</div>
      </div>
      <button class="remove-file">
        <i class="fas fa-times"></i>
      </button>
    </div>
  `;

  filePreview.classList.add('active');

  filePreview.querySelector('.remove-file').addEventListener('click', () => {
    state.currentFile = null;
    filePreview.classList.remove('active');
    filePreview.innerHTML = '';
    document.getElementById('predict-file').value = '';
  });
}

// ============================================================================
// PREDICTION HANDLING
// ============================================================================
function setupEventListeners() {
  document.getElementById('predict-submit').addEventListener('click', handlePrediction);
  document.getElementById('exportBtn')?.addEventListener('click', exportReport);
  document.getElementById('newPredictionBtn')?.addEventListener('click', resetForm);
}

async function handlePrediction() {
  const fileInput = document.getElementById('predict-file');
  const submitBtn = document.getElementById('predict-submit');
  const statusIndicator = document.getElementById('predict-status');
  const symbol = document.getElementById('predict-symbol').value || 'RELIANCE.NS';

  // Validation
  if (!state.currentFile) {
    showToast('Please upload a chart image first', 'error');
    return;
  }

  // Prepare request
  const body = new FormData();
  body.append('file', state.currentFile);
  body.append('symbol', symbol);
  body.append('timeframe', '5m');
  body.append('use_lstm', true);

  // Update UI state
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
  statusIndicator.querySelector('.status-dot').classList.add('running');
  statusIndicator.querySelector('.status-text').textContent = 'Running';
  document.getElementById('loadingSpinner').style.display = 'flex';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || payload.error?.message || 'Prediction failed');
    }

    // Process results
    displayPredictionResults(payload, symbol);
    addToHistory(payload, symbol);
    showToast('Prediction completed successfully!', 'success');

    // Update status
    statusIndicator.querySelector('.status-dot').classList.remove('running');
    statusIndicator.querySelector('.status-text').textContent = 'Ready';
  } catch (error) {
    console.error('Prediction error:', error);
    displayError(error.message);
    statusIndicator.querySelector('.status-dot').classList.add('error');
    statusIndicator.querySelector('.status-text').textContent = 'Error';
    showToast(error.message, 'error');
  } finally {
    // Reset button
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-magic"></i> Run Prediction';
    document.getElementById('loadingSpinner').style.display = 'none';
  }
}

function displayPredictionResults(payload, symbol) {
  const resultPlaceholder = document.getElementById('predict-result');
  const resultDetails = document.getElementById('resultDetails');
  const signalBox = document.getElementById('signalBox');

  const pattern = payload.data.pattern;
  const signal = payload.data.signal;
  const metadata = payload.metadata;
  const signalClass = signal.label.toLowerCase();

  // Hide placeholder and show details
  resultPlaceholder.style.display = 'none';
  resultDetails.style.display = 'block';

  // Signal box
  signalBox.className = `result-signal-box ${signalClass}`;
  signalBox.innerHTML = `
    <i class="fas fa-${getSignalIcon(signalClass)}"></i>
    ${signal.label} Signal
  `;

  // Metrics
  document.getElementById('resultSignal').textContent = signal.label;
  document.getElementById('resultConfidence').textContent = `${(signal.confidence * 100).toFixed(2)}%`;
  document.getElementById('resultPattern').textContent = pattern.label;
  document.getElementById('resultPatternConfidence').textContent = `${(pattern.confidence * 100).toFixed(2)}%`;

  // OHLC Data
  renderMetrics('predict-ohlc', metadata.latest_candle);

  // Info
  document.getElementById('infoStock').textContent = metadata.instrument;
  document.getElementById('infoTimeframe').textContent = metadata.timeframe;
  document.getElementById('infoCandle').textContent = metadata.latest_candle?.time || 'N/A';
  document.getElementById('infoModel').textContent = metadata.model_enhancements?.attached_model_path || 'Attached stock model';

  // Result badge
  const resultBadge = document.getElementById('resultBadge');
  resultBadge.className = `result-badge`;
  resultBadge.innerHTML = `<span class="badge-signal ${signalClass}">${signal.label}</span>`;
  resultBadge.style.display = 'flex';
}

function getSignalIcon(signalClass) {
  const icons = {
    'buy': 'arrow-up',
    'sell': 'arrow-down',
    'hold': 'pause'
  };
  return icons[signalClass] || 'question';
}

function renderMetrics(containerId, candle) {
  const container = document.getElementById(containerId);
  if (!candle) {
    container.innerHTML = '<p class="muted-text">No market data available</p>';
    return;
  }

  const metrics = [
    ['Open', candle.open],
    ['High', candle.high],
    ['Low', candle.low],
    ['Close', candle.close],
  ];

  container.innerHTML = metrics.map(([label, value]) => `
    <div class="ohlc-card">
      <span class="ohlc-label">${label}</span>
      <span class="ohlc-value">₹${Number(value).toFixed(2)}</span>
    </div>
  `).join('');
}

function displayError(message) {
  const resultPlaceholder = document.getElementById('predict-result');
  const resultDetails = document.getElementById('resultDetails');

  resultDetails.style.display = 'none';
  resultPlaceholder.style.display = 'block';
  resultPlaceholder.innerHTML = `
    <div style="text-align: center; color: #ef4444;">
      <i class="fas fa-exclamation-circle" style="font-size: 3rem; margin-bottom: 16px;"></i>
      <p><strong>Error</strong></p>
      <p>${message}</p>
    </div>
  `;
}

// ============================================================================
// HISTORY MANAGEMENT
// ============================================================================
function addToHistory(payload, symbol) {
  const prediction = {
    symbol,
    signal: payload.data.signal.label,
    confidence: payload.data.signal.confidence,
    pattern: payload.data.pattern.label,
    timestamp: new Date().toLocaleTimeString(),
  };

  state.predictions.unshift(prediction);
  if (state.predictions.length > 10) {
    state.predictions.pop();
  }

  renderHistory();
}

function renderHistory() {
  const historyList = document.getElementById('historyList');

  if (state.predictions.length === 0) {
    historyList.innerHTML = '<p class="muted-text">No predictions yet</p>';
    return;
  }

  historyList.innerHTML = state.predictions.map((pred, index) => `
    <div class="history-item">
      <div class="history-item-header">
        <span class="history-item-symbol">${pred.symbol}</span>
        <span class="history-item-signal ${pred.signal.toLowerCase()}">
          ${pred.signal}
        </span>
      </div>
      <div class="history-item-details">
        <p>${pred.pattern} • ${(pred.confidence * 100).toFixed(1)}% confidence</p>
        <p style="margin-top: 4px; font-size: 0.8rem; opacity: 0.7;">${pred.timestamp}</p>
      </div>
    </div>
  `).join('');
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function resetForm() {
  document.getElementById('predict-file').value = '';
  document.getElementById('filePreview').classList.remove('active');
  document.getElementById('filePreview').innerHTML = '';
  state.currentFile = null;

  const resultPlaceholder = document.getElementById('predict-result');
  const resultDetails = document.getElementById('resultDetails');
  resultDetails.style.display = 'none';
  resultPlaceholder.style.display = 'block';
  resultPlaceholder.innerHTML = `
    <div class="placeholder-icon">
      <i class="fas fa-chart-area"></i>
    </div>
    <p>Upload a chart image and run the prediction to see results</p>
  `;

  document.getElementById('resultBadge').style.display = 'none';
}

function exportReport() {
  const data = {
    signal: document.getElementById('resultSignal').textContent,
    confidence: document.getElementById('resultConfidence').textContent,
    pattern: document.getElementById('resultPattern').textContent,
    stock: document.getElementById('infoStock').textContent,
    timeframe: document.getElementById('infoTimeframe').textContent,
    timestamp: new Date().toISOString(),
  };

  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `prediction-${Date.now()}.json`;
  a.click();
  window.URL.revokeObjectURL(url);

  showToast('Report exported successfully!', 'success');
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
    <span>${message}</span>
    <button class="toast-close">
      <i class="fas fa-times"></i>
    </button>
  `;

  container.appendChild(toast);

  // Auto remove after 5 seconds
  setTimeout(() => {
    toast.style.animation = 'slideOutRight 300ms ease-out forwards';
    setTimeout(() => toast.remove(), 300);
  }, 5000);

  // Manual close
  toast.querySelector('.toast-close').addEventListener('click', () => {
    toast.style.animation = 'slideOutRight 300ms ease-out forwards';
    setTimeout(() => toast.remove(), 300);
  });
}
