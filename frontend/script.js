/* ═══════════════════════════════════════════════════════════════════════════
   PersonaVerify — ML Interface Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API_PREDICT = 'http://127.0.0.1:8000/api/predict/';
const API_BULK = 'http://127.0.0.1:8000/api/predict-bulk/';
const API_STATS = 'http://127.0.0.1:8000/api/stats/';

document.addEventListener('DOMContentLoaded', () => {

    // ─── View Switcher Logic ───
    const navBtns = document.querySelectorAll('.nav-view-btn');
    const views = document.querySelectorAll('.view-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = btn.getAttribute('data-target');
            
            navBtns.forEach(b => b.classList.remove('active'));
            views.forEach(v => {
                v.classList.remove('active');
                v.style.display = 'none'; // Force hide for robust switching
            });
            
            btn.classList.add('active');
            const targetView = document.getElementById(targetId);
            targetView.classList.add('active');
            targetView.style.display = 'flex'; // Restore flex

            if (targetId === 'view-metrics') {
                loadMetrics();
            }
        });
    });

    // ─── Loading Overlay Logic ───
    const overlay = document.getElementById('loadingOverlay');
    const overlayText = document.getElementById('loadingText');
    
    function showLoading(text) {
        overlayText.textContent = text || 'Processing...';
        overlay.style.display = 'flex';
    }
    function hideLoading() {
        overlay.style.display = 'none';
    }

    // ─── DOM Elements (Predictor) ───
    const form = document.getElementById('mlForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsContent = document.getElementById('resultsContent');
    const verdictBanner = document.getElementById('verdictBanner');
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const ruleList = document.getElementById('ruleList');
    const xaiChart = document.getElementById('xaiChart');

    // Tabs inside Predictor
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // Test Data Loaders
    document.getElementById('btnFillFake').addEventListener('click', () => {
        document.getElementById('profile_pic').value = "0";
        document.getElementById('private').value = "0";
        document.getElementById('external_url').value = "0";
        document.getElementById('nums_length_username').value = "0.75";
        document.getElementById('name_eq_username').value = "1";
        document.getElementById('fullname_words').value = "0";
        document.getElementById('nums_length_fullname').value = "0.0";
        document.getElementById('description_length').value = "0";
        document.getElementById('posts').value = "0";
        document.getElementById('followers').value = "10";
        document.getElementById('follows').value = "1200";
    });

    document.getElementById('btnFillReal').addEventListener('click', () => {
        document.getElementById('profile_pic').value = "1";
        document.getElementById('private').value = "0";
        document.getElementById('external_url').value = "1";
        document.getElementById('nums_length_username').value = "0.05";
        document.getElementById('name_eq_username').value = "0";
        document.getElementById('fullname_words').value = "2";
        document.getElementById('nums_length_fullname').value = "0.0";
        document.getElementById('description_length').value = "150";
        document.getElementById('posts').value = "300";
        document.getElementById('followers').value = "2500";
        document.getElementById('follows').value = "800";
    });

    // Single Prediction Form Submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoading('Running Tree Prediction...');
        submitBtn.disabled = true;

        const fd = new FormData(form);
        const payload = {};
        const INT_FIELDS = ['profile_pic', 'private', 'external_url', 'name_eq_username', 'fullname_words', 'description_length', 'posts', 'followers', 'follows'];
        const FLOAT_FIELDS = ['nums_length_username', 'nums_length_fullname'];

        for (const [k, v] of fd.entries()) {
            if (INT_FIELDS.includes(k)) payload[k] = parseInt(v, 10);
            else if (FLOAT_FIELDS.includes(k)) payload[k] = parseFloat(v);
            else payload[k] = v;
        }

        try {
            const res = await fetch(API_PREDICT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error('API Request Failed');
            const data = await res.json();
            renderOutput(data);
        } catch (err) {
            alert('Failed to connect to API. Is the server running?');
        } finally {
            hideLoading();
            submitBtn.disabled = false;
        }
    });

    function renderOutput(data) {
        resultsEmpty.style.display = 'none';
        resultsContent.style.display = 'flex';
        document.querySelector('[data-target="tab-overview"]').click();

        const isFake = data.prediction === 'Fake';
        const confPct = (data.confidence_score * 100).toFixed(2);

        verdictBanner.className = `verdict-banner ${isFake ? 'v-fake' : 'v-real'}`;
        predictionText.textContent = isFake ? 'Fake Profile Detected' : 'Real Profile Detected';
        confidenceText.textContent = `${confPct}%`;

        // Risk Rules
        ruleList.innerHTML = '';
        const risks = data.explainability.risk_factors;
        if (risks.length === 0) {
            ruleList.innerHTML = '<li><span class="rule-icon">✓</span> Standard parameters detected. No severe risk patterns.</li>';
        } else {
            risks.forEach(risk => {
                const li = document.createElement('li');
                const isAlert = risk.direction === 'toward_fake';
                li.className = isAlert ? 'rule-fake' : 'rule-real';
                li.innerHTML = `<span class="rule-icon">${isAlert ? '⚠️' : 'ℹ️'}</span> ${risk.message}`;
                ruleList.appendChild(li);
            });
        }

        // XAI Chart
        xaiChart.innerHTML = '';
        const contributions = data.explainability.feature_contributions.slice(0, 8);
        const maxContrib = Math.max(...contributions.map(c => Math.abs(c.contribution)), 0.001);

        contributions.forEach(item => {
            const row = document.createElement('div');
            row.className = 'xai-row';
            const pushesFake = item.contribution > 0;
            const widthPct = Math.min((Math.abs(item.contribution) / maxContrib) * 100, 100);
            const sign = pushesFake ? '+' : '';
            const valStr = `${sign}${item.contribution.toFixed(4)}`;

            row.innerHTML = `
                <div class="xai-label" title="${item.label}">${item.label}</div>
                <div class="xai-track">
                    <div class="xai-bar ${pushesFake ? 'push-fake' : 'push-real'}" style="width: ${widthPct}%"></div>
                </div>
                <div class="xai-val-outside ${pushesFake ? 'push-fake-text' : 'push-real-text'}">${valStr}</div>
            `;
            xaiChart.appendChild(row);
        });
    }

    // ─── Bulk CSV Logic ───
    const csvInput = document.getElementById('csvFile');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const bulkForm = document.getElementById('bulkForm');

    csvInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
            fileNameDisplay.style.color = 'var(--primary)';
        } else {
            fileNameDisplay.textContent = 'No file selected';
            fileNameDisplay.style.color = 'var(--text-muted)';
        }
    });

    bulkForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (csvInput.files.length === 0) return alert('Please select a CSV file');
        
        showLoading('Processing Bulk Queue...');
        
        const formData = new FormData();
        formData.append('file', csvInput.files[0]);

        try {
            const res = await fetch(API_BULK, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (!res.ok) {
                alert(`Error: ${data.error}`);
                return;
            }

            document.getElementById('bulkResults').style.display = 'block';
            document.getElementById('bulkTotal').textContent = data.total_processed;
            document.getElementById('bulkFake').textContent = data.total_fake;
            document.getElementById('bulkReal').textContent = data.total_real;

            const tbody = document.querySelector('#bulkTable tbody');
            tbody.innerHTML = '';
            
            data.results.forEach(r => {
                const tr = document.createElement('tr');
                if (r.error) {
                    tr.innerHTML = `<td>${r.row}</td><td colspan="3" style="color:red;">Error: ${r.error}</td>`;
                } else {
                    const conf = (r.confidence * 100).toFixed(1) + '%';
                    const color = r.prediction === 'Fake' ? 'var(--fake-color)' : 'var(--real-color)';
                    tr.innerHTML = `
                        <td>${r.row}</td>
                        <td style="color:${color}; font-weight:bold;">${r.prediction}</td>
                        <td>${conf}</td>
                        <td>${r.top_features.slice(0,2).join(', ')}</td>
                    `;
                }
                tbody.appendChild(tr);
            });

        } catch (err) {
            alert('Bulk processing failed. See console.');
            console.error(err);
        } finally {
            hideLoading();
        }
    });

    // ─── Metrics Dashboard Logic ───
    async function loadMetrics() {
        showLoading('Loading Metrics...');
        try {
            const res = await fetch(API_STATS);
            if (!res.ok) throw new Error('API Request Failed');
            const data = await res.json();
            
            document.getElementById('dashTotal').textContent = data.total_predictions;
            document.getElementById('dashFake').textContent = data.fake_predictions;
            document.getElementById('dashReal').textContent = data.real_predictions;
        } catch (err) {
            console.error("Could not load metrics", err);
        } finally {
            hideLoading();
        }
    }

});
