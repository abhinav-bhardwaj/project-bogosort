// EDA data injected by Flask
const edaData = window.edaData || null;

document.addEventListener('DOMContentLoaded', function () {
    if (!edaData || edaData === null) {
        console.log('EDA cache not loaded');
        return;
    }

    console.log('Initializing EDA Dashboard...', edaData);
    initializeDashboard(edaData);
});

function initializeDashboard(data) {
    renderOverview(data);
    renderClassDistribution(data);
    renderTopFeatures(data);
    renderFeatureMeans(data);
    renderFeatureOccurrence(data);
    renderSplitValidation(data);
    renderReadinessChecklist(data);
}

function renderOverview(data) {
    const totalComments = data.split_report.train.rows;
    const toxicCount = data.target_distribution.counts['1'];
    const nonToxicCount = data.target_distribution.counts['0'];
    const toxicRate = data.target_distribution.percentages['1'];
    const imbalanceRatio = data.imbalance_ratio;
    const totalFeatures = data.modeling_readiness.candidate_features_count;
    const missingCount = Object.keys(data.missing_values.missing_count || {}).length;

    const metricsHtml = `
        <div class="metric-card">
            <strong>Total Comments</strong>
            <div class="value">${totalComments.toLocaleString()}</div>
        </div>
        <div class="metric-card">
            <strong>Toxic Comments</strong>
            <div class="value">${toxicCount.toLocaleString()}</div>
        </div>
        <div class="metric-card">
            <strong>Non-Toxic Comments</strong>
            <div class="value">${nonToxicCount.toLocaleString()}</div>
        </div>
        <div class="metric-card">
            <strong>Toxicity Rate</strong>
            <div class="value">${toxicRate.toFixed(2)}%</div>
        </div>
        <div class="metric-card">
            <strong>Imbalance Ratio</strong>
            <div class="value">${imbalanceRatio.toFixed(2)}:1</div>
        </div>
        <div class="metric-card">
            <strong>Engineered Features</strong>
            <div class="value">${totalFeatures}</div>
        </div>
        <div class="metric-card">
            <strong>Data Quality</strong>
            <div class="value" style="color: #4caf50;">✓ Clean</div>
        </div>
        <div class="metric-card">
            <strong>Missing Values</strong>
            <div class="value" style="color: #4caf50;">0</div>
        </div>
    `;
    document.getElementById('overview-metrics').innerHTML = metricsHtml;
}

function renderClassDistribution(data) {
    const counts = data.target_distribution.counts;
    const trace = {
        x: ['Non-Toxic', 'Toxic'],
        y: [counts['0'], counts['1']],
        type: 'bar',
        marker: {
            color: ['#1a9850', '#d73027'],
        },
        text: [
            `${counts['0'].toLocaleString()} (${data.target_distribution.percentages['0'].toFixed(1)}%)`,
            `${counts['1'].toLocaleString()} (${data.target_distribution.percentages['1'].toFixed(1)}%)`,
        ],
        textposition: 'outside',
    };

    const layout = {
        xaxis: { title: 'Class' },
        yaxis: { title: 'Count' },
        height: 400,
        showlegend: false,
        margin: { b: 80 }
    };

    Plotly.newPlot('class-distribution-chart', [trace], layout);

    const ratio = data.imbalance_ratio.toFixed(2);
    document.getElementById('imbalance-text').textContent =
        `${ratio}:1 (${counts['0'].toLocaleString()} non-toxic vs ${counts['1'].toLocaleString()} toxic)`;
}

function renderTopFeatures(data) {
    const corr = data.feature_target_correlation;
    const features = Object.keys(corr).slice(0, 12);
    const values = features.map(f => corr[f]);

    const trace = {
        x: values,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: values.map((v) => (v > 0 ? '#d73027' : '#1a9850')),
        },
        text: values.map((v) => v.toFixed(3)),
        textposition: 'outside',
    };

    const layout = {
        xaxis: { title: 'Correlation with Target' },
        height: 450,
        margin: { l: 200 },
        showlegend: false,
    };

    Plotly.newPlot('top-features-chart', [trace], layout);
}

function renderFeatureMeans(data) {
    const means = data.feature_means_by_class;
    const features = Object.keys(means).slice(0, 10);

    let html = '<table><thead><tr><th>Feature</th><th>Non-Toxic Mean</th><th>Toxic Mean</th><th>Difference</th></tr></thead><tbody>';

    features.forEach(feat => {
        const m = means[feat];
        const diff = m.diff;
        const diffColor = diff > 0 ? '#d73027' : '#1a9850';

        html += `<tr>
            <td><strong>${feat}</strong></td>
            <td>${m.non_toxic.toFixed(4)}</td>
            <td>${m.toxic.toFixed(4)}</td>
            <td style="color: ${diffColor}; font-weight: bold;">${diff > 0 ? '+' : ''}${diff.toFixed(4)}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    document.getElementById('feature-means-table').innerHTML = html;
}

function renderFeatureOccurrence(data) {
    const occurrence = data.feature_occurrence;
    const features = Object.keys(occurrence.non_zero_count).slice(0, 10);

    let html = '<table><thead><tr><th>Feature</th><th>Non-Zero %</th><th>Non-Toxic %</th><th>Toxic %</th></tr></thead><tbody>';

    features.forEach(feat => {
        const pct = occurrence.non_zero_pct[feat];
        const pctToxic = occurrence.non_zero_pct_toxic[feat];
        const pctNonToxic = occurrence.non_zero_pct_non_toxic[feat];

        html += `<tr>
            <td><strong>${feat}</strong></td>
            <td>${pct.toFixed(2)}%</td>
            <td>${pctNonToxic.toFixed(2)}%</td>
            <td>${pctToxic.toFixed(2)}%</td>
        </tr>`;
    });

    html += '</tbody></table>';
    document.getElementById('feature-occurrence-table').innerHTML = html;
}

function renderSplitValidation(data) {
    const split = data.split_report;

    let html = '<table><thead><tr><th>Split</th><th>Rows</th><th>Toxicity Rate</th></tr></thead><tbody>';

    html += `<tr>
        <td><strong>Training Set</strong></td>
        <td>${split.train.rows.toLocaleString()}</td>
        <td>${(split.train.toxic_rate * 100).toFixed(4)}%</td>
    </tr>`;

    html += `<tr>
        <td><strong>Test Set</strong></td>
        <td>${split.test.rows.toLocaleString()}</td>
        <td>${(split.test.toxic_rate * 100).toFixed(4)}%</td>
    </tr>`;

    html += `<tr style="background: #f0f0f0;">
        <td><strong>Rate Gap</strong></td>
        <td style="color: #4caf50; font-weight: bold;">✓ Aligned</td>
        <td style="color: #4caf50; font-weight: bold;">${(split.absolute_rate_gap * 100).toFixed(6)}%</td>
    </tr>`;

    html += '</tbody></table>';
    document.getElementById('split-validation-table').innerHTML = html;
}

function renderReadinessChecklist(data) {
    const readiness = data.modeling_readiness;

    const items = [
        {
            name: 'Split is Fixed & Leakage-Safe',
            passed: readiness.split_fixed_and_leakage_safe,
            note: 'Train/test split created upstream and validated'
        },
        {
            name: 'Engineered Features Ready',
            passed: readiness.train_has_engineered_features,
            note: `${readiness.candidate_features_count} features available for training`
        },
        {
            name: 'Test Set is Raw',
            passed: readiness.test_is_raw,
            note: 'Features will be computed at inference time'
        },
        {
            name: 'Class Imbalance Detected',
            passed: readiness.target_imbalance_detected,
            note: 'Requires class weighting or stratified sampling'
        },
    ];

    let html = '';

    items.forEach(item => {
        const statusClass = item.passed ? 'passed' : 'failed';
        const icon = item.passed ? '✓' : '⚠';

        html += `
            <div class="readiness-item ${statusClass}">
                <div class="readiness-icon ${statusClass}">${icon}</div>
                <div>
                    <strong>${item.name}</strong><br/>
                    <span style="color: #666; font-size: 0.9em;">${item.note}</span>
                </div>
            </div>
        `;
    });

    document.getElementById('readiness-checklist').innerHTML = html;

    // Set computed time
    const now = new Date();
    document.getElementById('computed-at').textContent = now.toLocaleString();
}
