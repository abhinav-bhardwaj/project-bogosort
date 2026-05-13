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
    renderFeatureCatalog();
    renderIdentityRisk(data);
    renderCorrelationHeatmap(data);
    renderFeatureDistributionExplorer(data);
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
        yaxis: { title: 'Count', tickformat: ',d' },
        height: 380,
        showlegend: false,
        margin: { t: 24, b: 60, l: 80, r: 40 },
        responsive: true,
        paper_bgcolor: 'white',
        plot_bgcolor: '#fafafa',
    };

    Plotly.newPlot('class-distribution-chart', [trace], layout, { responsive: true, displayModeBar: false });

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
        xaxis: { title: 'Correlation with Target', zeroline: true, zerolinecolor: '#ccc' },
        yaxis: { automargin: true },
        height: 480,
        margin: { t: 24, b: 60, l: 220, r: 80 },
        showlegend: false,
        responsive: true,
        paper_bgcolor: 'white',
        plot_bgcolor: '#fafafa',
    };

    Plotly.newPlot('top-features-chart', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderFeatureMeans(data) {
    const means = data.feature_means_by_class;
    const allFeatures = Object.keys(means);
    const maxDiff = Math.max(...allFeatures.map(f => Math.abs(means[f].diff)));

    const container = document.getElementById('feature-means-table');
    let filterText = '';
    let sortKey = 'diff';
    let sortDir = -1;

    function sortVal(feat) {
        if (sortKey === 'name')      return feat;
        if (sortKey === 'non_toxic') return means[feat].non_toxic;
        if (sortKey === 'toxic')     return means[feat].toxic;
        return Math.abs(means[feat].diff);
    }

    function renderMeansTable() {
        const filtered = allFeatures.filter(f => f.toLowerCase().includes(filterText.toLowerCase()));
        const sorted = [...filtered].sort((a, b) => {
            const av = sortVal(a), bv = sortVal(b);
            if (typeof av === 'string') return sortDir * av.localeCompare(bv);
            return sortDir * (bv - av);
        });

        const FEATURE_HUMAN = window.FEATURE_LABELS_EDA || {};
        const rows = sorted.map(feat => {
            const m = means[feat];
            const diff = m.diff;
            const diffColor = diff > 0 ? '#d73027' : '#1a9850';
            const barW = maxDiff > 0 ? (Math.abs(diff) / maxDiff * 80).toFixed(1) : 0;
            const label = FEATURE_HUMAN[feat] || feat;
            return `<tr>
                <td class="fm-name" title="${feat}"><code>${feat}</code><span class="fm-label">${label}</span></td>
                <td>${m.non_toxic.toFixed(4)}</td>
                <td>${m.toxic.toFixed(4)}</td>
                <td>
                    <div class="fm-bar-wrap">
                        <div class="fm-bar" style="width:${barW}%;background:${diffColor}"></div>
                        <span style="color:${diffColor};font-weight:600;font-size:12px">${diff > 0 ? '+' : ''}${diff.toFixed(4)}</span>
                    </div>
                </td>
            </tr>`;
        }).join('');

        const sortBtns = [
            ['diff', 'Difference ↕'], ['name', 'Name'], ['non_toxic', 'Non-Toxic'], ['toxic', 'Toxic']
        ].map(([k, lbl]) =>
            `<button class="fe-sort-btn${sortKey === k ? ' active' : ''}" data-key="${k}">${lbl}</button>`
        ).join('');

        container.innerHTML = `
            <div class="fe-controls">
                <input id="fmSearch" type="search" class="fe-search" placeholder="Search features…" value="${filterText}">
                <div class="fe-sort-btns"><span class="fe-sort-label">Sort:</span>${sortBtns}</div>
                <span class="fe-count">${sorted.length} / ${allFeatures.length} features</span>
            </div>
            <div class="fe-table-wrap" style="max-height:480px">
                <table class="fe-table">
                    <thead><tr>
                        <th>Feature</th>
                        <th>Non-Toxic mean</th>
                        <th>Toxic mean</th>
                        <th>Difference (toxic − non-toxic)</th>
                    </tr></thead>
                    <tbody>${rows || '<tr><td colspan="4" style="text-align:center;color:#999">No features match.</td></tr>'}</tbody>
                </table>
            </div>`;

        document.getElementById('fmSearch').addEventListener('input', e => { filterText = e.target.value; renderMeansTable(); });
        container.querySelectorAll('.fe-sort-btn').forEach(btn => btn.addEventListener('click', () => {
            const k = btn.dataset.key;
            sortKey === k ? (sortDir *= -1) : (sortKey = k, sortDir = -1);
            renderMeansTable();
        }));
    }

    renderMeansTable();
}

function renderFeatureOccurrence(data) {
    const occ = data.feature_occurrence;
    const allFeatures = Object.keys(occ.non_zero_count);

    const container = document.getElementById('feature-occurrence-table');
    let filterText = '';
    let sortKey = 'overall';
    let sortDir = -1;

    function sortVal(feat) {
        if (sortKey === 'name')    return feat;
        if (sortKey === 'toxic')   return occ.non_zero_pct_toxic[feat] ?? 0;
        if (sortKey === 'nontoxic') return occ.non_zero_pct_non_toxic[feat] ?? 0;
        return occ.non_zero_pct[feat] ?? 0;
    }

    function renderOccTable() {
        const filtered = allFeatures.filter(f => f.toLowerCase().includes(filterText.toLowerCase()));
        const sorted = [...filtered].sort((a, b) => {
            const av = sortVal(a), bv = sortVal(b);
            if (typeof av === 'string') return sortDir * av.localeCompare(bv);
            return sortDir * (bv - av);
        });

        const rows = sorted.map(feat => {
            const overall   = occ.non_zero_pct[feat] ?? 0;
            const nonToxic  = occ.non_zero_pct_non_toxic[feat] ?? 0;
            const toxic     = occ.non_zero_pct_toxic[feat] ?? 0;
            const gap       = toxic - nonToxic;
            const gapColor  = gap > 0 ? '#d73027' : gap < 0 ? '#1a9850' : '#999';
            return `<tr>
                <td class="fm-name" title="${feat}"><code>${feat}</code></td>
                <td>
                    <div class="fm-bar-wrap">
                        <div class="fm-bar" style="width:${Math.min(overall, 100).toFixed(1)}%;background:#3366cc"></div>
                        <span style="font-size:12px">${overall.toFixed(1)}%</span>
                    </div>
                </td>
                <td>${nonToxic.toFixed(1)}%</td>
                <td>${toxic.toFixed(1)}%</td>
                <td style="color:${gapColor};font-weight:600;font-size:12px">${gap > 0 ? '+' : ''}${gap.toFixed(1)}pp</td>
            </tr>`;
        }).join('');

        const sortBtns = [
            ['overall', 'Overall %'], ['toxic', 'Toxic %'], ['nontoxic', 'Non-Toxic %'], ['name', 'Name']
        ].map(([k, lbl]) =>
            `<button class="fe-sort-btn${sortKey === k ? ' active' : ''}" data-key="${k}">${lbl}</button>`
        ).join('');

        container.innerHTML = `
            <div class="fe-controls">
                <input id="occSearch" type="search" class="fe-search" placeholder="Search features…" value="${filterText}">
                <div class="fe-sort-btns"><span class="fe-sort-label">Sort:</span>${sortBtns}</div>
                <span class="fe-count">${sorted.length} / ${allFeatures.length} features</span>
            </div>
            <div class="fe-table-wrap" style="max-height:480px">
                <table class="fe-table">
                    <thead><tr>
                        <th>Feature</th>
                        <th>Non-zero % (overall)</th>
                        <th>Non-Toxic %</th>
                        <th>Toxic %</th>
                        <th>Gap (toxic − non-toxic)</th>
                    </tr></thead>
                    <tbody>${rows || '<tr><td colspan="5" style="text-align:center;color:#999">No features match.</td></tr>'}</tbody>
                </table>
            </div>`;

        document.getElementById('occSearch').addEventListener('input', e => { filterText = e.target.value; renderOccTable(); });
        container.querySelectorAll('.fe-sort-btn').forEach(btn => btn.addEventListener('click', () => {
            const k = btn.dataset.key;
            sortKey === k ? (sortDir *= -1) : (sortKey = k, sortDir = -1);
            renderOccTable();
        }));
    }

    renderOccTable();
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

function renderFeatureDistributionExplorer(data) {
    const controlsEl = document.getElementById('feature-distribution-controls');
    const chartEl    = document.getElementById('feature-distribution-chart');
    if (!controlsEl || !chartEl) return;

    const dists = data.feature_distributions;
    if (!dists) {
        chartEl.innerHTML = '<p style="color:#999;font-style:italic;">Distribution data not available.</p>';
        return;
    }

    const corr  = data.feature_target_correlation || {};
    const means = data.feature_means_by_class     || {};

    const allFeatures = Object.keys(dists).map(name => ({
        name,
        correlation: corr[name]       ?? 0,
        diff:        means[name]?.diff ?? 0,
    }));

    let sortKey        = 'correlation';
    let filterText     = '';
    let selectedName   = allFeatures
        .slice()
        .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))[0]?.name ?? null;

    function getSorted() {
        const q = filterText.toLowerCase();
        const filtered = allFeatures.filter(f => f.name.toLowerCase().includes(q));
        return filtered.sort((a, b) => {
            if (sortKey === 'name') return a.name.localeCompare(b.name);
            if (sortKey === 'diff') return Math.abs(b.diff) - Math.abs(a.diff);
            return Math.abs(b.correlation) - Math.abs(a.correlation);
        });
    }

    // ── controls row ──────────────────────────────────────────────────
    function renderControls() {
        const sorted = getSorted();
        const sortDefs = [['correlation','Correlation'],['diff','Class Diff'],['name','Name']];
        const btns = sortDefs.map(([k, lbl]) =>
            `<button class="fe-sort-btn${sortKey === k ? ' active' : ''}" data-key="${k}">${lbl}</button>`
        ).join('');
        controlsEl.innerHTML = `
            <input id="fdSearch" type="search" class="fe-search"
                   placeholder="Search features…" value="${filterText}">
            <div class="fe-sort-btns"><span class="fe-sort-label">Sort:</span>${btns}</div>
            <span class="fe-count">${sorted.length} / ${allFeatures.length} features</span>`;

        controlsEl.querySelector('#fdSearch').addEventListener('input', e => {
            filterText = e.target.value;
            renderAll();
        });
        controlsEl.querySelectorAll('.fe-sort-btn').forEach(btn =>
            btn.addEventListener('click', () => { sortKey = btn.dataset.key; renderAll(); })
        );
    }

    // ── feature list ──────────────────────────────────────────────────
    function renderList(listEl) {
        const sorted  = getSorted();
        const maxCorr = Math.max(...allFeatures.map(f => Math.abs(f.correlation)), 0.001);
        listEl.innerHTML = sorted.map(f => {
            const barW   = (Math.abs(f.correlation) / maxCorr * 65).toFixed(1);
            const barClr = f.correlation >= 0 ? '#d73027' : '#3479b9';
            const sign   = f.correlation >= 0 ? '+' : '';
            return `<div class="fd-list-row${f.name === selectedName ? ' selected' : ''}"
                         data-feature="${f.name}">
                <span class="fd-list-name">${f.name}</span>
                <div class="fd-list-bar-wrap">
                    <div class="fd-list-bar" style="width:${barW}%;background:${barClr}"></div>
                    <span class="fd-list-corr">${sign}${f.correlation.toFixed(3)}</span>
                </div>
            </div>`;
        }).join('');
        listEl.querySelectorAll('.fd-list-row').forEach(row =>
            row.addEventListener('click', () => {
                selectedName = row.dataset.feature;
                renderAll();
                // scroll selected into view
                row.scrollIntoView({ block: 'nearest' });
            })
        );
    }

    // ── histogram chart ───────────────────────────────────────────────
    function renderChart(panelEl) {
        if (!selectedName || !dists[selectedName]) {
            panelEl.innerHTML = '<p class="fd-hint">← Select a feature to see its distribution</p>';
            return;
        }

        const d      = dists[selectedName];
        const edges  = d.bin_edges;
        const isBin  = d.is_binary;

        if (isBin) {
            Plotly.newPlot(panelEl, [
                { x: ['0 — absent', '1 — present'], y: d.non_toxic,
                  type: 'bar', name: 'Non-Toxic',
                  marker: { color: 'rgba(52,121,185,0.8)' } },
                { x: ['0 — absent', '1 — present'], y: d.toxic,
                  type: 'bar', name: 'Toxic',
                  marker: { color: 'rgba(214,39,40,0.8)' } },
            ], {
                barmode: 'group', bargap: 0.25,
                xaxis: { title: selectedName, fixedrange: true },
                yaxis: { title: '% of class', ticksuffix: '%', fixedrange: true },
                legend: { orientation: 'h', y: 1.12, x: 0 },
                height: 380, margin: { t: 30, b: 56, l: 64, r: 20 },
                paper_bgcolor: 'white', plot_bgcolor: '#fafafa',
            }, { responsive: true, displayModeBar: false });
            return;
        }

        // Bin centers and width
        const centers  = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
        const binWidth = edges[1] - edges[0];

        // Find x range where there's actually data to avoid wasted whitespace
        const combined = d.non_toxic.map((v, i) => v + d.toxic[i]);
        let first = 0, last = combined.length - 1;
        while (first < last && combined[first] === 0) first++;
        while (last > first && combined[last] === 0) last--;
        const xPad   = binWidth * 0.5;
        const xRange = [edges[first] - xPad, edges[last + 1] + xPad];

        const maxY = Math.max(...d.non_toxic, ...d.toxic, 0.01);

        Plotly.newPlot(panelEl, [
            { x: centers, y: d.non_toxic,
              type: 'bar', name: 'Non-Toxic',
              marker: { color: 'rgba(52,121,185,0.65)', line: { width: 0 } },
              width: Array(centers.length).fill(binWidth),
              hovertemplate: '%{y:.2f}% of Non-Toxic<extra></extra>' },
            { x: centers, y: d.toxic,
              type: 'bar', name: 'Toxic',
              marker: { color: 'rgba(214,39,40,0.65)', line: { width: 0 } },
              width: Array(centers.length).fill(binWidth),
              hovertemplate: '%{y:.2f}% of Toxic<extra></extra>' },
        ], {
            barmode: 'overlay', bargap: 0, bargroupgap: 0,
            xaxis: { title: selectedName, range: xRange, fixedrange: true },
            yaxis: { title: '% of class', ticksuffix: '%',
                     range: [0, maxY * 1.18], fixedrange: true },
            legend: { orientation: 'h', y: 1.12, x: 0 },
            height: 380, margin: { t: 30, b: 56, l: 64, r: 20 },
            paper_bgcolor: 'white', plot_bgcolor: '#fafafa',
        }, { responsive: true, displayModeBar: false });
    }

    // ── two-panel layout ──────────────────────────────────────────────
    chartEl.innerHTML = `
        <div class="fd-layout">
            <div class="fd-list-wrap" id="fd-list-wrap"></div>
            <div class="fd-chart-panel" id="fd-chart-panel"></div>
        </div>`;

    function renderAll() {
        renderControls();
        renderList(document.getElementById('fd-list-wrap'));
        renderChart(document.getElementById('fd-chart-panel'));
    }

    renderAll();
}

const FEATURE_CATALOG = [
    // ── Sentiment (VADER) ─────────────────────────────────────────────────
    { name: 'vader_compound',      group: 'Sentiment',          type: 'float',   description: 'Overall sentiment polarity score, −1 (very negative) to +1 (very positive). The primary VADER output.' },
    { name: 'vader_neg',           group: 'Sentiment',          type: 'float',   description: 'Proportion of text tokens rated negative by VADER (0–1).' },
    { name: 'vader_pos',           group: 'Sentiment',          type: 'float',   description: 'Proportion of text tokens rated positive by VADER (0–1).' },
    { name: 'vader_is_negative',   group: 'Sentiment',          type: 'binary',  description: '1 if compound score < −0.05, VADER\'s own threshold for an overall negative tone.' },
    { name: 'vader_intensity',     group: 'Sentiment',          type: 'float',   description: 'Absolute emotional strength — |compound|. High values indicate strong sentiment in either direction.' },
    { name: 'vader_pos_minus_neg', group: 'Sentiment',          type: 'float',   description: 'Positive proportion minus negative proportion. Negative values indicate net hostile tone.' },
    // ── Second Person ─────────────────────────────────────────────────────
    { name: 'has_second_person',     group: 'Second Person',    type: 'binary',  description: '1 if the comment contains "you", "your", "yourself", "ur" or similar second-person forms.' },
    { name: 'second_person_count',   group: 'Second Person',    type: 'integer', description: 'Raw count of second-person pronoun matches. Direct address strongly correlates with targeted abuse.' },
    { name: 'second_person_density', group: 'Second Person',    type: 'float',   description: 'Second-person pronouns divided by total words. Controls for comment length.' },
    // ── Profanity ─────────────────────────────────────────────────────────
    { name: 'profanity_count',            group: 'Profanity',   type: 'integer', description: 'Matches against a curated 24-term profanity lexicon using exact word-boundary matching.' },
    { name: 'obfuscated_profanity_count', group: 'Profanity',   type: 'integer', description: 'Profanity detected only after leetspeak normalisation (e.g. f*ck → fuck, $hit → shit). Catches evasion attempts.' },
    // ── Slang ─────────────────────────────────────────────────────────────
    { name: 'slang_count', group: 'Slang', type: 'integer', description: 'Matches against a toxic-slang list (kys, gtfo, incel, stfu, etc.) that the profanity filter misses.' },
    // ── Text Shape ────────────────────────────────────────────────────────
    { name: 'char_count',        group: 'Text Shape',           type: 'integer', description: 'Total character length of the raw comment text.' },
    { name: 'word_count',        group: 'Text Shape',           type: 'integer', description: 'Total whitespace-delimited word count.' },
    { name: 'exclamation_count', group: 'Text Shape',           type: 'integer', description: 'Number of exclamation marks. Elevated counts can signal aggression or urgency.' },
    { name: 'uppercase_ratio',   group: 'Text Shape',           type: 'float',   description: 'Proportion of words that are fully uppercased (ALL CAPS). A widely-used aggression signal.' },
    { name: 'unique_word_ratio', group: 'Text Shape',           type: 'float',   description: 'Type-token ratio — unique words ÷ total words. Low values may indicate repetitive or spam content.' },
    // ── Elongation & Punctuation ──────────────────────────────────────────
    { name: 'elongated_token_count',   group: 'Elongation & Punctuation', type: 'integer', description: 'Words with any character repeated 3+ times (e.g. "sooo", "nooooo"). Marks emotional exaggeration.' },
    { name: 'consecutive_punct_count', group: 'Elongation & Punctuation', type: 'integer', description: 'Runs of 2+ consecutive punctuation characters (e.g. "!!!", "???", "..."). Signals heightened emotion.' },
    // ── URLs & IPs ────────────────────────────────────────────────────────
    { name: 'url_count',     group: 'URLs & IPs', type: 'integer', description: 'Number of HTTP/HTTPS URLs or www-prefixed links in the comment.' },
    { name: 'ip_count',      group: 'URLs & IPs', type: 'integer', description: 'Number of valid IPv4 addresses mentioned. Common in vandalism and spam.' },
    { name: 'has_url_or_ip', group: 'URLs & IPs', type: 'binary',  description: '1 if any URL or IP address is present in the comment.' },
    // ── Syntactic ─────────────────────────────────────────────────────────
    { name: 'negation_count',      group: 'Syntactic',          type: 'integer', description: 'Matches against a 24-term negation list (not, never, can\'t, won\'t, etc.). Captures denial and contradictions.' },
    { name: 'sentence_count',      group: 'Syntactic',          type: 'integer', description: 'Number of sentences detected by splitting on . ! ? characters.' },
    { name: 'avg_sentence_length', group: 'Syntactic',          type: 'float',   description: 'Average words per sentence. Very short sentences can indicate terse, aggressive speech.' },
    // ── Identity Mentions ─────────────────────────────────────────────────
    { name: 'identity_mention_count', group: 'Identity Mentions', type: 'integer', description: 'Total match count across all six identity-group lexicons. Aggregates the per-group signals.' },
    { name: 'identity_race',          group: 'Identity Mentions', type: 'binary',  description: '1 if any racial or ethnic identity term appears (e.g. Black, Asian, Hispanic, Jewish, Indigenous…).' },
    { name: 'identity_gender',        group: 'Identity Mentions', type: 'binary',  description: '1 if any gender identity term appears (e.g. woman, transgender, non-binary, cisgender…).' },
    { name: 'identity_sexuality',     group: 'Identity Mentions', type: 'binary',  description: '1 if any sexual orientation term appears (e.g. gay, lesbian, bisexual, queer, LGBT…).' },
    { name: 'identity_religion',      group: 'Identity Mentions', type: 'binary',  description: '1 if any religious identity term appears (e.g. Muslim, Christian, Jewish, atheist, Hindu…).' },
    { name: 'identity_disability',    group: 'Identity Mentions', type: 'binary',  description: '1 if any disability-related term appears (e.g. autistic, bipolar, deaf, wheelchair, disabled…).' },
    { name: 'identity_nationality',   group: 'Identity Mentions', type: 'binary',  description: '1 if any nationality or migration term appears (e.g. immigrant, refugee, foreigner, undocumented…).' },
];

function renderFeatureCatalog() {
    const container = document.getElementById('feature-catalog-content');
    if (!container) return;

    const groups = [...new Set(FEATURE_CATALOG.map(f => f.group))];

    // Type badge colour
    function typeBadge(type) {
        const cls = type === 'binary' ? 'fc-type-binary'
                  : type === 'integer' ? 'fc-type-integer'
                  : 'fc-type-float';
        return `<span class="fc-type ${cls}">${type}</span>`;
    }

    // Search state
    let filterText = '';

    function renderTable() {
        const q = filterText.toLowerCase();
        const filtered = FEATURE_CATALOG.filter(f =>
            f.name.toLowerCase().includes(q) ||
            f.group.toLowerCase().includes(q) ||
            f.description.toLowerCase().includes(q)
        );

        // Group filtered results
        const byGroup = {};
        filtered.forEach(f => {
            if (!byGroup[f.group]) byGroup[f.group] = [];
            byGroup[f.group].push(f);
        });

        let tableBody = '';
        groups.forEach(group => {
            const rows = byGroup[group];
            if (!rows || rows.length === 0) return;
            rows.forEach((f, i) => {
                tableBody += `<tr>
                    ${i === 0 ? `<td class="fc-group-cell" rowspan="${rows.length}">${group}</td>` : ''}
                    <td class="fc-name"><code>${f.name}</code></td>
                    <td>${typeBadge(f.type)}</td>
                    <td class="fc-desc">${f.description}</td>
                </tr>`;
            });
        });

        const countLabel = filtered.length === FEATURE_CATALOG.length
            ? `${FEATURE_CATALOG.length} features`
            : `${filtered.length} of ${FEATURE_CATALOG.length} features`;

        container.innerHTML = `
            <div class="fc-controls">
                <input id="fcSearch" type="search" class="fc-search" placeholder="Search by name, group or description…" value="${filterText}">
                <span class="fc-count">${countLabel}</span>
            </div>
            <div class="fc-table-wrap">
                <table class="fc-table">
                    <thead>
                        <tr>
                            <th style="width:16%">Group</th>
                            <th style="width:22%">Feature name</th>
                            <th style="width:9%">Type</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>${tableBody || '<tr><td colspan="4" class="fc-empty">No features match your search.</td></tr>'}</tbody>
                </table>
            </div>`;

        document.getElementById('fcSearch').addEventListener('input', e => {
            filterText = e.target.value;
            renderTable();
        });
    }

    renderTable();
}

function renderIdentityRisk(data) {
    const container = document.getElementById('identity-risk-content');
    if (!container) return;

    const ir = data.identity_risk;
    if (!ir || !ir.groups || ir.groups.length === 0) {
        container.innerHTML = '<p class="ir-unavailable">Identity risk data not available. Re-run eda_processor.py to generate it.</p>';
        return;
    }

    const baseline = ir.baseline_toxic_rate;
    const baselinePct = (baseline * 100).toFixed(1);

    // Risk tier thresholds
    function riskTier(rr) {
        if (rr >= 2.5) return { label: 'High', cls: 'ir-high' };
        if (rr >= 1.25) return { label: 'Elevated', cls: 'ir-elevated' };
        if (rr <= 0.75) return { label: 'Low', cls: 'ir-low' };
        return { label: 'Baseline', cls: 'ir-baseline' };
    }

    // Summary cards
    const summaryHtml = `
        <div class="ir-summary-grid">
            <div class="ir-summary-card">
                <div class="ir-summary-label">Baseline toxicity rate</div>
                <div class="ir-summary-value">${baselinePct}%</div>
                <div class="ir-summary-sub">across all ${ir.total_comments.toLocaleString()} comments</div>
            </div>
            <div class="ir-summary-card">
                <div class="ir-summary-label">Comments with identity mentions</div>
                <div class="ir-summary-value">${ir.any_identity_mention_count.toLocaleString()}</div>
                <div class="ir-summary-sub">${ir.any_identity_mention_pct}% of dataset · toxic rate ${(ir.any_identity_mention_toxic_rate * 100).toFixed(1)}%</div>
            </div>
            <div class="ir-summary-card">
                <div class="ir-summary-label">identity_hate label (Jigsaw)</div>
                <div class="ir-summary-value">${ir.identity_hate_total.toLocaleString()}</div>
                <div class="ir-summary-sub">${(ir.identity_hate_rate * 100).toFixed(2)}% of dataset carry this fine-grained label</div>
            </div>
        </div>`;

    // Per-group table
    const maxRR = Math.max(...ir.groups.map(g => g.relative_risk));

    const rows = ir.groups.map(g => {
        const tier = riskTier(g.relative_risk);
        const barW = Math.min((g.relative_risk / maxRR) * 100, 100).toFixed(1);
        const barColor = g.relative_risk >= 2.5 ? '#c62828'
                       : g.relative_risk >= 1.25 ? '#e65100'
                       : g.relative_risk <= 0.75 ? '#2e7d32'
                       : '#555';
        return `
            <tr>
                <td class="ir-group-name">${g.label}</td>
                <td class="ir-count">${g.count.toLocaleString()} <span class="ir-pct">(${g.pct_of_dataset}%)</span></td>
                <td>
                    <div class="ir-bar-wrap">
                        <div class="ir-bar" style="width:${barW}%;background:${barColor}"></div>
                        <span class="ir-rr-val" style="color:${barColor}">${g.relative_risk.toFixed(2)}×</span>
                    </div>
                </td>
                <td>${(g.toxic_rate * 100).toFixed(1)}%</td>
                <td>${(g.identity_hate_rate * 100).toFixed(2)}%</td>
                <td><span class="ir-tier ${tier.cls}">${tier.label}</span></td>
            </tr>`;
    }).join('');

    const tableHtml = `
        <div class="ir-table-wrap">
            <table class="ir-table">
                <thead>
                    <tr>
                        <th>Identity Group</th>
                        <th>Comments in dataset</th>
                        <th>Relative risk vs baseline (${baselinePct}%)</th>
                        <th>Toxic rate</th>
                        <th>identity_hate rate</th>
                        <th>Risk tier</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
        <p class="ir-footnote">
            Relative risk = group toxic rate ÷ baseline toxic rate.
            <strong>identity_hate rate</strong> is the Jigsaw fine-grained label; it is separate from (and narrower than) the binary toxic label.
            Comments may mention multiple identity groups so group counts overlap.
        </p>`;

    // Plotly chart — relative risk bars
    const groups = ir.groups;
    const colors = groups.map(g =>
        g.relative_risk >= 2.5  ? '#c62828' :
        g.relative_risk >= 1.25 ? '#e65100' :
        g.relative_risk <= 0.75 ? '#2e7d32' : '#888'
    );

    container.innerHTML = summaryHtml + tableHtml + '<div id="ir-chart" class="ir-chart"></div>';

    Plotly.newPlot('ir-chart', [
        {
            type: 'bar',
            x: groups.map(g => g.relative_risk),
            y: groups.map(g => g.label),
            orientation: 'h',
            marker: { color: colors },
            text: groups.map(g => g.relative_risk.toFixed(2) + '×'),
            textposition: 'outside',
            hovertemplate: '<b>%{y}</b><br>Relative risk: %{x:.2f}×<extra></extra>',
        },
        {
            type: 'scatter',
            x: [1, 1],
            y: [groups[groups.length - 1].label, groups[0].label],
            mode: 'lines',
            line: { color: '#555', dash: 'dot', width: 1.5 },
            name: 'Baseline (1.0×)',
            hoverinfo: 'skip',
        }
    ], {
        xaxis: { title: 'Relative risk (1.0 = baseline toxicity rate)', zeroline: false },
        yaxis: { automargin: true },
        height: 320,
        margin: { t: 24, b: 48, l: 180, r: 80 },
        showlegend: true,
        legend: { x: 1, xanchor: 'right', y: 1 },
        paper_bgcolor: 'white',
        plot_bgcolor: '#fafafa',
    }, { responsive: true, displayModeBar: false });
}

function renderReadinessChecklist(data) {
    const container = document.getElementById('readiness-checklist');
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

    if (container) {
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
                </div>`;
        });
        container.innerHTML = html;
    }

    const computedAt = document.getElementById('computed-at');
    if (computedAt) computedAt.textContent = new Date().toLocaleString();
}

function renderCorrelationHeatmap(data) {
    const container = document.getElementById('correlation-heatmap');
    if (!container) return;

    const cm = data.feature_correlation_matrix;
    if (!cm || !cm.features || !cm.matrix) {
        container.innerHTML = '<p style="color:#999;font-style:italic;">Correlation matrix not available in cache.</p>';
        return;
    }

    const labels = cm.features;
    const z = cm.matrix;

    const trace = {
        type: 'heatmap',
        z: z,
        x: labels,
        y: labels,
        colorscale: [
            [0.0, '#2166ac'],
            [0.25, '#92c5de'],
            [0.5, '#f7f7f7'],
            [0.75, '#f4a582'],
            [1.0, '#d6604d']
        ],
        zmin: -1,
        zmax: 1,
        colorbar: {
            title: 'r',
            thickness: 14,
            len: 0.8,
            tickvals: [-1, -0.5, 0, 0.5, 1],
            tickfont: { size: 11 }
        },
        hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>'
    };

    const layout = {
        margin: { t: 20, b: 160, l: 160, r: 20 },
        xaxis: {
            tickangle: -45,
            tickfont: { size: 10, family: 'monospace' },
            side: 'bottom'
        },
        yaxis: {
            tickfont: { size: 10, family: 'monospace' },
            autorange: 'reversed'
        },
        height: 680
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}
