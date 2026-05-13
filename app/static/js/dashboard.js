let rocChart, prChart, perturbChart, compareChart;

async function loadEvaluation() {
  const select = document.getElementById("modelSelect");
  if (!select || !select.value) {
    const metricsGrid = document.getElementById("metricsGrid");
    if (metricsGrid) {
      metricsGrid.textContent = "No models available.";
    }
    return;
  }

  const modelId = select.value;
  try {
    const res = await fetch(`/api/models/${modelId}/evaluation`);
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const data = await res.json();

    renderMetrics(data.metrics || {});
    renderConfusionMatrix(data.confusion_matrix || []);
    const rocImage = data.roc_curve?.image_url || data.roc_image_url || data.artifacts?.images?.roc_curve;
    const prImage = data.pr_curve?.image_url || data.pr_image_url || data.artifacts?.images?.pr_curve;
    renderRoc(data.roc_curve || {}, rocImage);
    renderPr(data.pr_curve || {}, prImage);
    renderPerturbation(data.perturbation || []);
    renderArtifacts(data.artifacts || {}, data);
  } catch (err) {
    const metricsGrid = document.getElementById("metricsGrid");
    if (metricsGrid) {
      metricsGrid.textContent = `Failed to load evaluation: ${err.message}`;
    }
  }
}

async function initModelSelect() {
  const select = document.getElementById("modelSelect");
  if (!select) return;

  try {
    const res = await fetch("/api/models");
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const data = await res.json();
    select.innerHTML = "";
    
    const models = data.models || [];
    if (models.length === 0) {
      throw new Error("No models available");
    }
    
    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = model.model_name;
      select.appendChild(option);
    });
    
    if (select.options.length > 0) {
      select.value = select.options[0].value;
      await loadEvaluation();
    }
    loadComparison(models);
  } catch (err) {
    select.innerHTML = `<option value="">Error: ${err.message}</option>`;
    const metricsGrid = document.getElementById("metricsGrid");
    if (metricsGrid) {
      metricsGrid.textContent = `Failed to load models: ${err.message}`;
    }
    console.error("Model loading error:", err);
  }
}

function renderMetrics(metrics) {
  const grid = document.getElementById("metricsGrid");
  if (!grid) return;
  grid.innerHTML = "";
  Object.entries(metrics).forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "metric-box";
    card.innerHTML = `<h3>${key}</h3><p>${value}</p>`;
    grid.appendChild(card);
  });
}

function renderConfusionMatrix(matrixPayload) {
  const container = document.getElementById("confusionMatrix");
  if (!container) return;

  if (!matrixPayload || (Array.isArray(matrixPayload) && matrixPayload.length === 0)) {
    container.innerHTML = "<p>No confusion matrix available.</p>";
    return;
  }

  let matrix = Array.isArray(matrixPayload) ? matrixPayload : (matrixPayload.matrix || []);

  if (!matrix || matrix.length < 2 || !matrix[0] || !matrix[1]) {
    container.innerHTML = "<p>No confusion matrix available.</p>";
    return;
  }

  const tn = matrix[0][0] || 0;
  const fp = matrix[0][1] || 0;
  const fn = matrix[1][0] || 0;
  const tp = matrix[1][1] || 0;

  // Row-normalised percentages (% of each actual class correctly/incorrectly classified)
  const row0 = tn + fp;
  const row1 = fn + tp;
  const tnPct = row0 > 0 ? (tn / row0) * 100 : 0;
  const fpPct = row0 > 0 ? (fp / row0) * 100 : 0;
  const fnPct = row1 > 0 ? (fn / row1) * 100 : 0;
  const tpPct = row1 > 0 ? (tp / row1) * 100 : 0;

  function cell(count, pct, good, tooltip) {
    const alpha = (pct / 100) * 0.85;
    const bg = good ? `rgba(51,102,204,${alpha})` : `rgba(211,51,51,${alpha})`;
    const fg = pct > 55 ? "white" : "var(--wp-text)";
    return `<td class="cm-cell" style="background:${bg};color:${fg}" title="${tooltip}">
      <div class="cm-pct">${pct.toFixed(1)}%</div>
      <div class="cm-count">${count.toLocaleString()}</div>
    </td>`;
  }

  container.innerHTML = `
    <div class="cm-wrap">
      <div class="cm-col-label">Predicted →</div>
      <table class="confusion-matrix">
        <thead>
          <tr>
            <th class="cm-corner"></th>
            <th class="cm-col-head">Non-toxic</th>
            <th class="cm-col-head">Toxic</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th class="cm-row-head"><span class="cm-row-label">Actual</span>Non-toxic</th>
            ${cell(tn, tnPct, true,  "True Negative — correctly kept as non-toxic")}
            ${cell(fp, fpPct, false, "False Positive — non-toxic comment wrongly flagged")}
          </tr>
          <tr>
            <th class="cm-row-head"><span class="cm-row-label">Actual</span>Toxic</th>
            ${cell(fn, fnPct, false, "False Negative — toxic comment missed by model")}
            ${cell(tp, tpPct, true,  "True Positive — correctly identified as toxic")}
          </tr>
        </tbody>
      </table>
      <p class="cm-note">Percentages are row-normalised (share of each actual class). Raw counts shown below each value.</p>
      <div class="cm-legend">
        <span class="cm-leg-item"><span class="cm-leg-swatch" style="background:rgba(51,102,204,0.72)"></span>Correct predictions</span>
        <span class="cm-leg-item"><span class="cm-leg-swatch" style="background:rgba(211,51,51,0.72)"></span>Errors</span>
      </div>
    </div>`;
}

function renderRoc(curve, imageUrl) {
  const container = document.getElementById("rocContainer");
  const canvas = document.getElementById("rocChart");

  if (!container) return;

  if (imageUrl) {
    container.style.height = "auto";
    container.innerHTML = `<img src="${imageUrl}" alt="ROC curve">`;
    return;
  }
  
  if (!canvas) {
    container.innerHTML = "<p>Chart container not found.</p>";
    return;
  }

  if (!curve || !curve.fpr || !curve.tpr || curve.fpr.length === 0) {
    container.innerHTML = "<p>No ROC curve data available.</p>";
    return;
  }

  try {
    if (rocChart) rocChart.destroy();
    rocChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: curve.fpr,
        datasets: [
          {
            label: "ROC Curve",
            data: curve.tpr,
            borderColor: "#3366cc",
            backgroundColor: "rgba(51,102,204,0.08)",
            tension: 0.3,
            fill: true,
            pointRadius: 0,
          },
          {
            label: "Random classifier",
            data: curve.fpr,
            borderColor: "#aaa",
            borderDash: [5, 5],
            tension: 0,
            fill: false,
            pointRadius: 0,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: "bottom" },
          tooltip: { mode: "index", intersect: false },
        },
        scales: {
          x: { title: { display: true, text: "False Positive Rate" }, ticks: { maxTicksLimit: 6 } },
          y: { title: { display: true, text: "True Positive Rate" }, min: 0, max: 1 },
        }
      }
    });
  } catch (err) {
    console.error("Error rendering ROC chart:", err);
    container.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
  }
}

function renderPr(curve, imageUrl) {
  const container = document.getElementById("prContainer");
  const canvas = document.getElementById("prChart");

  if (!container) return;

  if (imageUrl) {
    container.style.height = "auto";
    container.innerHTML = `<img src="${imageUrl}" alt="PR curve">`;
    return;
  }
  
  if (!canvas) {
    container.innerHTML = "<p>Chart container not found.</p>";
    return;
  }

  if (!curve || !curve.precision || !curve.recall || curve.precision.length === 0) {
    container.innerHTML = "<p>No PR curve data available.</p>";
    return;
  }

  try {
    if (prChart) prChart.destroy();
    prChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: curve.recall,
        datasets: [{
          label: "Precision-Recall",
          data: curve.precision,
          borderColor: "#c43b3b",
          backgroundColor: "rgba(196,59,59,0.08)",
          tension: 0.3,
          fill: true,
          pointRadius: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: "bottom" },
          tooltip: { mode: "index", intersect: false },
        },
        scales: {
          x: { title: { display: true, text: "Recall" }, min: 0, max: 1, ticks: { maxTicksLimit: 6 } },
          y: { title: { display: true, text: "Precision" }, min: 0, max: 1 },
        }
      }
    });
  } catch (err) {
    console.error("Error rendering PR chart:", err);
    container.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
  }
}

function renderPerturbation(data) {
  const canvas = document.getElementById("perturbChart");
  if (!canvas) return;

  if (!data || data.length === 0) {
    const p = canvas.parentElement;
    p.style.height = "auto";
    p.innerHTML = "<p>No perturbation data available.</p>";
    return;
  }

  try {
    if (perturbChart) perturbChart.destroy();
    perturbChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: data.map(d => d.feature),
        datasets: [{
          label: "Permutation Importance",
          data: data.map(d => d.importance),
          backgroundColor: data.map(d => d.importance >= 0 ? "#6a5acd" : "#c43b3b"),
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(4)}` } },
        },
        scales: {
          x: { title: { display: true, text: "Accuracy drop when feature is permuted" } },
          y: { ticks: { font: { family: "monospace", size: 11 } } },
        }
      }
    });
  } catch (err) {
    console.error("Error rendering perturbation chart:", err);
    canvas.parentElement.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
  }
}

const COMPARE_METRICS = [
  { key: "accuracy",  label: "Accuracy",  default: true  },
  { key: "f1",        label: "F1 Score",  default: true  },
  { key: "precision", label: "Precision", default: true  },
  { key: "recall",    label: "Recall",    default: true  },
  { key: "roc_auc",   label: "ROC AUC",   default: false },
  { key: "pr_auc",    label: "PR AUC",    default: false },
];

const MODEL_COLORS = [
  "rgba(59,111,196,0.82)",
  "rgba(44,160,44,0.82)",
  "rgba(230,118,46,0.82)",
  "rgba(148,103,189,0.82)",
  "rgba(214,39,40,0.82)",
];

function loadComparison(models) {
  if (!models || models.length === 0) return;

  const activeMetrics = new Set(COMPARE_METRICS.filter(m => m.default).map(m => m.key));

  function rebuildChart() {
    const canvas = document.getElementById("compareChart");
    if (!canvas) return;

    const selected = COMPARE_METRICS.filter(m => activeMetrics.has(m.key));
    // X-axis = metric groups; within each group one bar per model
    const labels = selected.map(m => m.label);

    const datasets = models.map((model, i) => ({
      label: model.model_name,
      data: selected.map(m => model.metrics?.[m.key] ?? null),
      backgroundColor: MODEL_COLORS[i % MODEL_COLORS.length],
      borderRadius: 3,
      borderSkipped: false,
    }));

    try {
      if (compareChart) compareChart.destroy();
      compareChart = new Chart(canvas, {
        type: "bar",
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: "bottom",
              labels: { boxWidth: 12, padding: 14, font: { size: 12 } },
            },
            tooltip: {
              callbacks: {
                label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(4)}`,
              },
            },
          },
          scales: {
            x: { ticks: { font: { size: 12 } } },
            y: {
              min: 0, max: 1,
              title: { display: true, text: "Score" },
              ticks: { callback: v => v.toFixed(2) },
            },
          },
        },
      });
    } catch (err) {
      console.error("Comparison chart error:", err);
    }
  }

  function renderToggles() {
    const wrap = document.getElementById("compareMetricToggles");
    if (!wrap) return;
    wrap.innerHTML = COMPARE_METRICS.map(m =>
      `<button class="compare-toggle${activeMetrics.has(m.key) ? " active" : ""}"
               data-key="${m.key}">${m.label}</button>`
    ).join("");
    wrap.querySelectorAll(".compare-toggle").forEach(btn => {
      btn.addEventListener("click", () => {
        const k = btn.dataset.key;
        if (activeMetrics.has(k)) {
          if (activeMetrics.size > 1) activeMetrics.delete(k);
        } else {
          activeMetrics.add(k);
        }
        renderToggles();
        rebuildChart();
        renderCompareTable(models);
      });
    });
  }

  renderToggles();
  rebuildChart();
  renderCompareTable(models);
}

function renderCompareTable(models) {
  const container = document.getElementById("compareTable");
  if (!container || !models || models.length === 0) return;

  const allMetrics = COMPARE_METRICS;

  // Find best value per metric column
  const best = {};
  allMetrics.forEach(m => {
    best[m.key] = Math.max(...models.map(mo => mo.metrics?.[m.key] ?? 0));
  });

  const headerCells = allMetrics.map(m => `<th>${m.label}</th>`).join("");
  const rows = models.map((model, i) => {
    const swatch = `<span class="ct-swatch" style="background:${MODEL_COLORS[i % MODEL_COLORS.length]}"></span>`;
    const cells = allMetrics.map(m => {
      const v = model.metrics?.[m.key];
      if (v == null) return `<td>—</td>`;
      const isBest = Math.abs(v - best[m.key]) < 0.00001;
      return `<td class="${isBest ? "ct-best" : ""}">${v.toFixed(4)}</td>`;
    }).join("");
    return `<tr><td class="ct-name">${swatch}${model.model_name}</td>${cells}</tr>`;
  }).join("");

  container.innerHTML = `
    <div class="ct-wrap">
      <table class="ct-table">
        <thead><tr><th>Model</th>${headerCells}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="ct-note">Bold = best score in column across all models.</p>
    </div>`;
}

document.addEventListener("DOMContentLoaded", initModelSelect);

function renderArtifacts(artifacts, data) {
  renderImage("calibrationContainer", data.calibration_image_url || artifacts.images?.calibration, "Calibration plot");
  renderImage(
    "featureImportanceContainer",
    data.feature_importance_image_url || artifacts.images?.feature_importance,
    "Feature importance"
  );
  renderImage(
    "errorConfidenceContainer",
    data.error_confidence_distribution_url || artifacts.images?.error_confidence_distribution,
    "Error confidence distribution"
  );
  renderImage(
    "errorPatternsContainer",
    data.error_patterns_by_feature_url || artifacts.images?.error_patterns_by_feature,
    "Error patterns by feature"
  );
  renderTable("falsePositives", artifacts.samples?.false_positives, "No false positives sample.");
  renderTable("falseNegatives", artifacts.samples?.false_negatives, "No false negatives sample.");
  renderTable("errorPatternsTable", artifacts.samples?.error_patterns_by_feature, "No error pattern data.");
}

function renderImage(containerId, url, altText) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!url) {
    container.textContent = "Not available.";
    return;
  }
  container.innerHTML = `<img src="${url}" alt="${altText}" style="max-width: 100%; height: auto;">`;
}

function renderTable(containerId, rows, emptyMessage) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!rows || rows.length === 0) {
    container.textContent = emptyMessage;
    return;
  }
  const headers = Object.keys(rows[0]);
  let html = "<table class=\"detail-table\"><thead><tr>";
  headers.forEach(header => {
    html += `<th>${header}</th>`;
  });
  html += "</tr></thead><tbody>";
  rows.forEach(row => {
    html += "<tr>";
    headers.forEach(header => {
      html += `<td>${row[header]}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  container.innerHTML = html;
}

async function readErrorMessage(res) {
  try {
    const data = await res.json();
    return data.error || res.statusText;
  } catch {
    return res.statusText;
  }
}