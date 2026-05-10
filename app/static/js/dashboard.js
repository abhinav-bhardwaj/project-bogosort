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
    renderComparison(data);
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

  let labels = ["Pred 0", "Pred 1"];
  let matrix = matrixPayload;

  if (!Array.isArray(matrixPayload)) {
    labels = matrixPayload.labels || labels;
    matrix = matrixPayload.matrix || [];
  }

  if (!matrix || matrix.length === 0 || !matrix[0] || !matrix[1]) {
    container.innerHTML = "<p>No confusion matrix available.</p>";
    return;
  }

  let html = '<table class="confusion-matrix"><thead><tr><th></th>';
  html += `<th>${labels[0]}</th><th>${labels[1]}</th></tr></thead><tbody>`;
  html += `<tr><th>${labels[0]}</th><td>${matrix[0][0] || 0}</td><td>${matrix[0][1] || 0}</td></tr>`;
  html += `<tr><th>${labels[1]}</th><td>${matrix[1][0] || 0}</td><td>${matrix[1][1] || 0}</td></tr>`;
  html += "</tbody></table>";

  container.innerHTML = html;
}

function renderRoc(curve, imageUrl) {
  const container = document.getElementById("rocContainer");
  const canvas = document.getElementById("rocChart");
  
  if (!container) return;
  
  if (imageUrl) {
    container.innerHTML = `<img src="${imageUrl}" alt="ROC curve" style="max-width: 100%; height: auto;">`;
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
        datasets: [{
          label: "ROC",
          data: curve.tpr,
          borderColor: "#3366cc",
          tension: 0.4,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: true }
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
    container.innerHTML = `<img src="${imageUrl}" alt="PR curve" style="max-width: 100%; height: auto;">`;
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
          label: "PR",
          data: curve.precision,
          borderColor: "#c43b3b",
          tension: 0.4,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: true }
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
    canvas.parentElement.innerHTML = "<p>No perturbation data available.</p>";
    return;
  }

  try {
    if (perturbChart) perturbChart.destroy();
    perturbChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: data.map(d => d.feature),
        datasets: [{
          label: "Importance",
          data: data.map(d => d.importance),
          backgroundColor: "#6a5acd"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: true }
        }
      }
    });
  } catch (err) {
    console.error("Error rendering perturbation chart:", err);
    canvas.parentElement.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
  }
}

function renderComparison(data) {
  const canvas = document.getElementById("compareChart");
  if (!canvas) return;
  
  if (!data || !data.metrics || Object.keys(data.metrics).length === 0) {
    canvas.parentElement.innerHTML = "<p>No comparison data available.</p>";
    return;
  }

  try {
    if (compareChart) compareChart.destroy();
    compareChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: Object.keys(data.metrics),
        datasets: [{
          label: "Metrics",
          data: Object.values(data.metrics),
          backgroundColor: "#3b6fc4"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: true }
        }
      }
    });
  } catch (err) {
    console.error("Error rendering comparison chart:", err);
    canvas.parentElement.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
  }
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