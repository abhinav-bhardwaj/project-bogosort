let articleCharts = {};
const DEFAULT_AUTO_THRESHOLD = 0.75;
const DEFAULT_MANUAL_THRESHOLD = 0.55;

const MODEL_META = {
  lasso_log_reg: {
    tag: "Custom-built",
    usesThresholds: true,
    description: "L1-regularised logistic regression. Returns a probability score between 0 and 1 — thresholds control when to auto-ban or flag for review.",
  },
  ridge_log_reg: {
    usesThresholds: true,
    description: "L2-regularised logistic regression. Produces calibrated probability scores suitable for threshold-based decisions.",
  },
  random_forest: {
    usesThresholds: true,
    description: "Ensemble of decision trees. Returns class probabilities via vote averaging — thresholds apply to the blended score.",
  },
  svm: {
    usesThresholds: false,
    description: "Linear SVM that outputs a hard 0/1 class label, not a probability score.",
    thresholdNote: "SVM predicts a binary class directly — not a probability. Threshold sliders have no effect and are disabled while SVM is selected.",
  },
  ensemble: {
    usesThresholds: true,
    description: "Weighted vote of Lasso, Random Forest, and Ridge models. Combines the strengths of each — thresholds apply to the blended probability score.",
  },
};

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("articlesGrid")) {
    initLanding();
  }
  if (document.getElementById("articleDetail")) {
    initArticlePage();
  }
  if (document.getElementById("commentDetail")) {
    initCommentPage();
  }
});

async function initLanding() {
  await loadModels();
  updateLandingThresholdLabels();

  document.getElementById("autoThreshold").addEventListener("input", updateLandingThresholdLabels);
  document.getElementById("manualThreshold").addEventListener("input", updateLandingThresholdLabels);

  document.getElementById("addInput").addEventListener("click", () => {
    const input = document.createElement("input");
    input.type = "url";
    input.className = "article-input";
    input.placeholder = "https://en.wikipedia.org/wiki/Some_Page";
    document.getElementById("articleInputs").appendChild(input);
  });

  document.getElementById("articleForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const inputs = Array.from(document.querySelectorAll("#articleInputs .article-input"))
      .map(i => i.value.trim())
      .filter(Boolean);

    if (!inputs.length) {
      const url = document.getElementById("articleUrl").value.trim();
      if (url) inputs.push(url);
    }

    const limit = parseInt(document.getElementById("limit").value || "30", 10);
    const autoThreshold = parseFloat(document.getElementById("autoThreshold").value || DEFAULT_AUTO_THRESHOLD);
    const manualThreshold = parseFloat(document.getElementById("manualThreshold").value || DEFAULT_MANUAL_THRESHOLD);
    const modelName = document.getElementById("modelSelect").value || "ensemble";

    const statusContainer = document.getElementById("ingestStatus");
    const banner = document.getElementById("analysisBanner");
    statusContainer.innerHTML = "";
    if (banner) banner.hidden = true;

    if (!inputs.length) {
      statusContainer.textContent = "Please enter at least one Wikipedia URL.";
      return;
    }

    const analyzeBtn = document.getElementById("analyzeBtn");
    if (analyzeBtn) { analyzeBtn.disabled = true; analyzeBtn.textContent = "Analysing…"; }

    let successCount = 0;
    let failCount = 0;

    for (const url of inputs) {
      const statusRow = document.createElement("div");
      statusRow.className = "status-item";
      statusRow.textContent = `${url} — analysing…`;
      statusContainer.appendChild(statusRow);
      try {
        const res = await fetch("/api/articles/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, limit, auto_threshold: autoThreshold, manual_threshold: manualThreshold, model_name: modelName })
        });
        if (!res.ok) throw new Error(await readErrorMessage(res));
        statusRow.textContent = `${url} — done`;
        statusRow.classList.add("success");
        successCount++;
      } catch (err) {
        statusRow.textContent = `${url} — failed: ${err.message}`;
        statusRow.classList.add("error");
        failCount++;
      }
    }

    if (analyzeBtn) { analyzeBtn.disabled = false; analyzeBtn.textContent = "Analyse Comments"; }

    if (successCount > 0 && banner) {
      const bannerText = document.getElementById("analysisBannerText");
      const modelLabel = document.getElementById("modelSelect").selectedOptions[0]?.text || modelName;
      bannerText.textContent =
        `${successCount} article${successCount > 1 ? "s" : ""} processed with ${modelLabel}` +
        (failCount ? ` · ${failCount} failed` : "");
      banner.hidden = false;
      setTimeout(() => { banner.hidden = true; }, 10000);
    }

    document.getElementById("thresholdLabel").textContent = manualThreshold.toFixed(2);
    await loadArticles();
  });

  await loadArticles();
}

async function loadModels() {
  const select = document.getElementById("modelSelect");
  if (!select) return;

  try {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error(await readErrorMessage(res));
    const data = await res.json();
    select.innerHTML = "";

    const models = data.models || [];
    if (models.length === 0) throw new Error("No models returned from API");

    models.forEach(model => {
      const opt = document.createElement("option");
      opt.value = model.model_id;
      const meta = MODEL_META[model.model_id];
      opt.textContent = meta?.tag
        ? `${model.model_name} — ${meta.tag}`
        : model.model_name;
      select.appendChild(opt);
    });

    select.addEventListener("change", onModelChange);
    onModelChange();
  } catch (err) {
    select.innerHTML = `<option value="ensemble">Ensemble</option>`;
    console.error("Failed to load models:", err.message);
  }
}

function onModelChange() {
  const select = document.getElementById("modelSelect");
  if (!select) return;
  const meta = MODEL_META[select.value] || {};

  // Model info panel
  const panel = document.getElementById("modelInfoPanel");
  const infoText = document.getElementById("modelInfoText");
  if (panel && infoText) {
    if (meta.description) {
      infoText.textContent = meta.description;
      panel.hidden = false;
    } else {
      panel.hidden = true;
    }
  }

  // Threshold controls
  const usesThresholds = meta.usesThresholds !== false;
  const autoInput    = document.getElementById("autoThreshold");
  const manualInput  = document.getElementById("manualThreshold");
  const section      = document.getElementById("thresholdSection");
  const svmNote      = document.getElementById("svmThresholdNote");
  const svmNoteText  = svmNote?.querySelector("span:last-child");

  if (autoInput)   autoInput.disabled   = !usesThresholds;
  if (manualInput) manualInput.disabled = !usesThresholds;
  if (section)     section.classList.toggle("thresholds-disabled", !usesThresholds);
  if (svmNote)     svmNote.hidden = usesThresholds;
  if (svmNoteText && meta.thresholdNote) svmNoteText.textContent = meta.thresholdNote;
}

async function loadArticles() {
  const grid = document.getElementById("articlesGrid");
  grid.innerHTML = "";
  try {
    const res = await fetch("/api/articles");
    if (!res.ok) throw new Error(await readErrorMessage(res));
    const data = await res.json();
    (data.articles || []).forEach(article => {
      const card = document.createElement("div");
      card.className = "article-card";
      card.innerHTML = `
        <h3>${article.title}</h3>
        <p class="article-url">${article.url}</p>
        <p class="article-flagged">Flagged comments: ${article.flagged_count}</p>
        <div class="article-chart">
          <canvas id="trend-${article.id}"></canvas>
        </div>
        <a class="article-link" href="/articles/${article.id}/">Open details</a>
      `;
      grid.appendChild(card);
      renderTrendChart(article);
    });
  } catch (err) {
    grid.textContent = `Failed to load articles: ${err.message}`;
  }
}

function renderTrendChart(article) {
  const trend = article.trend || { dates: [], scores: [], threshold: 0.55 };
  const ctx = document.getElementById(`trend-${article.id}`).getContext("2d");
  if (articleCharts[article.id]) articleCharts[article.id].destroy();

  articleCharts[article.id] = new Chart(ctx, {
    type: "line",
    data: {
      labels: trend.dates,
      datasets: [
        { label: "Additive Toxicity Score", data: trend.scores, borderColor: "#c43b3b", backgroundColor: "rgba(196,59,59,0.1)", fill: true, tension: 0.3 },
        { label: "Action Threshold", data: trend.dates.map(() => trend.threshold), borderColor: "#3b6fc4", borderDash: [5, 5], fill: false, tension: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { maxTicksLimit: 4 } } }
    }
  });
}

async function initArticlePage() {
  const articleId = document.body.dataset.articleId;
  try {
    const res = await fetch(`/api/articles/${articleId}?include_comments=false`);
    if (!res.ok) throw new Error(await readErrorMessage(res));
    const article = await res.json();

    document.getElementById("articleTitle").textContent = article.title;
    const articleLink = document.getElementById("articleLink");
    articleLink.href = article.url;
    articleLink.textContent = article.url;
    document.getElementById("articleSummary").textContent = article.summary || "";
    document.getElementById("articleError").textContent = "";

    const autoThreshold   = document.getElementById("autoThreshold");
    const manualThreshold = document.getElementById("manualThreshold");
    autoThreshold.value   = article.auto_threshold;
    manualThreshold.value = article.manual_threshold;
    updateThresholdLabels();

    autoThreshold.addEventListener("input",   updateThresholdLabels);
    manualThreshold.addEventListener("input", updateThresholdLabels);

    const queues = {
      all:    { decision: null,            tableId: "allTable",     limitSelect: document.getElementById("allLimit"),    sortSelect: document.getElementById("allSort"),    prevButton: document.getElementById("allPrev"),    nextButton: document.getElementById("allNext"),    pageInfo: document.getElementById("allPageInfo"),    page: 0, total: 0, mode: "all" },
      auto:   { decision: "auto-ban",      tableId: "autoBanTable", limitSelect: document.getElementById("autoLimit"),   sortSelect: document.getElementById("autoSort"),   prevButton: document.getElementById("autoPrev"),   nextButton: document.getElementById("autoNext"),   pageInfo: document.getElementById("autoPageInfo"),   page: 0, total: 0, mode: "auto" },
      manual: { decision: "manual-review", tableId: "manualTable",  limitSelect: document.getElementById("manualLimit"), sortSelect: document.getElementById("manualSort"), prevButton: document.getElementById("manualPrev"), nextButton: document.getElementById("manualNext"), pageInfo: document.getElementById("manualPageInfo"), page: 0, total: 0, mode: "manual" },
    };

    Object.values(queues).forEach(queue => {
      queue.limitSelect.addEventListener("change", async () => { queue.page = 0; await loadQueue(articleId, queue); });
      queue.sortSelect.addEventListener("change",  async () => { queue.page = 0; await loadQueue(articleId, queue); });
      queue.prevButton.addEventListener("click",   async () => { if (queue.page > 0) { queue.page--; await loadQueue(articleId, queue); } });
      queue.nextButton.addEventListener("click",   async () => {
        const limit = parseInt(queue.limitSelect.value, 10);
        if ((queue.page + 1) * limit < queue.total) { queue.page++; await loadQueue(articleId, queue); }
      });
    });

    document.getElementById("applyThresholds").addEventListener("click", async () => {
      document.getElementById("articleError").textContent = "";
      try {
        const res = await fetch(`/api/articles/${articleId}/thresholds`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ auto_threshold: parseFloat(autoThreshold.value), manual_threshold: parseFloat(manualThreshold.value) })
        });
        if (!res.ok) throw new Error(await readErrorMessage(res));
        await Promise.all([loadQueue(articleId, queues.all), loadQueue(articleId, queues.auto), loadQueue(articleId, queues.manual)]);
      } catch (err) {
        document.getElementById("articleError").textContent = `Failed to update thresholds: ${err.message}`;
      }
    });

    await Promise.all([loadQueue(articleId, queues.all), loadQueue(articleId, queues.auto), loadQueue(articleId, queues.manual)]);
  } catch (err) {
    document.getElementById("articleDetail").textContent = `Failed to load article: ${err.message}`;
  }
}

async function loadQueue(articleId, queue) {
  const limit  = parseInt(queue.limitSelect.value, 10);
  const offset = queue.page * limit;
  try {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset), sort: queue.sortSelect.value });
    if (queue.decision) params.set("decision", queue.decision);
    const res = await fetch(`/api/articles/${articleId}/comments?${params.toString()}`);
    if (!res.ok) throw new Error(await readErrorMessage(res));
    const data = await res.json();
    queue.total = data.total || 0;
    renderQueue(queue.tableId, data.comments || [], queue.mode);
    updatePagination(queue, limit);
  } catch (err) {
    renderQueue(queue.tableId, [], queue.mode, err.message);
  }
}

function renderQueue(tableId, rows, mode, errorMessage = null) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";

  if (errorMessage) {
    tbody.innerHTML = `<tr><td colspan="5">Failed to load comments: ${errorMessage}</td></tr>`;
    return;
  }
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="5">No comments to display.</td></tr>`;
    return;
  }

  rows.forEach(row => {
    const tr = document.createElement("tr");
    let decisionCell =
      mode === "auto"   ? `<span class="badge danger">Auto-ban</span>` :
      mode === "manual" ? `<button class="btn small danger">Ban</button> <button class="btn small">Not ban</button> <span class="badge neutral">Pending</span>` :
      row.decision === "auto-ban"      ? `<span class="badge danger">Auto-ban</span>` :
      row.decision === "manual-review" ? `<span class="badge neutral">Manual review</span>` :
                                         `<span class="badge neutral">None</span>`;
    tr.innerHTML = `
      <td>${formatTimestamp(row.timestamp)}</td>
      <td>${row.author}</td>
      <td><a href="/articles/${document.body.dataset.articleId}/comments/${row.id}/">${row.text}</a></td>
      <td>${row.toxicity.toFixed(3)}</td>
      <td>${decisionCell}</td>`;
    tbody.appendChild(tr);
  });
}

function updateThresholdLabels() {
  document.getElementById("autoThresholdValue").textContent =
    parseFloat(document.getElementById("autoThreshold").value).toFixed(2);
  document.getElementById("manualThresholdValue").textContent =
    parseFloat(document.getElementById("manualThreshold").value).toFixed(2);
}

function updateLandingThresholdLabels() {
  const autoVal   = parseFloat(document.getElementById("autoThreshold")?.value   || DEFAULT_AUTO_THRESHOLD);
  const manualVal = parseFloat(document.getElementById("manualThreshold")?.value || DEFAULT_MANUAL_THRESHOLD);

  document.getElementById("autoThresholdValue").textContent   = autoVal.toFixed(2);
  document.getElementById("manualThresholdValue").textContent = manualVal.toFixed(2);

  const label = document.getElementById("thresholdLabel");
  if (label) label.textContent = manualVal.toFixed(2);

  updateZoneBar(manualVal, autoVal);
}

function updateZoneBar(reviewThr, banThr) {
  const lo = Math.min(reviewThr, banThr);
  const hi = Math.max(reviewThr, banThr);

  const safe   = document.getElementById("tzSafe");
  const review = document.getElementById("tzReview");
  const ban    = document.getElementById("tzBan");
  const rl     = document.getElementById("tzReviewLabel");
  const bl     = document.getElementById("tzBanLabel");

  if (!safe || !review || !ban) return;

  safe.style.width   = `${lo * 100}%`;
  review.style.width = `${(hi - lo) * 100}%`;
  ban.style.width    = `${(1 - hi) * 100}%`;
  if (rl) rl.textContent = lo.toFixed(2);
  if (bl) bl.textContent = hi.toFixed(2);
}

async function initCommentPage() {
  const articleId = document.body.dataset.articleId;
  const commentId = document.body.dataset.commentId;
  try {
    const res = await fetch(`/api/articles/${articleId}/comments/${commentId}`);
    if (!res.ok) throw new Error(await readErrorMessage(res));
    const payload = await res.json();

    document.getElementById("commentArticleLink").textContent = payload.article.title;
    document.getElementById("commentArticleLink").href = `/articles/${payload.article.id}/`;
    document.getElementById("commentText").textContent      = payload.comment.text;
    document.getElementById("commentAuthor").textContent    = payload.comment.author;
    document.getElementById("commentTimestamp").textContent = formatTimestamp(payload.comment.timestamp);

    const { decision, toxicity, top_features: features = [] } = payload.comment;
    const { auto_threshold, manual_threshold, model_name } = payload.article;

    document.getElementById("triggeredRule").textContent =
      decision === "auto-ban"      ? `toxicity ≥ ${auto_threshold}` :
      decision === "manual-review" ? `toxicity ≥ ${manual_threshold}` : "Below thresholds";
    document.getElementById("commentConfidence").textContent   = toxicity.toFixed(3);
    document.getElementById("autoThresholdInfo").textContent   = auto_threshold.toFixed(2);
    document.getElementById("manualThresholdInfo").textContent = manual_threshold.toFixed(2);
    document.getElementById("modelName").textContent           = model_name;
    document.getElementById("decisionMode").textContent        =
      decision === "auto-ban" ? "Auto-ban" : decision === "manual-review" ? "Manual review" : "No action";

    const featureList = document.getElementById("featureList");
    featureList.innerHTML = "";
    if (!features.length) {
      featureList.innerHTML = "<li>No explanation available yet.</li>";
      return;
    }
    features.forEach(f => {
      const li = document.createElement("li");
      li.textContent = `${f.feature}: value=${f.value.toFixed(2)}, shap=${f.shap >= 0 ? "+" : ""}${f.shap.toFixed(3)}`;
      featureList.appendChild(li);
    });
  } catch (err) {
    document.getElementById("commentDetail").textContent = `Failed to load comment: ${err.message}`;
  }
}

function formatTimestamp(ts) {
  if (!ts) return "";
  return ts.replace("T", " ").replace("Z", "");
}

async function readErrorMessage(res) {
  try {
    const data = await res.json();
    return data.error || res.statusText;
  } catch { return res.statusText; }
}

function updatePagination(queue, limit) {
  const total = queue.total || 0;
  if (total === 0) {
    queue.pageInfo.textContent = "No items";
    queue.prevButton.disabled = true;
    queue.nextButton.disabled = true;
    return;
  }
  const totalPages = Math.max(1, Math.ceil(total / limit));
  queue.pageInfo.textContent = `Page ${Math.min(queue.page + 1, totalPages)} of ${totalPages}`;
  queue.prevButton.disabled  = queue.page <= 0;
  queue.nextButton.disabled  = (queue.page + 1) * limit >= total;
}
