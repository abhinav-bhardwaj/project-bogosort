let articleCharts = {};
const DEFAULT_AUTO_THRESHOLD = 0.75;
const DEFAULT_MANUAL_THRESHOLD = 0.55;

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
    statusContainer.innerHTML = "";
    if (!inputs.length) {
      statusContainer.textContent = "Please enter at least one Wikipedia URL.";
      return;
    }
    for (const url of inputs) {
      const statusRow = document.createElement("div");
      statusRow.className = "status-item";
      statusRow.textContent = `${url} — ingesting...`;
      statusContainer.appendChild(statusRow);
      try {
        const res = await fetch("/api/articles/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url,
            limit,
            auto_threshold: autoThreshold,
            manual_threshold: manualThreshold,
            model_name: modelName
          })
        });
        if (!res.ok) {
          throw new Error(await readErrorMessage(res));
        }
        statusRow.textContent = `${url} — complete`;
      } catch (err) {
        statusRow.textContent = `${url} — failed: ${err.message}`;
      }
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
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const data = await res.json();
    select.innerHTML = "";
    
    const models = data.models || [];
    if (models.length === 0) {
      throw new Error("No models returned from API");
    }
    
    models.forEach(model => {
      const opt = document.createElement("option");
      opt.value = model.model_id;
      opt.textContent = model.model_name;
      select.appendChild(opt);
    });
  } catch (err) {
    select.innerHTML = `<option value="ensemble">Ensemble</option>`;
    console.error("Failed to load models:", err.message);
  }
}

async function loadArticles() {
  const grid = document.getElementById("articlesGrid");
  grid.innerHTML = "";
  try {
    const res = await fetch("/api/articles");
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
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

  if (articleCharts[article.id]) {
    articleCharts[article.id].destroy();
  }

  const thresholdLine = trend.dates.map(() => trend.threshold);

  articleCharts[article.id] = new Chart(ctx, {
    type: "line",
    data: {
      labels: trend.dates,
      datasets: [
        {
          label: "Additive Toxicity Score",
          data: trend.scores,
          borderColor: "#c43b3b",
          backgroundColor: "rgba(196,59,59,0.1)",
          fill: true,
          tension: 0.3
        },
        {
          label: "Action Threshold",
          data: thresholdLine,
          borderColor: "#3b6fc4",
          borderDash: [5, 5],
          fill: false,
          tension: 0
        }
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
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const article = await res.json();

    document.getElementById("articleTitle").textContent = article.title;
    const articleLink = document.getElementById("articleLink");
    articleLink.href = article.url;
    articleLink.textContent = article.url;
    document.getElementById("articleSummary").textContent = article.summary || "";
    const articleError = document.getElementById("articleError");
    articleError.textContent = "";

    const autoThreshold = document.getElementById("autoThreshold");
    const manualThreshold = document.getElementById("manualThreshold");
    autoThreshold.value = article.auto_threshold;
    manualThreshold.value = article.manual_threshold;

    updateThresholdLabels();

    autoThreshold.addEventListener("input", updateThresholdLabels);
    manualThreshold.addEventListener("input", updateThresholdLabels);

    const queues = {
      all: {
        decision: null,
        tableId: "allTable",
        limitSelect: document.getElementById("allLimit"),
        sortSelect: document.getElementById("allSort"),
        prevButton: document.getElementById("allPrev"),
        nextButton: document.getElementById("allNext"),
        pageInfo: document.getElementById("allPageInfo"),
        page: 0,
        total: 0,
        mode: "all",
      },
      auto: {
        decision: "auto-ban",
        tableId: "autoBanTable",
        limitSelect: document.getElementById("autoLimit"),
        sortSelect: document.getElementById("autoSort"),
        prevButton: document.getElementById("autoPrev"),
        nextButton: document.getElementById("autoNext"),
        pageInfo: document.getElementById("autoPageInfo"),
        page: 0,
        total: 0,
        mode: "auto"
      },
      manual: {
        decision: "manual-review",
        tableId: "manualTable",
        limitSelect: document.getElementById("manualLimit"),
        sortSelect: document.getElementById("manualSort"),
        prevButton: document.getElementById("manualPrev"),
        nextButton: document.getElementById("manualNext"),
        pageInfo: document.getElementById("manualPageInfo"),
        page: 0,
        total: 0,
        mode: "manual"
      }
    };

    Object.values(queues).forEach(queue => {
      queue.limitSelect.addEventListener("change", async () => {
        queue.page = 0;
        await loadQueue(articleId, queue);
      });
      queue.sortSelect.addEventListener("change", async () => {
        queue.page = 0;
        await loadQueue(articleId, queue);
      });
      queue.prevButton.addEventListener("click", async () => {
        if (queue.page > 0) {
          queue.page -= 1;
          await loadQueue(articleId, queue);
        }
      });
      queue.nextButton.addEventListener("click", async () => {
        const limit = parseInt(queue.limitSelect.value, 10);
        if ((queue.page + 1) * limit < queue.total) {
          queue.page += 1;
          await loadQueue(articleId, queue);
        }
      });
    });

    document.getElementById("applyThresholds").addEventListener("click", async () => {
      articleError.textContent = "";
      try {
        const res = await fetch(`/api/articles/${articleId}/thresholds`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            auto_threshold: parseFloat(autoThreshold.value),
            manual_threshold: parseFloat(manualThreshold.value)
          })
        });
        if (!res.ok) {
          throw new Error(await readErrorMessage(res));
        }
        await Promise.all([loadQueue(articleId, queues.all), loadQueue(articleId, queues.auto), loadQueue(articleId, queues.manual)]);
      } catch (err) {
        articleError.textContent = `Failed to update thresholds: ${err.message}`;
      }
    });

    await Promise.all([loadQueue(articleId, queues.all),loadQueue(articleId, queues.auto), loadQueue(articleId, queues.manual)]);
  } catch (err) {
    document.getElementById("articleDetail").textContent = `Failed to load article: ${err.message}`;
  }
}

async function loadQueue(articleId, queue) {
  const limit = parseInt(queue.limitSelect.value, 10);
  const offset = queue.page * limit;
  try {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
      sort: queue.sortSelect.value
    });
    if (queue.decision) {
      params.set("decision", queue.decision);
    }
    const res = await fetch(`/api/articles/${articleId}/comments?${params.toString()}`);
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
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
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">Failed to load comments: ${errorMessage}</td>`;
    tbody.appendChild(tr);
    return;
  }

  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">No comments to display.</td>`;
    tbody.appendChild(tr);
    return;
  }

  rows.forEach(row => {
    const tr = document.createElement("tr");
    let decisionCell = "";
    if (mode === "auto") {
      decisionCell = `<span class="badge danger">Auto-ban</span>`;
    } else if (mode === "manual") {
      decisionCell = `<button class="btn small danger">Ban</button>
      <button class="btn small">Not ban</button> <span class="badge neutral">Pending</span>`;
    } else {
      const label =
        row.decision === "auto-ban"
          ? `<span class="badge danger">Auto-ban</span>`
          : row.decision === "manual-review"
            ? `<span class="badge neutral">Manual review</span>`
            : `<span class="badge neutral">None</span>`;
      decisionCell = label;
    }

    tr.innerHTML = `
      <td>${formatTimestamp(row.timestamp)}</td>
      <td>${row.author}</td>
      <td><a href="/articles/${document.body.dataset.articleId}/comments/${row.id}/">${row.text}</a></td>
      <td>${row.toxicity.toFixed(3)}</td>
      <td>${decisionCell}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateThresholdLabels() {
  document.getElementById("autoThresholdValue").textContent =
    parseFloat(document.getElementById("autoThreshold").value).toFixed(2);
  document.getElementById("manualThresholdValue").textContent =
    parseFloat(document.getElementById("manualThreshold").value).toFixed(2);
}

async function initCommentPage() {
  const articleId = document.body.dataset.articleId;
  const commentId = document.body.dataset.commentId;

  try {
    const res = await fetch(`/api/articles/${articleId}/comments/${commentId}`);
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const payload = await res.json();

    document.getElementById("commentArticleLink").textContent = payload.article.title;
    document.getElementById("commentArticleLink").href = `/articles/${payload.article.id}/`;
    document.getElementById("commentText").textContent = payload.comment.text;
    document.getElementById("commentAuthor").textContent = payload.comment.author;
    document.getElementById("commentTimestamp").textContent = formatTimestamp(payload.comment.timestamp);

    const decision = payload.comment.decision;
    const autoThreshold = payload.article.auto_threshold;
    const manualThreshold = payload.article.manual_threshold;
    const modelName = payload.article.model_name;

    document.getElementById("triggeredRule").textContent =
      decision === "auto-ban"
        ? `toxicity >= ${autoThreshold}`
        : decision === "manual-review"
          ? `toxicity >= ${manualThreshold}`
          : "Below thresholds";
    document.getElementById("commentConfidence").textContent = payload.comment.toxicity.toFixed(3);
    document.getElementById("autoThresholdInfo").textContent = autoThreshold.toFixed(2);
    document.getElementById("manualThresholdInfo").textContent = manualThreshold.toFixed(2);
    document.getElementById("modelName").textContent = modelName;
    document.getElementById("decisionMode").textContent =
      decision === "auto-ban" ? "Auto-ban" : decision === "manual-review" ? "Manual review" : "No action";

    const featureList = document.getElementById("featureList");
    featureList.innerHTML = "";
    const features = payload.comment.top_features || [];
    if (!features.length) {
      const li = document.createElement("li");
      li.textContent = "No explanation available yet.";
      featureList.appendChild(li);
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

function updateLandingThresholdLabels() {
  const autoValue = parseFloat(
    document.getElementById("autoThreshold").value || DEFAULT_AUTO_THRESHOLD
  );
  const manualValue = parseFloat(
    document.getElementById("manualThreshold").value || DEFAULT_MANUAL_THRESHOLD
  );
  document.getElementById("autoThresholdValue").textContent = autoValue.toFixed(2);
  document.getElementById("manualThresholdValue").textContent = manualValue.toFixed(2);
  const label = document.getElementById("thresholdLabel");
  if (label) {
    label.textContent = manualValue.toFixed(2);
  }
}

async function readErrorMessage(res) {
  try {
    const data = await res.json();
    return data.error || res.statusText;
  } catch {
    return res.statusText;
  }
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
  queue.prevButton.disabled = queue.page <= 0;
  queue.nextButton.disabled = (queue.page + 1) * limit >= total;
}