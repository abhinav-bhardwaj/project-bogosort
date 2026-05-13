const FEATURE_LABELS = {
  // Sentiment
  vader_compound:      { label: "Overall sentiment",          fmt: v => v.toFixed(2) + " (−1 = very negative, +1 = positive)" },
  vader_neg:           { label: "Negative tone",              fmt: v => (v * 100).toFixed(0) + "% of text" },
  vader_pos:           { label: "Positive tone",              fmt: v => (v * 100).toFixed(0) + "% of text" },
  vader_is_negative:   { label: "Negative tone detected",     fmt: v => v === 1 ? "Yes" : "No" },
  vader_intensity:     { label: "Emotional intensity",        fmt: v => v.toFixed(2) + " (0 = neutral, 1 = peak)" },
  vader_pos_minus_neg: { label: "Net sentiment",              fmt: v => (v >= 0 ? "+" : "") + v.toFixed(2) },
  // Second person
  has_second_person:     { label: "Directly addresses someone",  fmt: v => v === 1 ? "Yes" : "No" },
  second_person_count:   { label: 'Words like "you" / "your"',   fmt: v => v + " word" + (v !== 1 ? "s" : "") },
  second_person_density: { label: "Direct-address density",      fmt: v => (v * 100).toFixed(1) + "% of words" },
  // Profanity
  profanity_count:            { label: "Profanity words",      fmt: v => v + " word" + (v !== 1 ? "s" : "") },
  obfuscated_profanity_count: { label: "Disguised profanity",  fmt: v => v + " word" + (v !== 1 ? "s" : "") },
  // Slang
  slang_count: { label: "Toxic slang terms", fmt: v => v + " term" + (v !== 1 ? "s" : "") },
  // Text shape
  char_count:        { label: "Comment length",       fmt: v => v + " characters" },
  word_count:        { label: "Word count",            fmt: v => v + " word" + (v !== 1 ? "s" : "") },
  exclamation_count: { label: "Exclamation marks",    fmt: v => v },
  uppercase_ratio:   { label: "ALL CAPS proportion",  fmt: v => (v * 100).toFixed(0) + "% of words" },
  unique_word_ratio: { label: "Vocabulary variety",   fmt: v => (v * 100).toFixed(0) + "% unique words" },
  // Elongation & punctuation
  elongated_token_count:   { label: "Exaggerated words",      fmt: v => v + " word" + (v !== 1 ? "s" : "") + ' (e.g. "soooo")' },
  consecutive_punct_count: { label: "Repeated punctuation",   fmt: v => v + ' run' + (v !== 1 ? "s" : "") + ' (e.g. "!!!")' },
  // URLs & IPs
  url_count:     { label: "Links included",          fmt: v => v },
  ip_count:      { label: "IP addresses mentioned",  fmt: v => v },
  has_url_or_ip: { label: "Contains link or IP",     fmt: v => v === 1 ? "Yes" : "No" },
  // Syntactic
  negation_count:      { label: 'Negation words (e.g. "not", "never")', fmt: v => v + " word" + (v !== 1 ? "s" : "") },
  sentence_count:      { label: "Number of sentences",                   fmt: v => v },
  avg_sentence_length: { label: "Average sentence length",               fmt: v => v.toFixed(1) + " words" },
  // Identity mentions
  identity_mention_count: { label: "Identity group mentions",      fmt: v => v + " mention" + (v !== 1 ? "s" : "") },
  identity_race:          { label: "Mentions race or ethnicity",   fmt: v => v === 1 ? "Yes" : "No" },
  identity_gender:        { label: "Mentions gender",              fmt: v => v === 1 ? "Yes" : "No" },
  identity_sexuality:     { label: "Mentions sexual orientation",  fmt: v => v === 1 ? "Yes" : "No" },
  identity_religion:      { label: "Mentions religion",            fmt: v => v === 1 ? "Yes" : "No" },
  identity_disability:    { label: "Mentions disability",          fmt: v => v === 1 ? "Yes" : "No" },
  identity_nationality:   { label: "Mentions nationality / origin", fmt: v => v === 1 ? "Yes" : "No" },
};

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

    const summaryEl = document.getElementById("articleSummary");
    const summaryToggle = document.getElementById("summaryToggle");
    const fullSummary = article.summary || "";
    const SUMMARY_MAX = 200;
    if (fullSummary.length > SUMMARY_MAX) {
      const shortSummary = fullSummary.slice(0, SUMMARY_MAX).trimEnd() + "…";
      summaryEl.textContent = shortSummary;
      summaryToggle.hidden = false;
      let expanded = false;
      summaryToggle.addEventListener("click", () => {
        expanded = !expanded;
        summaryEl.textContent = expanded ? fullSummary : shortSummary;
        summaryToggle.textContent = expanded ? "Show less" : "Show more";
      });
    } else {
      summaryEl.textContent = fullSummary;
    }
    document.getElementById("articleError").textContent = "";

    renderInferenceStats(article.inference_stats || {});

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
  const articleId = document.body.dataset.articleId;

  if (errorMessage) {
    tbody.innerHTML = `<tr><td colspan="5">Failed to load: ${errorMessage}</td></tr>`;
    return;
  }
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="5">No comments to display.</td></tr>`;
    return;
  }

  rows.forEach(row => {
    const tr = document.createElement("tr");

    const tsCell = document.createElement("td");
    tsCell.textContent = formatTimestamp(row.timestamp);
    tr.appendChild(tsCell);

    const authorCell = document.createElement("td");
    authorCell.textContent = row.author;
    tr.appendChild(authorCell);

    tr.appendChild(buildCommentCell(row, articleId));

    const toxCell = document.createElement("td");
    toxCell.textContent = row.toxicity.toFixed(3);
    tr.appendChild(toxCell);

    tr.appendChild(buildActionCell(row, mode, articleId));

    tbody.appendChild(tr);
  });
}

function buildCommentCell(row, articleId) {
  const td = document.createElement("td");
  const MAX = 120;
  const text = row.text || "";

  const a = document.createElement("a");
  a.href = `/articles/${articleId}/comments/${row.id}/`;

  if (text.length <= MAX) {
    a.textContent = text;
    td.appendChild(a);
    return td;
  }

  const shortSpan = document.createElement("span");
  shortSpan.textContent = text.slice(0, MAX).trimEnd() + "…";

  const fullSpan = document.createElement("span");
  fullSpan.textContent = text;
  fullSpan.hidden = true;

  a.appendChild(shortSpan);
  a.appendChild(fullSpan);
  td.appendChild(a);

  const toggle = document.createElement("button");
  toggle.className = "comment-toggle";
  toggle.textContent = "more";
  toggle.addEventListener("click", e => {
    e.preventDefault();
    const nowExpanded = !fullSpan.hidden;
    fullSpan.hidden = nowExpanded;
    shortSpan.hidden = !nowExpanded;
    toggle.textContent = nowExpanded ? "more" : "less";
  });
  td.appendChild(document.createTextNode(" "));
  td.appendChild(toggle);
  return td;
}

function buildActionCell(row, mode, articleId) {
  const td = document.createElement("td");

  if (mode === "auto") {
    td.innerHTML = `<span class="badge danger">Auto-ban</span>`;
    return td;
  }

  if (mode === "manual") {
    const d = row.decision;
    if (d === "auto-ban") {
      td.innerHTML = `<span class="badge danger">Banned</span>`;
    } else if (d === "none") {
      td.innerHTML = `<span class="badge success">Not banned</span>`;
    } else {
      const banBtn = document.createElement("button");
      banBtn.className = "btn small danger";
      banBtn.textContent = "Ban";

      const notBanBtn = document.createElement("button");
      notBanBtn.className = "btn small";
      notBanBtn.textContent = "Not ban";

      const badge = document.createElement("span");
      badge.className = "badge neutral";
      badge.textContent = "Pending";

      const act = decision => async () => {
        banBtn.disabled = notBanBtn.disabled = true;
        badge.textContent = "Saving…";
        try {
          const res = await fetch(`/api/articles/${articleId}/comments/${row.id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision })
          });
          if (!res.ok) throw new Error(await readErrorMessage(res));
          td.innerHTML = decision === "manual-ban"
            ? `<span class="badge manual-ban">Manually banned</span>`
            : `<span class="badge success">Not banned</span>`;
        } catch (err) {
          banBtn.disabled = notBanBtn.disabled = false;
          badge.textContent = "Error";
          badge.className = "badge danger";
        }
      };

      banBtn.addEventListener("click", act("manual-ban"));
      notBanBtn.addEventListener("click", act("none"));

      td.appendChild(banBtn);
      td.appendChild(document.createTextNode(" "));
      td.appendChild(notBanBtn);
      td.appendChild(document.createTextNode(" "));
      td.appendChild(badge);
    }
    return td;
  }

  // "all" mode
  const label =
    row.decision === "auto-ban"      ? `<span class="badge danger">Auto-ban</span>` :
    row.decision === "manual-ban"    ? `<span class="badge manual-ban">Manually banned</span>` :
    row.decision === "manual-review" ? `<span class="badge warning">Pending review</span>` :
    row.decision === "none"          ? `<span class="badge success">Not banned</span>` :
                                       `<span class="badge neutral">None</span>`;
  td.innerHTML = label;
  return td;
}

function renderInferenceStats(stats) {
  const el = document.getElementById("inferenceStats");
  if (!el) return;
  if (!stats || !stats.count) {
    el.hidden = true;
    return;
  }
  const totalSec = (stats.total_ms / 1000).toFixed(2);
  el.hidden = false;
  el.innerHTML = `
    <span class="inf-stat" title="Comments scored">⚙ ${stats.count} scored</span>
    <span class="inf-sep">·</span>
    <span class="inf-stat" title="Average inference time per comment">avg ${stats.avg_ms} ms/comment</span>
    <span class="inf-sep">·</span>
    <span class="inf-stat" title="Fastest inference">min ${stats.min_ms} ms</span>
    <span class="inf-sep">·</span>
    <span class="inf-stat" title="Slowest inference">max ${stats.max_ms} ms</span>
    <span class="inf-sep">·</span>
    <span class="inf-stat" title="Total inference wall time">total ${totalSec} s</span>
  `;
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
    const inferenceEl = document.getElementById("inferenceTime");
    if (inferenceEl) {
      const ms = payload.comment.inference_ms;
      inferenceEl.textContent = ms > 0 ? `${ms.toFixed(2)} ms` : "—";
    }
    document.getElementById("autoThresholdInfo").textContent   = auto_threshold.toFixed(2);
    document.getElementById("manualThresholdInfo").textContent = manual_threshold.toFixed(2);
    document.getElementById("modelName").textContent           = model_name;
    document.getElementById("decisionMode").textContent        =
      decision === "auto-ban" ? "Auto-ban" : decision === "manual-review" ? "Manual review" : "No action";

    const featureList = document.getElementById("featureList");
    featureList.innerHTML = "";
    if (!features.length) {
      featureList.innerHTML = "<p class='shap-empty'>No explanation available yet.</p>";
      return;
    }

    const maxShap = Math.max(...features.map(f => Math.abs(f.shap)));

    features.forEach(f => {
      const meta   = FEATURE_LABELS[f.feature] || { label: f.feature, fmt: v => v.toFixed(3) };
      const isPos  = f.shap >= 0;
      const barPct = maxShap > 0 ? (Math.abs(f.shap) / maxShap * 100).toFixed(1) : 0;
      const shapFmt = (isPos ? "+" : "") + f.shap.toFixed(3);

      const row = document.createElement("div");
      row.className = "shap-row";
      row.innerHTML = `
        <div class="shap-label">${meta.label}<span class="shap-feature-name">${f.feature}</span></div>
        <div class="shap-bar-track">
          <div class="shap-bar ${isPos ? "shap-bar-pos" : "shap-bar-neg"}"
               style="width:${barPct}%"></div>
        </div>
        <div class="shap-meta">
          <span class="shap-direction ${isPos ? "shap-dir-pos" : "shap-dir-neg"}">
            ${isPos ? "↑ increases risk" : "↓ reduces risk"}
          </span>
          <span class="shap-value" title="SHAP value: ${shapFmt}">${meta.fmt(f.value)}</span>
        </div>`;
      featureList.appendChild(row);
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
