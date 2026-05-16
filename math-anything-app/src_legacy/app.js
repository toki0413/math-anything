let _apiBase = null;
let _apiInitPromise = null;

async function getApiBase() {
  if (_apiBase) return _apiBase;
  if (!_apiInitPromise) {
    _apiInitPromise = (async () => {
      if (typeof window.__TAURI__ !== "undefined") {
        try {
          const port = await window.__TAURI__.invoke("get_backend_port");
          _apiBase = `http://localhost:${port}`;
          return _apiBase;
        } catch (_) {}
      }
      _apiBase = "http://localhost:8000";
      return _apiBase;
    })();
  }
  return _apiInitPromise;
}

async function api(path, opts = {}) {
  const base = await getApiBase();
  const res = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function $(id) { return document.getElementById(id); }

function showResult(id, html) {
  const el = $(id);
  el.style.display = "block";
  el.innerHTML = html;
}

function badge(text, cls) {
  return `<span class="badge ${cls}">${text}</span>`;
}

function jsonBlock(obj) {
  return `<pre>${syntaxHighlight(JSON.stringify(obj, null, 2))}</pre>`;
}

function syntaxHighlight(json) {
  return json.replace(/("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, (match) => {
    let cls = "color:#d4a054";
    if (/^"/.test(match)) {
      cls = /:$/.test(match) ? "color:#8890a8" : "color:#6ee7b7";
    } else if (/true|false/.test(match)) {
      cls = "color:#38bdf8";
    } else if (/null/.test(match)) {
      cls = "color:#4a5270";
    }
    return `<span style="${cls}">${match}</span>`;
  });
}

async function checkServer() {
  const el = $("serverStatus");
  try {
    await api("/health");
    el.innerHTML = '<span class="status-dot ok"></span> 已连接';
  } catch {
    el.innerHTML = '<span class="status-dot err"></span> 未启动';
  }
}

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((i) => i.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    item.classList.add("active");
    $("page-" + item.dataset.page).classList.add("active");
  });
});

async function doExtract() {
  const engine = $("extractEngine").value;
  const file = $("extractFile").files[0];
  showResult("extractResult", '<h3>提取中</h3><div class="loading"></div>');

  try {
    let data;
    if (file) {
      const fd = new FormData();
      fd.append("file", file);
      const base = await getApiBase();
      const res = await fetch(`${base}/extract/${engine}/file`, { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      data = await res.json();
    } else {
      const params = JSON.parse($("extractParams").value || "{}");
      data = await api(`/extract/${engine}`, {
        method: "POST",
        body: JSON.stringify({ params }),
      });
    }

    const statusBadge = data.success ? badge("SUCCESS", "ok") : badge("FAILED", "err");
    showResult("extractResult", `
      <h3>提取结果 ${statusBadge}</h3>
      <p style="color:var(--text2);margin-bottom:14px;font-family:var(--font-mono);font-size:12px">engine: ${data.engine}</p>
      ${jsonBlock(data.schema)}
      ${data.errors?.length ? `<h3 style="margin-top:20px">错误</h3>${jsonBlock(data.errors)}` : ""}
      ${data.warnings?.length ? `<h3 style="margin-top:20px">警告</h3>${jsonBlock(data.warnings)}` : ""}
    `);
  } catch (e) {
    showResult("extractResult", `<h3>错误</h3><pre style="color:var(--error)">${e.message}</pre>`);
  }
}

async function doGeometry() {
  const engine = $("geoEngine").value;
  const params = JSON.parse($("geoParams").value || "{}");
  let lattice = null;
  try { lattice = JSON.parse($("geoLattice").value || "null"); } catch {}
  const spaceGroup = $("geoSpaceGroup").value || null;

  showResult("geoResult", '<h3>提取中</h3><div class="loading"></div>');

  try {
    const data = await api(`/geometry/${engine}`, {
      method: "POST",
      body: JSON.stringify({ params, lattice_vectors: lattice, space_group: spaceGroup }),
    });

    const curvBadge = data.curvature?.type === "flat"
      ? badge("FLAT", "ok")
      : data.curvature?.type === "mixed"
        ? badge("MIXED", "warn")
        : badge(data.curvature?.type?.toUpperCase() || "?", "info");

    showResult("geoResult", `
      <h3>几何结构</h3>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value">${data.manifold?.topology || "?"}</div>
          <div class="stat-label">流形拓扑</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${data.manifold?.dimension || "?"}</div>
          <div class="stat-label">维度</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${data.curvature?.type || "?"}</div>
          <div class="stat-label">曲率 ${curvBadge}</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${data.symmetries?.length || 0}</div>
          <div class="stat-label">对称群</div>
        </div>
      </div>
      <h3>完整结构</h3>
      ${jsonBlock(data)}
    `);
  } catch (e) {
    showResult("geoResult", `<h3>错误</h3><pre style="color:var(--error)">${e.message}</pre>`);
  }
}

async function doVerify() {
  const statement = $("verifyStatement").value;
  const assumptions = $("verifyAssumptions").value.split("\n").filter((l) => l.trim());
  const goals = $("verifyGoals").value.split("\n").filter((l) => l.trim());
  const proof = $("verifyProof").value;
  const withGeo = $("verifyWithGeo").checked;
  const engine = $("verifyEngine").value;

  showResult("verifyResult", '<h3>验证中</h3><div class="loading"></div>');

  try {
    const data = await api("/verify", {
      method: "POST",
      body: JSON.stringify({
        statement, assumptions, goals, proof_text: proof,
        with_geometry: withGeo, engine,
      }),
    });

    const statusMap = {
      verified: ["VERIFIED", "ok"],
      unverified: ["UNVERIFIED", "warn"],
      contradicted: ["CONTRADICTED", "err"],
      inconclusive: ["INCONCLUSIVE", "info"],
    };
    const [statusText, statusCls] = statusMap[data.formal_status] || ["?", "info"];

    let layersHtml = "";
    if (data.layer_results?.length) {
      layersHtml = `<div class="layer-grid">${data.layer_results.map((lr) => {
        const [lt, lc] = statusMap[lr.status] || ["?", "info"];
        return `<div class="layer-card">
          <div class="layer-name">${lr.layer}</div>
          <div class="layer-status">${badge(lt, lc)}</div>
          <div class="layer-conf">${(lr.confidence * 100).toFixed(0)}% · ${lr.time_ms.toFixed(0)}ms</div>
        </div>`;
      }).join("")}</div>`;
    }

    showResult("verifyResult", `
      <h3>验证结果 ${badge(statusText, statusCls)}</h3>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value">${(data.overall_confidence * 100).toFixed(0)}%</div>
          <div class="stat-label">总体置信度</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${data.layer_results?.length || 0}</div>
          <div class="stat-label">验证层</div>
        </div>
      </div>
      ${layersHtml}
      ${data.issues?.length ? `<h3 style="margin-top:20px">问题</h3>${jsonBlock(data.issues)}` : ""}
      <h3 style="margin-top:20px">完整结果</h3>
      ${jsonBlock(data)}
    `);
  } catch (e) {
    showResult("verifyResult", `<h3>错误</h3><pre style="color:var(--error)">${e.message}</pre>`);
  }
}

async function doProposition() {
  const engine = $("propEngine").value;
  const params = JSON.parse($("propParams").value || "{}");

  showResult("propResult", '<h3>生成中</h3><div class="loading"></div>');

  try {
    const data = await api("/proposition", {
      method: "POST",
      body: JSON.stringify({ engine, params }),
    });

    let tasksHtml = "";
    if (data.proof_tasks?.length) {
      tasksHtml = data.proof_tasks.map((t, i) => `
        <div class="layer-card" style="margin-bottom:10px">
          <div class="layer-name">任务 ${i + 1}</div>
          <div style="font-weight:600;margin:6px 0;font-family:var(--font-display);font-size:15px">${t.name || t.id}</div>
          <div style="color:var(--text2);font-size:13px;line-height:1.5">${t.statement?.substring(0, 200) || ""}</div>
          <div style="color:var(--text3);font-size:11px;margin-top:8px;font-family:var(--font-mono)">type: ${t.type} · difficulty: ${t.difficulty || "?"}</div>
        </div>
      `).join("");
    }

    showResult("propResult", `
      <h3>核心问题</h3>
      <p style="color:var(--text2);margin-bottom:18px;font-size:14px;line-height:1.6">${data.core_problem || "无"}</p>
      <h3>证明任务 (${data.proof_tasks?.length || 0})</h3>
      ${tasksHtml}
      <h3 style="margin-top:20px">完整数据</h3>
      ${jsonBlock(data)}
    `);
  } catch (e) {
    showResult("propResult", `<h3>错误</h3><pre style="color:var(--error)">${e.message}</pre>`);
  }
}

async function loadFlywheel() {
  showResult("flywheelResult", '<h3>加载中</h3><div class="loading"></div>');

  try {
    const data = await api("/flywheel/stats");
    const s = data.stats;

    let degradedHtml = "";
    const engines = Object.keys(data.degraded_engines || {});
    if (engines.length) {
      degradedHtml = engines.map((e) => {
        const isDeg = data.degraded_engines[e];
        return `<div class="layer-card">
          <div class="layer-name">${e}</div>
          <div class="layer-status">${isDeg ? badge("DEGRADED", "err") : badge("OK", "ok")}</div>
        </div>`;
      }).join("");
    }

    showResult("flywheelResult", `
      <h3>飞轮统计</h3>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value">${s.total_records || 0}</div>
          <div class="stat-label">总记录数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${Object.values(s.success_rate_by_type || {}).map((v) => (v * 100).toFixed(0) + "%").join(" / ") || "N/A"}</div>
          <div class="stat-label">成功率 (按类型)</div>
        </div>
      </div>
      ${degradedHtml ? `<h3 style="margin-top:20px">引擎状态</h3><div class="layer-grid">${degradedHtml}</div>` : ""}
      <h3 style="margin-top:20px">详细数据</h3>
      ${jsonBlock(data)}
    `);
  } catch (e) {
    showResult("flywheelResult", `<h3>错误</h3><pre style="color:var(--error)">${e.message}</pre>`);
  }
}

checkServer();
setInterval(checkServer, 10000);

/* ── Settings Panel ── */
const STORAGE_KEY = "math-anything-api-config";

function loadConfig() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch { return {}; }
}

function saveConfig(cfg) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}

function updateConfigUI(cfg) {
  $("cfgProvider").value = cfg.provider || "openai";
  $("cfgApiKey").value = cfg.api_key || "";
  $("cfgBaseUrl").value = cfg.base_url || "";
  $("cfgModel").value = cfg.model || "";
  $("cfgBaseUrlRow").style.display = cfg.provider === "custom" ? "block" : "none";
  updateConfigStatus(cfg);
}

function updateConfigStatus(cfg) {
  const el = $("cfgStatus");
  if (cfg && cfg.api_key) {
    const model = cfg.model || "default";
    el.innerHTML = `<span class="status-dot ok"></span> LLM 已配置 · ${cfg.provider}/${model}`;
  } else {
    el.innerHTML = '<span class="status-dot"></span> 未配置 LLM · 第4层语义验证不可用';
  }
}

async function refreshConfigStatus() {
  try {
    const data = await api("/config/status");
    updateConfigStatus({ api_key: data.has_key ? "***" : "", provider: data.provider, model: data.model });
  } catch { }
}

$("settingsBtn").addEventListener("click", () => {
  const cfg = loadConfig();
  updateConfigUI(cfg);
  $("settingsOverlay").classList.add("open");
  if (cfg.api_key) {
    refreshConfigStatus();

/* ── Cross-Validation Matrix ── */
function buildCrossVal() {
  const methods = $("cvMethods").value.split(",").map(s => s.trim()).filter(Boolean);
  const conclusions = $("cvConclusions").value.split(",").map(s => s.trim()).filter(Boolean);

  if (!methods.length || !conclusions.length) {
    showResult("crossvalResult", '<h3>错误</h3><pre style="color:var(--error)">请输入至少一个方法和一个结论</pre>');
    return;
  }

  const statusSymbols = { confirmed: "✓", partially_confirmed: "~", unconfirmed: "?", contradicted: "✗", not_tested: "·" };
  const statusClasses = { confirmed: "ok", partially_confirmed: "warn", unconfirmed: "info", contradicted: "err", not_tested: "" };

  let header = "<th></th>";
  conclusions.forEach(c => { header += `<th class="cv-th">${c}</th>`; });

  let rows = "";
  methods.forEach(m => {
    let row = `<td class="cv-method">${m}</td>`;
    conclusions.forEach(c => {
      row += `<td class="cv-cell ${statusClasses.not_tested}" data-method="${m}" data-conclusion="${c}" onclick="toggleCvCell(this)">
        ${statusSymbols.not_tested}
      </td>`;
    });
    rows += `<tr>${row}</tr>`;
  });

  showResult("crossvalResult", `
    <h3>交叉验证矩阵</h3>
    <p style="color:var(--text2);margin-bottom:16px;font-size:12px">点击单元格切换状态: · → ✓ → ~ → ? → ✗ → ·</p>
    <table class="cv-table">
      <thead><tr>${header}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div id="cvSummary" style="margin-top:20px"></div>
  `);
}

const CV_CYCLE = ["not_tested", "confirmed", "partially_confirmed", "unconfirmed", "contradicted"];
function toggleCvCell(td) {
  const current = td.dataset.status || "not_tested";
  const idx = CV_CYCLE.indexOf(current);
  const next = CV_CYCLE[(idx + 1) % CV_CYCLE.length];
  td.dataset.status = next;

  const statusSymbols = { confirmed: "✓", partially_confirmed: "~", unconfirmed: "?", contradicted: "✗", not_tested: "·" };
  const statusClasses = { confirmed: "ok", partially_confirmed: "warn", unconfirmed: "info", contradicted: "err", not_tested: "" };
  td.textContent = statusSymbols[next];
  td.className = `cv-cell ${statusClasses[next]}`;
  updateCvSummary();
}

function updateCvSummary() {
  const el = $("cvSummary");
  if (!el) return;

  const conclusions = $("cvConclusions").value.split(",").map(s => s.trim()).filter(Boolean);
  let html = '<h3>结论可靠性</h3><div class="stat-grid">';
  conclusions.forEach(c => {
    const cells = document.querySelectorAll(`td[data-conclusion="${c}"]`);
    let confirmed = 0, total = 0;
    cells.forEach(cell => {
      const s = cell.dataset.status;
      if (s !== "not_tested") total++;
      if (s === "confirmed") confirmed += 1;
      if (s === "partially_confirmed") confirmed += 0.5;
    });
    const reliability = total > 0 ? (confirmed / total * 100).toFixed(0) : "—";
    html += `<div class="stat-card"><div class="stat-value">${reliability}%</div><div class="stat-label">${c}</div></div>`;
  });
  html += '</div>';
  el.innerHTML = html;
}

/* ── Falsifiable Prediction Table ── */
function buildPredictions() {
  const lines = $("predList").value.split("\n").filter(l => l.trim());
  const predictions = lines.map(line => {
    const parts = line.split("|").map(s => s.trim());
    return {
      id: parts[0] || "?",
      statement: parts[1] || "",
      condition: parts[2] || "",
      method: parts[3] || "",
      status: "pending"
    };
  });

  let rows = predictions.map(p => `
    <div class="pred-card" data-pid="${p.id}">
      <div class="pred-header">
        <span class="pred-id">${p.id}</span>
        <span class="badge info pred-status" data-pid="${p.id}" onclick="cyclePredStatus(this)">PENDING</span>
      </div>
      <div class="pred-statement">${p.statement}</div>
      <div class="pred-detail"><span class="pred-label">条件</span> ${p.condition}</div>
      <div class="pred-detail"><span class="pred-label">检验</span> ${p.method}</div>
    </div>
  `).join("");

  showResult("predictionsResult", `
    <h3>可证伪预测表</h3>
    <p style="color:var(--text2);margin-bottom:16px;font-size:12px">点击状态标签切换: PENDING → VERIFIED → FALSIFIED → INCONCLUSIVE → PENDING</p>
    <div class="pred-grid">${rows}</div>
    <div id="predSummary" style="margin-top:20px"></div>
  `);
}

const PRED_CYCLE = ["pending", "verified", "falsified", "inconclusive"];
const PRED_LABELS = { pending: "PENDING", verified: "VERIFIED", falsified: "FALSIFIED", inconclusive: "INCONCLUSIVE" };
const PRED_CLASSES = { pending: "info", verified: "ok", falsified: "err", inconclusive: "warn" };

function cyclePredStatus(badge) {
  const current = badge.dataset.status || "pending";
  const idx = PRED_CYCLE.indexOf(current);
  const next = PRED_CYCLE[(idx + 1) % PRED_CYCLE.length];
  badge.dataset.status = next;
  badge.textContent = PRED_LABELS[next];
  badge.className = `badge ${PRED_CLASSES[next]} pred-status`;
  updatePredSummary();
}

function updatePredSummary() {
  const el = $("predSummary");
  if (!el) return;
  const badges = document.querySelectorAll(".pred-status");
  let verified = 0, falsified = 0, total = badges.length;
  badges.forEach(b => {
    if (b.dataset.status === "verified") verified++;
    if (b.dataset.status === "falsified") falsified++;
  });
  let verdict = "UNVERIFIED";
  let vClass = "info";
  if (falsified > 0) { verdict = "PARTIALLY_FALSIFIED"; vClass = "err"; }
  else if (verified === total && total > 0) { verdict = "ALL_VERIFIED"; vClass = "ok"; }
  else if (verified > total / 2) { verdict = "MOSTLY_VERIFIED"; vClass = "ok"; }
  else if (verified > 0) { verdict = "PARTIALLY_VERIFIED"; vClass = "warn"; }

  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-value">${verified}/${total}</div><div class="stat-label">已验证</div></div>
      <div class="stat-card"><div class="stat-value">${falsified}/${total}</div><div class="stat-label">已证伪</div></div>
      <div class="stat-card"><div class="stat-value">${badge(verdict, vClass)}</div><div class="stat-label">总体判定</div></div>
    </div>
  `;
}

/* ── Dual Perspective Analysis ── */
function buildDualPerspective() {
  const conclusion = $("dualConclusion").value;
  const geoChecks = $("dualGeoChecks").value.split("\n").filter(l => l.trim());
  const anaChecks = $("dualAnaChecks").value.split("\n").filter(l => l.trim());

  let geoHtml = geoChecks.map((c, i) => `
    <div class="dual-check" data-perspective="geo" data-idx="${i}">
      <span class="dual-check-mark" onclick="toggleDualCheck(this)">·</span>
      <span class="dual-check-text">${c}</span>
      <input type="text" class="dual-evidence" placeholder="证据..." onchange="updateDualVerdict()" />
    </div>
  `).join("");

  let anaHtml = anaChecks.map((c, i) => `
    <div class="dual-check" data-perspective="ana" data-idx="${i}">
      <span class="dual-check-mark" onclick="toggleDualCheck(this)">·</span>
      <span class="dual-check-text">${c}</span>
      <input type="text" class="dual-evidence" placeholder="证据..." onchange="updateDualVerdict()" />
    </div>
  `).join("");

  showResult("dualResult", `
    <h3>双视角分析: ${conclusion}</h3>
    <div class="dual-grid">
      <div class="dual-panel">
        <div class="dual-panel-header geo">
          <span class="dual-panel-icon">◈</span>
          Yau 视角 (微分几何)
        </div>
        <div class="dual-panel-subtitle">有什么几何结构？</div>
        ${geoHtml}
      </div>
      <div class="dual-panel">
        <div class="dual-panel-header ana">
          <span class="dual-panel-icon">◇</span>
          Tao 视角 (概率 + 调和分析)
        </div>
        <div class="dual-panel-subtitle">统计信号是真实的吗？</div>
        ${anaHtml}
      </div>
    </div>
    <div id="dualVerdict" style="margin-top:20px"></div>
  `);
}

function toggleDualCheck(mark) {
  const current = mark.textContent;
  const cycle = { "·": "✓", "✓": "✗", "✗": "?" , "?": "·" };
  mark.textContent = cycle[current] || "·";
  const parent = mark.closest(".dual-check");
  if (mark.textContent === "✓") parent.classList.add("done");
  else parent.classList.remove("done");
  updateDualVerdict();
}

function updateDualVerdict() {
  const el = $("dualVerdict");
  if (!el) return;

  const geoMarks = document.querySelectorAll('.dual-check[data-perspective="geo"] .dual-check-mark');
  const anaMarks = document.querySelectorAll('.dual-check[data-perspective="ana"] .dual-check-mark');

  const geoConfirmed = [...geoMarks].filter(m => m.textContent === "✓").length;
  const geoDenied = [...geoMarks].filter(m => m.textContent === "✗").length;
  const anaConfirmed = [...anaMarks].filter(m => m.textContent === "✓").length;
  const anaDenied = [...anaMarks].filter(m => m.textContent === "✗").length;

  let geoVerdict = "INCONCLUSIVE";
  if (geoConfirmed > 0 && geoDenied === 0) geoVerdict = "SUPPORTS";
  else if (geoDenied > 0 && geoConfirmed === 0) geoVerdict = "CONTRADICTS";
  else if (geoConfirmed > 0 && geoDenied > 0) geoVerdict = "MIXED";

  let anaVerdict = "INCONCLUSIVE";
  if (anaConfirmed > 0 && anaDenied === 0) anaVerdict = "SUPPORTS";
  else if (anaDenied > 0 && anaConfirmed === 0) anaVerdict = "CONTRADICTS";
  else if (anaConfirmed > 0 && anaDenied > 0) anaVerdict = "MIXED";

  let agreement = null;
  if (geoVerdict === anaVerdict && geoVerdict !== "INCONCLUSIVE") agreement = true;
  else if (geoVerdict !== anaVerdict && !geoVerdict.includes("INCONCLUSIVE") && !anaVerdict.includes("INCONCLUSIVE")) agreement = false;

  const verdictMap = { SUPPORTS: ["ok", "支持"], CONTRADICTS: ["err", "反驳"], MIXED: ["warn", "混合"], INCONCLUSIVE: ["info", "不确定"] };
  const [gc, gt] = verdictMap[geoVerdict];
  const [ac, at] = verdictMap[anaVerdict];

  let agreeText = "至少一个视角缺乏充分证据 ?";
  let agreeCls = "info";
  if (agreement === true) { agreeText = "两个视角收敛于同一结论 ✓"; agreeCls = "ok"; }
  else if (agreement === false) { agreeText = "视角分歧 — 需要进一步调查 ✗"; agreeCls = "err"; }

  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-value">${badge(gt, gc)}</div><div class="stat-label">几何视角判定</div></div>
      <div class="stat-card"><div class="stat-value">${badge(at, ac)}</div><div class="stat-label">分析视角判定</div></div>
      <div class="stat-card"><div class="stat-value">${badge(agreeText, agreeCls)}</div><div class="stat-label">一致性</div></div>
    </div>
  `;
}
  }
});

$("settingsClose").addEventListener("click", () => {
  $("settingsOverlay").classList.remove("open");
});

$("settingsOverlay").addEventListener("click", (e) => {
  if (e.target === $("settingsOverlay")) {
    $("settingsOverlay").classList.remove("open");
  }
});

$("cfgProvider").addEventListener("change", () => {
  const v = $("cfgProvider").value;
  $("cfgBaseUrlRow").style.display = v === "custom" ? "block" : "none";
  if (v === "anthropic") {
    $("cfgBaseUrl").value = "";
    $("cfgBaseUrlRow").style.display = "none";
  }
});

$("keyToggle").addEventListener("click", () => {
  const inp = $("cfgApiKey");
  const isPass = inp.type === "password";
  inp.type = isPass ? "text" : "password";
  $("keyToggle").innerHTML = isPass
    ? '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M1 1l14 14M8 4C5.3 4 3 5.5 1.5 8c1 1.7 3 3 6.5 3 .8 0 1.6-.1 2.3-.3"/></svg>'
    : '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M8 3C5 3 2 8 2 8s3 5 6 5 6-5 6-5-3-5-6-5z"/><circle cx="8" cy="8" r="2"/></svg>';
});

$("cfgSave").addEventListener("click", async () => {
  const cfg = {
    provider: $("cfgProvider").value,
    api_key: $("cfgApiKey").value.trim(),
    base_url: $("cfgBaseUrl").value.trim(),
    model: $("cfgModel").value.trim(),
  };

  if (!cfg.api_key) {
    $("cfgStatus").innerHTML = '<span class="status-dot" style="background:var(--error)"></span> 请输入 API Key';
    return;
  }

  $("cfgSave").textContent = "保存中...";
  $("cfgSave").disabled = true;

  try {
    await api("/config", {
      method: "POST",
      body: JSON.stringify(cfg),
    });
    saveConfig(cfg);
    updateConfigUI(cfg);
    $("settingsOverlay").classList.remove("open");
  } catch (e) {
    $("cfgStatus").innerHTML = `<span class="status-dot" style="background:var(--error)"></span> 保存失败: ${e.message}`;
  } finally {
    $("cfgSave").textContent = "保存配置";
    $("cfgSave").disabled = false;
  }
});

$("cfgClear").addEventListener("click", async () => {
  localStorage.removeItem(STORAGE_KEY);
  $("cfgProvider").value = "openai";
  $("cfgApiKey").value = "";
  $("cfgBaseUrl").value = "";
  $("cfgModel").value = "";
  $("cfgBaseUrlRow").style.display = "none";
  updateConfigStatus({});

  try {
    await api("/config", {
      method: "POST",
      body: JSON.stringify({ provider: "", api_key: "", base_url: "", model: "" }),
    });
  } catch { }
});

refreshConfigStatus();
