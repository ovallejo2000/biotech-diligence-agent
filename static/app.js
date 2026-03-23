const MODULES = [
  "A. Rapid Screening",
  "B. Scientific & Mechanistic Validation",
  "C. Data & Evidence Quality",
  "D. Development Pathway & Inflection Points",
  "E. Competitive Landscape & Positioning",
  "F. Market & Commercial Reality",
  "G. Team & Execution Risk",
  "H. IP & Regulatory Risk",
  "I. Risk Decomposition",
  "J. Investment Framing",
  "K. Decision Engine"
];

let currentCompany = "";
let currentMemo = "";
let attachedFiles = []; // { file, name }

// ── File attachment handling ──────────────────────────────────────

function handleFiles(fileList) {
  for (const f of fileList) {
    if (!attachedFiles.find(x => x.name === f.name)) {
      attachedFiles.push({ file: f, name: f.name });
    }
  }
  // reset input so same file can be re-added if removed
  document.getElementById("fileInput").value = "";
  renderFileList();
}

function removeFile(name) {
  attachedFiles = attachedFiles.filter(x => x.name !== name);
  renderFileList();
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function renderFileList() {
  const list = document.getElementById("fileList");
  if (!list) return;
  list.innerHTML = attachedFiles.map(function(x) {
    return '<div class="file-chip">' +
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0">' +
      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
      '<span class="fname">' + x.name + '</span>' +
      '<span class="fsize">' + formatBytes(x.file.size) + '</span>' +
      '<span class="fremove" onclick="removeFile(\'' + x.name.replace(/'/g, "\\'") + '\')">&times;</span>' +
      '</div>';
  }).join("");
}

async function extractAttachments() {
  if (!attachedFiles.length) return "";
  const form = new FormData();
  attachedFiles.forEach(x => form.append("files", x.file, x.name));
  try {
    const res = await fetch("/extract", { method: "POST", body: form });
    const data = await res.json();
    return data.text || "";
  } catch(e) {
    return "";
  }
}

function buildModuleList() {
  const el = document.getElementById("module-list");
  el.innerHTML = MODULES.map((m, i) => `
    <div class="module-row" id="mod-${i}">
      <div class="module-dot" id="dot-${i}"></div>
      <span class="module-label" id="lbl-${i}">${m}</span>
    </div>`).join("");
}

function setModuleState(step, status) {
  const i = step - 1;
  const dot = document.getElementById("dot-" + i);
  const lbl = document.getElementById("lbl-" + i);
  if (!dot) return;
  dot.className = "module-dot " + status;
  lbl.className = "module-label " + status;
}

function updateProgress(step, total) {
  document.getElementById("progressBar").style.width = ((step / total) * 100) + "%";
  document.getElementById("progress-label").textContent = "Module " + step + " of " + total;
}

async function checkTokens() {
  try {
    const res = await fetch("/tokens/check");
    return await res.json();
  } catch(e) {
    return { available: true, warn: false };
  }
}

async function runDiligence() {
  const company = document.getElementById("company").value.trim();
  const manualInputs = document.getElementById("inputs").value.trim();
  if (!company) { alert("Please enter a company name."); return; }

  // Pre-flight token check — hard block if not enough for a full run
  const tokens = await checkTokens();
  if (!tokens.available) {
    document.getElementById("rightPanel").innerHTML =
      '<div class="rate-limit-box"><strong>\u23F3 Not enough tokens</strong>' +
      '<p>' + (tokens.message || "Not enough tokens for a full analysis. Please try again later.") + '</p></div>';
    return;
  }

  currentCompany = company;
  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing\u2026";

  buildModuleList();
  document.getElementById("progress-panel").style.display = "flex";
  document.getElementById("progressBar").style.width = "0%";
  document.getElementById("progress-label").textContent = "Starting\u2026";

  document.getElementById("rightPanel").innerHTML =
    '<div class="empty-state">' +
    '<div class="icon">\u23F3</div>' +
    '<h2>Running diligence on ' + company + '\u2026</h2>' +
    '<p>All 11 modules running in sequence. This takes 60\u201390 seconds.</p>' +
    '</div>';

  // Extract text from any attached files
  let fileText = "";
  if (attachedFiles.length) {
    document.getElementById("progress-label").textContent = "Extracting attachments\u2026";
    fileText = await extractAttachments();
  }
  const inputs = [manualInputs, fileText].filter(Boolean).join("\n\n");

  let url = "/diligence/stream?company=" + encodeURIComponent(company);
  if (inputs) url += "&inputs=" + encodeURIComponent(inputs);

  const es = new EventSource(url);

  es.addEventListener("progress", function(e) {
    const d = JSON.parse(e.data);
    setModuleState(d.step, d.status);
    updateProgress(d.step, d.total);
  });

  es.addEventListener("memo", function(e) {
    const payload = JSON.parse(e.data);
    currentMemo = payload.memo;
    es.close();
    renderMemo(company, payload.memo);
    loadHistory();
    btn.disabled = false;
    btn.textContent = "Run Full Diligence";
    document.getElementById("progress-label").textContent = "Complete \u2713";
  });

  es.addEventListener("error", function(e) {
    // Ignore native connection-close events (no data) — only handle server-sent errors
    if (!e.data) return;
    es.close();
    let msg = "An error occurred.";
    try { msg = JSON.parse(e.data).message; } catch(ex) {}
    const isRateLimit = msg.indexOf("token limit") !== -1 || msg.indexOf("Rate limit") !== -1 || msg.indexOf("rolling window") !== -1;
    const boxClass = isRateLimit ? "rate-limit-box" : "error-box";
    const label = isRateLimit ? "\u23F3 Token limit reached" : "Error";
    document.getElementById("rightPanel").innerHTML =
      '<div class="' + boxClass + '"><strong>' + label + '</strong><p>' + msg + '</p></div>';
    btn.disabled = false;
    btn.textContent = "Run Full Diligence";
  });

  es.addEventListener("done", function() {
    es.close();
    btn.disabled = false;
    btn.textContent = "Run Full Diligence";
    document.getElementById("progress-label").textContent = "Complete \u2713";
  });
}

function extractVerdict(md) {
  const EMOJI = { INVEST: "\u2705", WATCH: "\u26A0\uFE0F", PASS: "\u274C" };
  const verdictSection = md.indexOf("K. FINAL VERDICT");
  if (verdictSection !== -1) {
    const chunk = md.slice(verdictSection, verdictSection + 300);
    for (const label of ["INVEST", "WATCH", "PASS"]) {
      if (chunk.indexOf("# " + label) !== -1 || chunk.indexOf("**" + label + "**") !== -1) {
        return { emoji: EMOJI[label], label: label };
      }
    }
  }
  for (const label of ["INVEST", "WATCH", "PASS"]) {
    if (md.indexOf("Verdict: " + EMOJI[label]) !== -1) return { emoji: EMOJI[label], label: label };
  }
  return null;
}

function extractOneLiner(md) {
  const lines = md.split("\n");
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith(">") && t.length > 5) return t.replace(/^>\s*/, "").trim();
  }
  return "";
}

function extractPTCS(md) {
  const idx = md.indexOf("Overall PTCS");
  if (idx === -1) return "";
  const chunk = md.slice(idx, idx + 80);
  const parts = chunk.split("|");
  return parts.length > 1 ? parts[1].trim() : "";
}

function renderMemo(company, md) {
  const verdict = extractVerdict(md);
  const oneLiner = extractOneLiner(md);
  const ptcs = extractPTCS(md);
  const confMatch = md.match(/\*\*Confidence:\*\*\s*(\w+)/);
  const conf = confMatch ? confMatch[1] : "";

  let bannerHtml = "";
  if (verdict) {
    const cls = verdict.label.toLowerCase();
    bannerHtml =
      '<div class="verdict-banner ' + cls + '">' +
      '<div class="verdict-emoji">' + verdict.emoji + '</div>' +
      '<div class="verdict-text ' + cls + '">' +
      '<h2>' + verdict.label + '</h2>' +
      '<p>' + oneLiner + '</p>' +
      '</div>' +
      '<div class="verdict-meta">' +
      (conf ? '<span>Confidence: ' + conf + '</span>' : '') +
      (ptcs ? '<span>PTCS: ' + ptcs + '</span>' : '') +
      '</div>' +
      '</div>';
  }

  const html = marked.parse(md);

  document.getElementById("rightPanel").innerHTML =
    '<div class="memo-content">' +
    '<div class="memo-toolbar">' +
    '<span class="memo-company">' + company + '</span>' +
    '<button class="toolbar-btn" onclick="copyMemo()">Copy Markdown</button>' +
    '<button class="toolbar-btn" onclick="downloadMemo()">Download .md</button>' +
    '</div>' +
    bannerHtml +
    html +
    '</div>';
}

function copyMemo() {
  navigator.clipboard.writeText(currentMemo).then(function() {
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(function() { btn.textContent = orig; }, 1500);
  });
}

function downloadMemo() {
  const slug = currentCompany.toLowerCase().replace(/\s+/g, "_");
  const blob = new Blob([currentMemo], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = slug + "_diligence.md";
  a.click();
}

async function loadHistory() {
  if (!currentCompany) return;
  try {
    const res = await fetch("/history/" + encodeURIComponent(currentCompany));
    const data = await res.json();
    const runs = (data.runs || []).slice().reverse().slice(0, 5);
    if (!runs.length) return;
    document.getElementById("history-divider").style.display = "block";
    document.getElementById("history-section").style.display = "block";
    const VMAP = { INVEST: "\u2705", WATCH: "\u26A0\uFE0F", PASS: "\u274C" };
    const list = document.getElementById("history-list");
    list.innerHTML = runs.map(function(r) {
      const em = VMAP[r.verdict] || "";
      const date = r.timestamp ? r.timestamp.slice(0, 10) : "";
      return '<div class="history-item">' +
        '<div class="hi-company">' + em + ' ' + r.verdict +
        ' <span style="font-weight:400;color:var(--muted)">\u00B7 ' + date + '</span></div>' +
        '<div class="hi-meta">' + (r.ic_one_liner ? r.ic_one_liner.slice(0, 80) + "\u2026" : r.run_id) + '</div>' +
        '</div>';
    }).join("");
  } catch(e) {}
}

document.addEventListener("DOMContentLoaded", function() {
  const co = document.getElementById("company");
  if (co) co.addEventListener("keydown", function(e) {
    if (e.key === "Enter") runDiligence();
  });

  const zone = document.getElementById("uploadZone");
  if (zone) {
    zone.addEventListener("dragover", function(e) {
      e.preventDefault(); zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", function() {
      zone.classList.remove("drag-over");
    });
    zone.addEventListener("drop", function(e) {
      e.preventDefault(); zone.classList.remove("drag-over");
      handleFiles(e.dataTransfer.files);
    });
  }
});
