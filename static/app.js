// CyberForge AML Graph Detection Analyst Interface

let activeCaseId = null;
let currentCases = [];
let animFrameId = null;
let graphParticles = [];

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadCases();
  loadMetrics();
  setInterval(loadCases, 3000);
  setInterval(pollReplayStatus, 2000);
});

function initEventListeners() {
  document.getElementById("budgetSlider").addEventListener("input", (e) => {
    document.getElementById("budgetValue").innerText = e.target.value;
    filterCaseList();
  });

  document.getElementById("btnReplayToggle").addEventListener("click", toggleReplay);
  document.getElementById("btnModelBenchmark").addEventListener("click", openBenchmarkModal);
  document.getElementById("modalCloseBtn").addEventListener("click", closeBenchmarkModal);

  document.getElementById("btnGalacticView").addEventListener("click", openGalacticModal);
  document.getElementById("galacticCloseBtn").addEventListener("click", closeGalacticModal);
  document.getElementById("btnCopyCypher").addEventListener("click", copyNeo4jCypher);

  document.getElementById("btnConfirmSAR").addEventListener("click", () => handleCaseAction("CONFIRMED_SAR"));
  document.getElementById("btnDismissCase").addEventListener("click", () => handleCaseAction("DISMISSED"));
  document.getElementById("btnInvestigate").addEventListener("click", () => handleCaseAction("UNDER_INVESTIGATION"));
}

// --- Case Queue Management ---

async function loadCases() {
  try {
    const res = await fetch("/cases");
    const data = await res.json();
    currentCases = data.cases || [];
    updateKpis();
    filterCaseList();

    if (!activeCaseId && currentCases.length > 0) {
      selectCase(currentCases[0].case_id);
    }
  } catch (err) {
    console.error("Error loading cases:", err);
  }
}

function updateKpis() {
  const total = currentCases.length;
  const critical = currentCases.filter(c => c.risk_score >= 0.80).length;
  const sars = currentCases.filter(c => c.status === "CONFIRMED_SAR").length;
  const totalVol = currentCases.reduce((acc, c) => acc + (c.total_volume_usd || 0), 0);

  document.getElementById("kpiTotalCases").innerText = total;
  document.getElementById("kpiCriticalCases").innerText = critical;
  document.getElementById("kpiConfirmedSARs").innerText = sars;
  document.getElementById("kpiLaunderedVol").innerText = `$${(totalVol / 1000).toFixed(1)}k`;
}

function filterCaseList() {
  const maxBudget = parseInt(document.getElementById("budgetSlider").value, 10);
  const container = document.getElementById("caseListContainer");
  container.innerHTML = "";

  const displayedCases = currentCases.slice(0, maxBudget);

  displayedCases.forEach(c => {
    const card = document.createElement("div");
    card.className = `case-card ${c.case_id === activeCaseId ? 'active' : ''}`;
    card.onclick = () => selectCase(c.case_id);

    const tier = c.risk_score >= 0.80 ? 'CRITICAL' : (c.risk_score >= 0.50 ? 'HIGH' : 'MEDIUM');

    card.innerHTML = `
      <div class="case-top-row">
        <span class="case-id-txt">${c.case_id}</span>
        <span class="tier-badge tier-${tier}">${tier} (${Math.round(c.risk_score * 100)}%)</span>
      </div>
      <div>
        <span class="topology-pill">${c.topology}</span>
        ${c.status === 'CONFIRMED_SAR' ? '<span class="tier-badge" style="background:#dc2626; color:#fff;">SAR FILED</span>' : ''}
      </div>
      <div class="case-desc">${c.primary_narrative || 'Multi-hop illicit flow detected.'}</div>
      <div class="case-meta-row">
        <span>${c.accounts_count} accounts · ${c.transaction_count} tx</span>
        <span>$${c.total_volume_usd.toLocaleString()}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

async function selectCase(caseId) {
  activeCaseId = caseId;
  filterCaseList();

  try {
    const res = await fetch(`/cases/${caseId}`);
    const c = await res.json();
    renderCaseDetail(c);
  } catch (err) {
    console.error("Error fetching case details:", err);
  }
}

function renderCaseDetail(c) {
  document.getElementById("detailCaseId").innerText = c.case_id;
  document.getElementById("detailTopology").innerText = c.topology;
  document.getElementById("detailRiskBadge").innerText = `${Math.round(c.risk_score * 100)}% Risk`;

  // Render Human Dossier
  const alerts = c.alerts || [];
  let dossier = null;
  if (alerts.length > 0 && alerts[0].dossier) {
    dossier = alerts[0].dossier;
  } else {
    // Fallback template
    dossier = {
      headline: `${c.topology} Multi-Account Syndicate Detected (${c.accounts_count} Accounts)`,
      summary: c.primary_narrative || "Cluster of interacting accounts flagged by graph topology algorithms.",
      why_flagged_points: [
        { icon: "🕸️", title: "Coordinated Ring Structure", detail: `${c.accounts_count} accounts across multiple banks exchanging $${c.total_volume_usd.toLocaleString()} within a tight timeframe.` },
        { icon: "⚠️", title: "Extreme Network Imbalance", detail: "Significant rapid transfer flow with near-zero holding duration." }
      ],
      typology_analogy: "Mule Ring: Multiple accounts collaborating to layer transactions and avoid institutional scrutiny.",
      recommended_action: "Place accounts under enhanced monitoring and request source of funds declaration."
    };
  }

  document.getElementById("dossierHeadline").innerText = dossier.headline;
  document.getElementById("dossierSummary").innerText = dossier.summary;
  document.getElementById("dossierAnalogyText").innerText = dossier.typology_analogy || "Coordinated multi-hop financial layering.";
  document.getElementById("dossierActionText").innerText = dossier.recommended_action || "Review account KYC and freeze suspicious outflows.";

  const whyGrid = document.getElementById("whyFlaggedPoints");
  whyGrid.innerHTML = "";
  (dossier.why_flagged_points || []).forEach(pt => {
    const item = document.createElement("div");
    item.className = "why-flagged-item";
    item.innerHTML = `
      <div class="why-flagged-icon">${pt.icon || '📌'}</div>
      <div>
        <div class="why-flagged-title">${pt.title}</div>
        <div class="why-flagged-desc">${pt.detail}</div>
      </div>
    `;
    whyGrid.appendChild(item);
  });

  // Render Subgraph
  let rootNode = c.accounts && c.accounts[0] ? c.accounts[0] : "ACC_ROOT";
  let subgraphData = null;

  if (alerts.length > 0 && alerts[0].subgraph) {
    subgraphData = alerts[0].subgraph;
  } else {
    subgraphData = buildSyntheticSubgraph(c);
  }

  drawNetworkGraph(subgraphData);

  // Render SHAP Feature Drivers
  renderShapDrivers(alerts.length > 0 ? alerts[0].drivers : []);

  // Render Member Transactions
  renderTransactionLedger(alerts);
}

function buildSyntheticSubgraph(c) {
  const nodes = (c.accounts || []).map((acc, i) => ({ id: acc, is_root: i === 0 }));
  const edges = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    edges.push({
      source: nodes[i].id,
      target: nodes[i + 1].id,
      amount: c.total_volume_usd / Math.max(1, nodes.length - 1)
    });
  }
  return { root_account: nodes[0]?.id || "", nodes, edges };
}

// --- Interactive Canvas Graph Engine ---

function drawNetworkGraph(subgraph) {
  const canvas = document.getElementById("subgraphCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  canvas.width = canvas.parentElement.clientWidth * dpr;
  canvas.height = canvas.parentElement.clientHeight * dpr;
  ctx.scale(dpr, dpr);

  const width = canvas.parentElement.clientWidth;
  const height = canvas.parentElement.clientHeight;

  if (animFrameId) cancelAnimationFrame(animFrameId);

  const nodes = subgraph.nodes || [];
  const edges = subgraph.edges || [];

  // Position nodes in a force-directed / radial ring layout
  const nodePositions = {};
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.38;

  nodes.forEach((n, idx) => {
    if (n.is_root || n.id === subgraph.root_account) {
      nodePositions[n.id] = { x: cx, y: cy, is_root: true, id: n.id };
    } else {
      const angle = (idx / (nodes.length - 1)) * 2 * Math.PI;
      nodePositions[n.id] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
        is_root: false,
        id: n.id
      };
    }
  });

  // Spawn flow particles
  graphParticles = [];
  edges.forEach((e) => {
    for (let p = 0; p < 2; p++) {
      graphParticles.push({
        source: e.source,
        target: e.target,
        progress: Math.random(),
        speed: 0.008 + Math.random() * 0.008
      });
    }
  });

  function renderFrame() {
    ctx.clearRect(0, 0, width, height);

    // 1. Draw Edges
    edges.forEach((e) => {
      const src = nodePositions[e.source];
      const dst = nodePositions[e.target];
      if (!src || !dst) return;

      ctx.beginPath();
      ctx.moveTo(src.x, src.y);
      ctx.lineTo(dst.x, dst.y);
      ctx.strokeStyle = "rgba(75, 85, 99, 0.45)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Amount Badge on Midpoint
      const mx = (src.x + dst.x) / 2;
      const my = (src.y + dst.y) / 2;
      ctx.fillStyle = "rgba(10, 14, 23, 0.8)";
      ctx.fillRect(mx - 22, my - 8, 44, 16);
      ctx.fillStyle = "#9ca3af";
      ctx.font = "9px 'JetBrains Mono'";
      ctx.textAlign = "center";
      ctx.fillText(`$${Math.round(e.amount || 0)}`, mx, my + 3);
    });

    // 2. Animate Money Flow Particles
    graphParticles.forEach((p) => {
      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;

      const src = nodePositions[p.source];
      const dst = nodePositions[p.target];
      if (!src || !dst) return;

      const px = src.x + (dst.x - src.x) * p.progress;
      const py = src.y + (dst.y - src.y) * p.progress;

      ctx.beginPath();
      ctx.arc(px, py, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "#f43f5e";
      ctx.shadowColor = "#f43f5e";
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // 3. Draw Nodes
    Object.values(nodePositions).forEach((n) => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.is_root ? 16 : 11, 0, 2 * Math.PI);
      ctx.fillStyle = n.is_root ? "#06b6d4" : "#8b5cf6";
      ctx.shadowColor = n.is_root ? "#06b6d4" : "#8b5cf6";
      ctx.shadowBlur = n.is_root ? 14 : 6;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Node ID label
      ctx.fillStyle = "#f9fafb";
      ctx.font = "10px 'JetBrains Mono'";
      ctx.textAlign = "center";
      ctx.fillText(n.id, n.x, n.y + (n.is_root ? 28 : 22));
    });

    animFrameId = requestAnimationFrame(renderFrame);
  }

  renderFrame();
}

function renderShapDrivers(drivers) {
  const container = document.getElementById("shapDriversContainer");
  container.innerHTML = "";

  if (!drivers || drivers.length === 0) {
    container.innerHTML = `<div style="font-size:0.8rem; color:#6b7280;">No high-marginal drivers for this baseline score.</div>`;
    return;
  }

  drivers.forEach((d) => {
    const item = document.createElement("div");
    item.className = "driver-bar-item";
    const widthPct = Math.min(100, Math.abs(parseFloat(d.impact) * 220));

    item.innerHTML = `
      <div class="driver-info">
        <span style="font-weight:600;">${d.name || d.feature}</span>
        <span style="font-family:'JetBrains Mono'; color:#f43f5e;">${d.impact}</span>
      </div>
      <div class="driver-track">
        <div class="driver-fill" style="width:${widthPct}%;"></div>
      </div>
    `;
    container.appendChild(item);
  });
}

function renderTransactionLedger(alerts) {
  const tbody = document.getElementById("txLedgerBody");
  tbody.innerHTML = "";

  if (!alerts || alerts.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#6b7280;">No individual transactions recorded.</td></tr>`;
    return;
  }

  alerts.forEach((tx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-family:'JetBrains Mono';">${tx.Timestamp || '-'}</td>
      <td style="font-family:'JetBrains Mono'; font-weight:600;">${tx.Account}</td>
      <td style="font-family:'JetBrains Mono';">${tx['Account.1']}</td>
      <td><span class="topology-pill">${tx.Payment_Format || 'Wire'}</span></td>
      <td style="font-family:'JetBrains Mono'; font-weight:700; color:#10b981;">$${(tx.Amount_Paid_USD || 0).toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  });
}

// --- Compliance Case Actions ---

async function handleCaseAction(actionType) {
  if (!activeCaseId) return;

  const notes = prompt(`Enter compliance rationale for ${actionType}:`, "Standard KYC / AML pattern review complete.");
  if (notes === null) return;

  try {
    const res = await fetch(`/cases/${activeCaseId}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision: actionType, notes })
    });
    const result = await res.json();
    alert(`Case ${activeCaseId} updated to: ${actionType}`);
    loadCases();
  } catch (err) {
    console.error("Action failed:", err);
  }
}

// --- Replay Controls ---

async function toggleReplay() {
  const btn = document.getElementById("btnReplayToggle");
  const isRunning = btn.getAttribute("data-running") === "true";

  if (!isRunning) {
    await fetch("/replay/start?speed=25.0", { method: "POST" });
    btn.setAttribute("data-running", "true");
    btn.innerText = "Pause Replay Stream";
    btn.style.background = "#e11d48";
  } else {
    await fetch("/replay/stop", { method: "POST" });
    btn.setAttribute("data-running", "false");
    btn.innerText = "Stream Replay Test Set";
    btn.style.background = "";
  }
}

async function pollReplayStatus() {
  try {
    const res = await fetch("/replay/status");
    const st = await res.json();
    if (st.is_running) {
      document.getElementById("replayStatusText").innerText = 
        `Streaming test feed: ${st.processed_count}/${st.total_count} tx scored (${st.alerts_generated} alerts)`;
    } else {
      document.getElementById("replayStatusText").innerText = "Stream Idle. Press to replay test slice.";
    }
  } catch (err) {}
}

// --- Model Benchmark Modal ---

async function openBenchmarkModal() {
  document.getElementById("benchmarkModal").style.display = "flex";
  await loadMetrics();
}

function closeBenchmarkModal() {
  document.getElementById("benchmarkModal").style.display = "none";
}

async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    const data = await res.json();
    if (!data.models) return;

    const tbody = document.getElementById("benchmarkTableBody");
    tbody.innerHTML = "";

    data.models.forEach((m) => {
      const p = m.pattern_recalls || {};
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:700;">${m.model_name}</td>
        <td style="font-family:'JetBrains Mono'; color:#06b6d4;">${m.pr_auc.toFixed(4)}</td>
        <td style="font-family:'JetBrains Mono';">${m.f1_score.toFixed(4)}</td>
        <td style="font-family:'JetBrains Mono'; font-weight:700; color:#10b981;">${((m.budget_metrics['Recall@Top100'] || 0) * 100).toFixed(1)}%</td>
        <td style="font-family:'JetBrains Mono';">${((p.cycle || 0) * 100).toFixed(1)}%</td>
        <td style="font-family:'JetBrains Mono';">${((p.scatter_gather || 0) * 100).toFixed(1)}%</td>
        <td style="font-family:'JetBrains Mono';">${((p.fan_out || 0) * 100).toFixed(1)}%</td>
        <td style="font-family:'JetBrains Mono';">${((p.fan_in || 0) * 100).toFixed(1)}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

// --- Galactic Graph (Neo4j Macro View) ---

let galacticAnimId = null;
let galacticData = null;

async function openGalacticModal() {
  const modal = document.getElementById("galacticModal");
  modal.style.display = "flex";

  try {
    const res = await fetch("/galactic/graph");
    galacticData = await res.json();
    setTimeout(() => initGalacticCanvas(), 60);
  } catch (err) {
    console.error("Error loading galactic graph:", err);
  }
}

function closeGalacticModal() {
  document.getElementById("galacticModal").style.display = "none";
  if (galacticAnimId) cancelAnimationFrame(galacticAnimId);
}

async function copyNeo4jCypher() {
  const btn = document.getElementById("btnCopyCypher");
  try {
    const res = await fetch("/neo4j/cypher");
    const data = await res.json();
    await navigator.clipboard.writeText(data.cypher);
    const orig = btn.innerText;
    btn.innerText = "✅ Cypher Copied to Clipboard!";
    btn.style.background = "#10b981";
    setTimeout(() => {
      btn.innerText = orig;
      btn.style.background = "";
    }, 2500);
  } catch (err) {
    alert("Cypher script is available at data/neo4j/import_and_queries.cypher");
  }
}

function initGalacticCanvas() {
  const canvas = document.getElementById("galacticCanvas");
  const tooltip = document.getElementById("galacticTooltip");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  const parent = canvas.parentElement;
  const width = parent.clientWidth || 1100;
  const height = parent.clientHeight || 650;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  const cx = width / 2;
  const cy = height / 2;

  if (galacticAnimId) cancelAnimationFrame(galacticAnimId);

  const nodes = galacticData.nodes || [];
  const edges = galacticData.edges || [];
  const nodeMap = {};

  nodes.forEach(n => {
    nodeMap[n.id] = {
      id: n.id,
      baseX: cx + n.x * 1.15,
      baseY: cy + n.y * 1.15,
      x: cx + n.x * 1.15,
      y: cy + n.y * 1.15,
      is_flagged: n.is_flagged,
      is_case: n.is_case,
      volume: n.volume || 1000,
      size: n.is_flagged ? 6.5 : 3.2
    };
  });

  // Pan & Zoom State
  let panX = 0;
  let panY = 0;
  let scale = 1.0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let hoveredNode = null;

  canvas.onmousedown = (e) => {
    isDragging = true;
    startX = e.clientX - panX;
    startY = e.clientY - panY;
    canvas.style.cursor = "grabbing";
  };

  window.onmousemove = (e) => {
    if (isDragging) {
      panX = e.clientX - startX;
      panY = e.clientY - startY;
    }
  };

  window.onmouseup = () => {
    isDragging = false;
    canvas.style.cursor = "grab";
  };

  canvas.onwheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
    scale = Math.min(3.5, Math.max(0.4, scale * zoomFactor));
  };

  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left - cx - panX) / scale + cx;
    const mouseY = (e.clientY - rect.top - cy - panY) / scale + cy;

    hoveredNode = null;
    for (const n of Object.values(nodeMap)) {
      const dx = mouseX - n.x;
      const dy = mouseY - n.y;
      if (dx * dx + dy * dy < 64) {
        hoveredNode = n;
        break;
      }
    }

    if (hoveredNode) {
      tooltip.style.display = "block";
      tooltip.style.left = `${e.clientX - rect.left + 16}px`;
      tooltip.style.top = `${e.clientY - rect.top + 16}px`;
      tooltip.innerHTML = `
        <div style="font-family:'JetBrains Mono'; font-weight:700; color:${hoveredNode.is_flagged ? '#f43f5e' : '#06b6d4'};">
          ${hoveredNode.id}
        </div>
        <div style="font-size:0.75rem; color:#f3f4f6; margin-top:2px;">
          ${hoveredNode.is_flagged ? '🚨 Flagged Illicit Syndicate Ring' : '✓ Legitimate Commercial Account'}
        </div>
        <div style="font-size:0.75rem; color:#9ca3af; margin-top:3px; font-family:'JetBrains Mono';">
          Turnover: $${Math.round(hoveredNode.volume).toLocaleString()}
        </div>
        ${hoveredNode.is_flagged ? '<div style="font-size:0.7rem; color:#f59e0b; margin-top:4px;">Click star to inspect case</div>' : ''}
      `;
    } else {
      tooltip.style.display = "none";
    }
  };

  canvas.onclick = () => {
    if (hoveredNode && hoveredNode.is_flagged) {
      // Find matching case
      const matchingCase = currentCases.find(c => (c.accounts || []).includes(hoveredNode.id));
      if (matchingCase) {
        closeGalacticModal();
        selectCase(matchingCase.case_id);
      }
    }
  };

  let pulseAngle = 0;

  function renderGalacticFrame() {
    ctx.clearRect(0, 0, width, height);
    pulseAngle += 0.04;

    ctx.save();
    // Center-based zoom and pan
    ctx.translate(cx + panX, cy + panY);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);

    // Cosmic background radial nebula
    const grad = ctx.createRadialGradient(cx, cy, 30, cx, cy, width * 0.6);
    grad.addColorStop(0, "rgba(6, 182, 212, 0.07)");
    grad.addColorStop(0.4, "rgba(139, 92, 246, 0.04)");
    grad.addColorStop(1, "rgba(4, 7, 17, 0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // 1. Draw Macro Galactic Edges
    ctx.lineWidth = 0.6 / scale;
    edges.forEach(e => {
      const src = nodeMap[e.source];
      const dst = nodeMap[e.target];
      if (!src || !dst) return;

      ctx.beginPath();
      ctx.moveTo(src.x, src.y);
      ctx.lineTo(dst.x, dst.y);
      ctx.strokeStyle = e.is_laundering ? "rgba(244, 63, 94, 0.45)" : "rgba(100, 116, 139, 0.12)";
      ctx.stroke();
    });

    // 2. Draw Stars (Accounts)
    nodes.forEach(n => {
      const pos = nodeMap[n.id];
      if (!pos) return;

      ctx.beginPath();
      let r = pos.size;
      const isHovered = hoveredNode && hoveredNode.id === pos.id;

      if (pos.is_flagged) {
        r += Math.sin(pulseAngle) * 2;
        ctx.fillStyle = "#f43f5e";
        ctx.shadowColor = "#f43f5e";
        ctx.shadowBlur = (14 + Math.sin(pulseAngle) * 8) * scale;
      } else {
        ctx.fillStyle = "rgba(139, 92, 246, 0.8)";
        ctx.shadowBlur = 0;
      }

      if (isHovered) {
        r += 4;
        ctx.fillStyle = "#38bdf8";
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = 18;
      }

      ctx.arc(pos.x, pos.y, Math.max(2, r), 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    ctx.restore();

    galacticAnimId = requestAnimationFrame(renderGalacticFrame);
  }

  renderGalacticFrame();
}
