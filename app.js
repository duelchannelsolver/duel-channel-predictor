let model = null;
let currentImage = null;
// Every enemy the detector may recognise: all sprites in sprites/ (via
// sprites/manifest.json), including ones with no match history yet.
let spriteNames = [];
let detectionRun = 0; // guards against an older detection finishing after a newer one

// detect.js v4 finds the icon row by itself (faint line + fixed slots), so
// there is nothing to configure here. debug:false skips the console table
// and overlay canvas.
function getDetectionConfig() {
  return { debug: false };
}

function renderTeam(containerId, detections) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  detections.forEach((d) => addEnemyRow(container, d.name, d.count));
  if (detections.length === 0) addEnemyRow(container, "", 1);
}

function addEnemyRow(container, name = "", count = 1) {
  const row = document.createElement("div");
  row.className = "enemyRow";
  row.innerHTML = `
    <input type="text" list="enemyNames" placeholder="enemy name">
    <input type="number" min="1">
    <button class="removeRow">x</button>`;
  const [nameInput, countInput] = row.querySelectorAll("input");
  nameInput.value = name; // set as properties so quotes in names can't break the markup
  countInput.value = count;
  row.querySelector(".removeRow").onclick = () => {
    row.remove();
    updatePrediction();
  };
  container.appendChild(row);
}

function readTeam(containerId) {
  const team = {};
  document.querySelectorAll(`#${containerId} .enemyRow`).forEach((row) => {
    const [nameInput, countInput] = row.querySelectorAll("input");
    const name = nameInput.value.trim();
    const count = Number(countInput.value) || 0;
    if (name && count > 0) team[name] = (team[name] || 0) + count;
  });
  return team;
}

// Splits a team into enemies the model has history for and ones it has never
// seen (recognised from sprites, but no matches recorded yet).
function splitByHistory(team) {
  const known = new Set(model.enemies);
  const kept = {}, unseen = {};
  for (const [name, count] of Object.entries(team)) {
    (known.has(name) ? kept : unseen)[name] = count;
  }
  return { kept, unseen };
}

function setPredictionNote(text) {
  let note = document.getElementById("predictionNote");
  if (!note) {
    note = document.createElement("div");
    note.id = "predictionNote";
    note.style.cssText = "font-size:0.85em;opacity:0.75;margin-top:4px";
    document.getElementById("predictionOutput").insertAdjacentElement("afterend", note);
  }
  note.textContent = text;
}

// Team A = left team, Team B = right team (internal naming only).
function updatePrediction() {
  const out = document.getElementById("predictionOutput");
  if (!model) return;
  setPredictionNote("");
  const L = splitByHistory(readTeam("teamAList"));
  const R = splitByHistory(readTeam("teamBList"));
  const ignoredText = [
    ...Object.entries(L.unseen).map(([n, c]) => `${n} ×${c} (left)`),
    ...Object.entries(R.unseen).map(([n, c]) => `${n} ×${c} (right)`),
  ].join(", ");

  const leftAny = Object.keys(L.kept).length + Object.keys(L.unseen).length;
  const rightAny = Object.keys(R.kept).length + Object.keys(R.unseen).length;
  if (!leftAny || !rightAny) {
    out.textContent = "Add at least one enemy to each team to see a prediction.";
    return;
  }
  if (!Object.keys(L.kept).length || !Object.keys(R.kept).length) {
    out.textContent = "Not enough match history to predict this matchup.";
    if (ignoredText) setPredictionNote(`No match history yet for: ${ignoredText}`);
    return;
  }
  let p;
  try {
    p = model.predictProba(L.kept, R.kept); // P(left team wins)
  } catch {
    out.textContent = "Couldn't compute a prediction -- check the enemy names.";
    return;
  }
  if (Math.abs(p - 0.5) < 0.0005) {
    out.textContent = "Winner: Toss-up, 50.0%";
  } else if (p > 0.5) {
    out.textContent = `Winner: Left Team, ${(p * 100).toFixed(1)}%`;
  } else {
    out.textContent = `Winner: Right Team, ${((1 - p) * 100).toFixed(1)}%`;
  }
  if (ignoredText) {
    setPredictionNote(`Not counted (no match history yet): ${ignoredText}`);
  }
}

async function runDetection() {
  const run = ++detectionRun;
  const status = document.getElementById("status");
  status.textContent = "Detecting enemies…";
  try {
    const { left, right } = await detectEnemies(currentImage, getDetectionConfig(), spriteNames);
    if (run !== detectionRun) return; // a newer screenshot was pasted meanwhile
    renderTeam("teamAList", left);
    renderTeam("teamBList", right);
    document.getElementById("results").style.display = "block";
    updatePrediction(); // predict immediately
    status.textContent =
      "Detection is best-effort -- please check/edit names and counts.";
  } catch (err) {
    console.error(err);
    if (run === detectionRun) status.textContent = "Detection failed -- see the console for details.";
  }
}

function handleImage(imgSrc) {
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    runDetection();
  };
  img.src = imgSrc;
}

document.addEventListener("paste", (e) => {
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith("image/")) {
      handleImage(URL.createObjectURL(item.getAsFile()));
    }
  }
});

document.getElementById("fileInput").addEventListener("change", (e) => {
  if (e.target.files[0]) handleImage(URL.createObjectURL(e.target.files[0]));
});

document.getElementById("redetect").addEventListener("click", () => {
  if (currentImage) runDetection();
});

document.querySelectorAll(".addRow").forEach((btn) => {
  btn.addEventListener("click", () => {
    const container = document.getElementById(
      btn.dataset.side === "A" ? "teamAList" : "teamBList"
    );
    addEnemyRow(container);
  });
});

// Keep the prediction live while the user corrects names/counts.
["teamAList", "teamBList"].forEach((id) => {
  document.getElementById(id).addEventListener("input", updatePrediction);
  document.getElementById(id).addEventListener("change", updatePrediction);
});

// --------------------------------------------------------------- page setup

// "Team A"/"Team B" -> "Left Team"/"Right Team" everywhere in the page text.
function relabelTeams() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const n of nodes) {
    const tag = n.parentElement && n.parentElement.tagName;
    if (tag === "SCRIPT" || tag === "STYLE" || tag === "TEXTAREA") continue;
    const t = n.nodeValue.replace(/\bTeam A\b/g, "Left Team").replace(/\bTeam B\b/g, "Right Team");
    if (t !== n.nodeValue) n.nodeValue = t;
  }
  document.title = document.title.replace(/\bTeam A\b/g, "Left Team").replace(/\bTeam B\b/g, "Right Team");
}

// Moves the band / icon-size / threshold controls behind an
// "Advanced Settings" toggle button (collapsed by default).
function setupAdvancedSettings() {
  const ids = ["lx", "ly", "lw", "lh", "rx", "ry", "rw", "rh", "iconSize", "threshold"];
  const inputs = ids.map((id) => document.getElementById(id)).filter(Boolean);
  if (!inputs.length) return;

  const mustStayVisible = ["fileInput", "redetect", "results", "status", "predictBtn"]
    .map((id) => document.getElementById(id))
    .filter(Boolean);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.id = "advancedToggle";
  toggle.textContent = "Advanced Settings";
  toggle.setAttribute("aria-expanded", "false");
  const panel = document.createElement("div");
  panel.id = "advancedPanel";
  panel.style.display = "none";
  toggle.addEventListener("click", () => {
    const open = panel.style.display === "none";
    panel.style.display = open ? "" : "none";
    toggle.setAttribute("aria-expanded", String(open));
  });

  // Best case: the controls live in their own block -> move the whole block.
  let box = inputs[0].parentElement;
  while (box && !inputs.every((el) => box.contains(el))) box = box.parentElement;
  const isolated =
    box && box !== document.body && !mustStayVisible.some((el) => box.contains(el));

  if (isolated) {
    box.parentNode.insertBefore(toggle, box);
    box.parentNode.insertBefore(panel, box);
    panel.appendChild(box);
  } else {
    // Otherwise move each control's own row (its <label> or parent element).
    const rows = [...new Set(inputs.map((el) => el.closest("label") || el.parentElement))];
    rows[0].parentNode.insertBefore(toggle, rows[0]);
    rows[0].parentNode.insertBefore(panel, rows[0]);
    rows.forEach((r) => panel.appendChild(r));
  }
}

// Auto-predict makes the manual button redundant.
function hidePredictButton() {
  const btn = document.getElementById("predictBtn");
  if (btn) btn.style.display = "none";
}

// The page no longer advertises training-set size.
function hideTrainingStats() {
  const el = document.getElementById("matchCount");
  if (!el) return;
  const host = el.parentElement;
  if (host && host !== document.body && host.textContent.length < 150) {
    host.style.display = "none";
  } else {
    el.style.display = "none";
  }
}

// ------------------------------------------------------------ copy to Excel

// One spreadsheet row, tab-separated, matching the columns:
//   Left Enemy 1 | # | Left Enemy 2 | # | Left Enemy 3 | #
//   Right Enemy A | # | Right Enemy B | # | Right Enemy C | #
// Empty slots become empty cells (so trailing tabs are intentional).
const SLOTS_PER_SIDE = 3;

function teamToCells(team) {
  const entries = Object.entries(team).slice(0, SLOTS_PER_SIDE); // on-screen order
  const cells = [];
  for (let i = 0; i < SLOTS_PER_SIDE; i++) {
    cells.push(entries[i] ? entries[i][0] : "", entries[i] ? String(entries[i][1]) : "");
  }
  return cells;
}

function buildExcelRow() {
  const left = readTeam("teamAList");
  const right = readTeam("teamBList");
  const overflow =
    Math.max(0, Object.keys(left).length - SLOTS_PER_SIDE) +
    Math.max(0, Object.keys(right).length - SLOTS_PER_SIDE);
  return {
    text: [...teamToCells(left), ...teamToCells(right)].join("\t"),
    empty: !Object.keys(left).length && !Object.keys(right).length,
    overflow,
  };
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fallback for non-HTTPS / older browsers.
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  if (!ok) throw new Error("copy failed");
}

function setupCopyButton() {
  const anchor = document.getElementById("predictionOutput");
  if (!anchor) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "copyExcelBtn";
  btn.textContent = "Copy for Excel";
  btn.style.marginTop = "8px";
  let timer = null;
  const flash = (msg) => {
    btn.textContent = msg;
    clearTimeout(timer);
    timer = setTimeout(() => (btn.textContent = "Copy for Excel"), 1800);
  };
  btn.addEventListener("click", async () => {
    const { text, empty, overflow } = buildExcelRow();
    if (empty) return flash("Nothing to copy");
    try {
      await copyText(text);
      flash(overflow ? `Copied (${overflow} extra enemy type(s) left out)` : "Copied!");
    } catch (err) {
      console.error(err);
      flash("Copy failed");
    }
  });
  anchor.insertAdjacentElement("afterend", btn);
}

relabelTeams();
setupAdvancedSettings();
setupCopyButton();
hidePredictButton();
hideTrainingStats();

// Reads sprites/manifest.json (see scripts/make-sprite-manifest.js). Falls
// back to just the enemies from the match data if it isn't there.
async function loadSpriteNames(modelEnemies) {
  try {
    const r = await fetch("sprites/manifest.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const list = await r.json();
    return [...new Set([...list, ...modelEnemies])].sort((a, b) => a.localeCompare(b));
  } catch (err) {
    console.warn("sprites/manifest.json unavailable; only enemies with match history can be detected.", err);
    return modelEnemies;
  }
}

// Train on page load -- fast enough (well under a second for ~100 rows) to
// not need precomputed weights.
fetch("data/matches.json")
  .then((r) => r.json())
  .then(async (matches) => {
    model = new EnemyStrengthModel({ nTrees: 200, maxDepth: 12, minSamplesLeaf: 3, stormWeight: 0.75 });
    model.fit(matches);
    spriteNames = await loadSpriteNames(model.enemies);
    // Populate the <datalist> so enemy-name text inputs autocomplete.
    const datalist = document.createElement("datalist");
    datalist.id = "enemyNames";
    spriteNames.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      datalist.appendChild(opt);
    });
    document.body.appendChild(datalist);
    // Start loading sprite templates + the OCR engine in the background so
    // the first paste doesn't have to wait for them.
    warmUp(spriteNames);
    document.getElementById("status").textContent = "Paste a screenshot to begin.";
  });