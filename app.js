let model = null;
let currentImage = null;
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

// Team A = left team, Team B = right team (internal naming only).
function updatePrediction() {
  const out = document.getElementById("predictionOutput");
  if (!model) return;
  const left = readTeam("teamAList");
  const right = readTeam("teamBList");
  if (!Object.keys(left).length || !Object.keys(right).length) {
    out.textContent = "Add at least one enemy to each team to see a prediction.";
    return;
  }
  let p;
  try {
    p = model.predictProba(left, right); // P(left team wins)
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
}

async function runDetection() {
  const run = ++detectionRun;
  const status = document.getElementById("status");
  status.textContent = "Detecting enemies…";
  try {
    const { left, right } = await detectEnemies(currentImage, getDetectionConfig(), model.enemies);
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

relabelTeams();
setupAdvancedSettings();
hidePredictButton();
hideTrainingStats();

// Train on page load -- fast enough (well under a second for ~100 rows) to
// not need precomputed weights.
fetch("data/matches.json")
  .then((r) => r.json())
  .then((matches) => {
    model = new EnemyStrengthModel(1.0, 0.75);
    model.fit(matches, 1000);
    // Populate the <datalist> so enemy-name text inputs autocomplete.
    const datalist = document.createElement("datalist");
    datalist.id = "enemyNames";
    model.enemies.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      datalist.appendChild(opt);
    });
    document.body.appendChild(datalist);
    // Start loading sprite templates + the OCR engine in the background so
    // the first paste doesn't have to wait for them.
    warmUp(model.enemies);
    document.getElementById("status").textContent = "Paste a screenshot to begin.";
  });