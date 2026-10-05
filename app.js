let model = null;
let currentImage = null;
let spriteNames = [];
let detectionRun = 0;
let isReady = false;

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
  nameInput.value = name;
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

function splitByHistory(team) {
  const known = new Set(model ? model.enemies : []);
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
  let breakdown = null;
  try {
    if (typeof model.predictDetailed === "function") {
      breakdown = model.predictDetailed(L.kept, R.kept);
      p = breakdown.p;
    } else {
      p = model.predictProba(L.kept, R.kept);
    }
  } catch (err) {
    console.error(err);
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

  const notes = [];
  if (breakdown) {
    notes.push(
      `Sub-models: NN ${(breakdown.nn * 100).toFixed(1)}% | LR ${(breakdown.lr * 100).toFixed(1)}% | RF ${(breakdown.rf * 100).toFixed(1)}%`
    );
  }
  if (ignoredText) notes.push(`Not counted (no match history): ${ignoredText}`);
  if (notes.length) setPredictionNote(notes.join(" • "));
}

async function runDetection() {
  const status = document.getElementById("status");
  if (!isReady || !spriteNames.length) {
    status.textContent = "Still loading sprite database, please wait…";
    return;
  }

  const run = ++detectionRun;
  status.textContent = "Detecting enemies…";
  try {
    const { left, right } = await detectEnemies(currentImage, getDetectionConfig(), spriteNames);
    if (run !== detectionRun) return;
    renderTeam("teamAList", left);
    renderTeam("teamBList", right);
    document.getElementById("results").style.display = "block";
    updatePrediction();
    status.textContent = "Detection is best-effort -- please check/edit names and counts.";
  } catch (err) {
    console.error(err);
    if (run === detectionRun) status.textContent = "Detection failed -- see console.";
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

["teamAList", "teamBList"].forEach((id) => {
  document.getElementById(id).addEventListener("input", updatePrediction);
  document.getElementById(id).addEventListener("change", updatePrediction);
});

// --------------------------------------------------------------- Setup & Load

async function loadSpriteNames(modelEnemies = []) {
  try {
    const r = await fetch("sprites/manifest.json");
    if (r.ok) {
      const list = await r.json();
      return [...new Set([...list, ...modelEnemies])].sort((a, b) => a.localeCompare(b));
    }
  } catch (_) {}
  return modelEnemies;
}

async function fetchJson(paths) {
  for (const p of paths) {
    try {
      const r = await fetch(p);
      if (r.ok) return await r.json();
    } catch (_) {}
  }
  throw new Error(`Could not load file from paths: ${paths.join(", ")}`);
}

async function init() {
  const status = document.getElementById("status");
  status.textContent = "Initializing stack model & sprites…";

  try {
    const [matches, enemiesData] = await Promise.all([
      fetchJson(["data/matches.json", "matches.json"]),
      fetchJson(["data/duel_channel_enemies.json", "duel_channel_enemies.json"]),
    ]);

    const StackClass = window.DuelStackModel || window.EnemyStrengthModel;
    if (!StackClass) {
      throw new Error("DuelStackModel not found. Ensure stack.js loads before app.js.");
    }

    model = new StackClass({ rfTrees: 100, nnEpochs: 60, folds: 3 });
    model.fit(matches, enemiesData);
    window.duelModel = model;

    spriteNames = await loadSpriteNames(model.enemies);

    const datalist = document.createElement("datalist");
    datalist.id = "enemyNames";
    spriteNames.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      datalist.appendChild(opt);
    });
    document.body.appendChild(datalist);

    if (typeof warmUp === "function") warmUp(spriteNames);
    isReady = true;
    status.textContent = "Paste a screenshot to begin.";
  } catch (err) {
    console.error("Initialization error:", err);
    status.textContent = "Failed to load model data. Check console.";
  }
}

init();