let model = null;
let currentImage = null;

function getDetectionConfig() {
  const w = currentImage.naturalWidth,
    h = currentImage.naturalHeight;
  const pct = (id) => Number(document.getElementById(id).value) / 100;
  return {
    leftBand: { x: pct("lx") * w, y: pct("ly") * h, w: pct("lw") * w, h: pct("lh") * h },
    rightBand: { x: pct("rx") * w, y: pct("ry") * h, w: pct("rw") * w, h: pct("rh") * h },
    iconSize: pct("iconSize") * w,
    threshold: Number(document.getElementById("threshold").value),
  };
}

function renderTeam(containerId, side, detections) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  detections.forEach((d) => addEnemyRow(container, d.name, d.count));
  if (detections.length === 0) addEnemyRow(container, "", 1);
}

function addEnemyRow(container, name = "", count = 1) {
  const row = document.createElement("div");
  row.className = "enemyRow";
  row.innerHTML = `
    <input type="text" list="enemyNames" value="${name}" placeholder="enemy name">
    <input type="number" min="1" value="${count}">
    <button class="removeRow">x</button>`;
  row.querySelector(".removeRow").onclick = () => row.remove();
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

async function runDetection() {
  document.getElementById("status").textContent = "Detecting enemies…";
  const { left, right } = await detectEnemies(currentImage, getDetectionConfig(), model.enemies);
  renderTeam("teamAList", "A", left);
  renderTeam("teamBList", "B", right);
  document.getElementById("results").style.display = "block";
  document.getElementById("status").textContent =
    "Detection is best-effort -- please check/edit names and counts before predicting.";
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

document.getElementById("predictBtn").addEventListener("click", () => {
  const teamA = readTeam("teamAList");
  const teamB = readTeam("teamBList");
  const p = predictProba(model, teamA, teamB);
  document.getElementById("predictionOutput").textContent =
    `P(Team A wins) = ${(p * 100).toFixed(1)}%  |  P(Team B wins) = ${((1 - p) * 100).toFixed(1)}%`;
});

// Train on page load -- fast enough (well under a second for ~100 rows) to
// not need precomputed weights.
fetch("data/matches.json")
  .then((r) => r.json())
  .then((matches) => {
    document.getElementById("matchCount").textContent = matches.length;
    model = new EnemyStrengthModel(1.0, 0.75);
    myModel.fit(matches, 500);
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
    document.getElementById("status").textContent =
      `Model trained on ${matches.length} matches (${model.enemies.length} enemies). Paste a screenshot to begin.`;
  });
