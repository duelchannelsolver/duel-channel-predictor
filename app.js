/**
 * app.js - page logic for the Duel Channel Predictor.
 *
 * Needs, next to index.html:
 *   stack.js                   the model code (defines DuelStackModel)
 *   detect.js                  screenshot detection (defines detectEnemies, warmUp)
 *   model.json                 trained model, made with `node train.js`
 *   duel_channel_enemies.json  the enemies file the model was trained with
 *   sprites/<name>.png         enemy icons used by detect.js
 *
 * If model.json is missing, the page falls back to training in the browser
 * from matches.json, which is slow.
 */
(function () {
  'use strict';

  const MODEL_URL = 'model.json';
  const ENEMIES_URL = 'duel_channel_enemies.json';
  const MATCHES_URL = 'matches.json'; // only used by the fallback

  const $ = (id) => document.getElementById(id);
  const statusEl = $('status');
  const resultsEl = $('results');
  const outputEl = $('predictionOutput');
  const previewEl = $('preview');
  const lists = { A: $('teamAList'), B: $('teamBList') };

  let model = null;
  let enemyNames = [];   // names offered in the dropdowns and given to detect.js
  let lastImage = null;  // last pasted / chosen screenshot, for Re-detect
  let detectRun = 0;     // guards against overlapping detections

  const setStatus = (text, isError) => {
    statusEl.textContent = text;
    statusEl.style.color = isError ? '#b00020' : '';
  };

  const fetchJson = async (url) => {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return res.json();
  };

  // --------------------------------------------------------------------------
  // Model
  // --------------------------------------------------------------------------
  async function loadModel() {
    setStatus('Loading model…');
    const enemies = await fetchJson(ENEMIES_URL);

    let saved = null;
    try {
      saved = await fetchJson(MODEL_URL);
    } catch (err) {
      console.warn('No trained model found (' + err.message + '); training in the browser instead.');
    }

    if (saved) {
      model = DuelStackModel.fromJSON(saved, enemies);
    } else {
      const matches = (await fetchJson(MATCHES_URL)).filter((m) => m.winner === 'A' || m.winner === 'B');
      setStatus(`No model.json found. Training on ${matches.length} matches in the browser; ` +
        'the page will freeze for a minute or more…');
      await new Promise((resolve) => setTimeout(resolve, 50)); // let the message paint first
      console.time('fit');
      model = new DuelStackModel();
      model.fit(matches, enemies);
      console.timeEnd('fit');
    }

    enemyNames = model.enemies.slice();
    const dl = document.createElement('datalist');
    dl.id = 'enemyNames';
    for (const name of enemyNames) {
      const opt = document.createElement('option');
      opt.value = name;
      dl.appendChild(opt);
    }
    document.body.appendChild(dl);

    resultsEl.style.display = '';
    if (!lists.A.children.length) addRow('A');
    if (!lists.B.children.length) addRow('B');
    setStatus(`Model ready (${enemyNames.length} enemy types). Paste a screenshot or enter the teams below.`);

    if (typeof warmUp === 'function') warmUp(enemyNames); // preload sprites and the OCR worker
  }

  const modelReady = loadModel().catch((err) => {
    console.error(err);
    setStatus('Could not load the model: ' + err.message, true);
    throw err;
  });

  // --------------------------------------------------------------------------
  // Team editor
  // --------------------------------------------------------------------------
  function addRow(side, name = '', count = 1) {
    const row = document.createElement('div');
    row.className = 'enemyRow';

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'enemyName';
    nameInput.setAttribute('list', 'enemyNames');
    nameInput.placeholder = 'enemy name';
    nameInput.value = name;

    const countInput = document.createElement('input');
    countInput.type = 'number';
    countInput.className = 'enemyCount';
    countInput.min = '1';
    countInput.step = '1';
    countInput.value = String(count);
    countInput.size = 4;

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.title = 'Remove';
    remove.addEventListener('click', () => { row.remove(); outputEl.textContent = ''; });

    for (const el of [nameInput, countInput]) {
      el.addEventListener('input', () => { outputEl.textContent = ''; });
      el.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') predict(); });
    }

    row.append(nameInput, ' × ', countInput, ' ', remove);
    lists[side].appendChild(row);
    return row;
  }

  function setTeam(side, entries) {
    lists[side].textContent = '';
    for (const e of entries) addRow(side, e.name, e.count);
    if (!entries.length) addRow(side);
  }

  // Reads one side's rows into { name: count }, flagging names the model can't use.
  function readTeam(side) {
    const team = {};
    const unknown = [];
    for (const row of lists[side].children) {
      const name = row.querySelector('.enemyName').value.trim();
      const count = Math.round(Number(row.querySelector('.enemyCount').value));
      if (!name) continue;
      if (!(count >= 1)) continue;
      if (model._expandTeam({ [name]: 1 }).length === 0) { unknown.push(name); continue; }
      // Use the spelling the model was trained with when there is one
      const known = enemyNames.find((n) => n.toLowerCase() === name.toLowerCase()) || name;
      team[known] = (team[known] || 0) + count;
    }
    return { team, unknown };
  }

  // --------------------------------------------------------------------------
  // Prediction
  // --------------------------------------------------------------------------
  function predict() {
    if (!model) { outputEl.textContent = 'The model is still loading.'; return; }
    const a = readTeam('A'), b = readTeam('B');
    const unknown = a.unknown.concat(b.unknown);
    if (unknown.length) {
      outputEl.textContent = 'Unknown enemy: ' + unknown.join(', ') + '. Pick names from the list.';
      return;
    }
    if (!Object.keys(a.team).length || !Object.keys(b.team).length) {
      outputEl.textContent = 'Enter at least one enemy on each side.';
      return;
    }

    const t0 = performance.now();
    const d = model.predictDetailed(a.team, b.team);
    const ms = Math.round(performance.now() - t0);

    const aWins = d.p >= 0.5;
    const conf = Math.round(100 * (aWins ? d.p : 1 - d.p));
    const pct = (v) => Math.round(100 * v) + '%';

    outputEl.textContent = '';
    const headline = document.createElement('div');
    headline.textContent = `${aWins ? 'Team A (left)' : 'Team B (right)'} wins: ${conf}%`;
    const detail = document.createElement('small');
    detail.style.fontWeight = 'normal';
    detail.textContent =
      `Chance team A wins: ${pct(d.p)} (linear model ${pct(d.wide)}, trees ${pct(d.xgb)}, ` +
      `simulator ${pct(d.simWinRate)} of battles) · ${ms} ms`;
    outputEl.append(headline, detail);
  }

  // --------------------------------------------------------------------------
  // Screenshot handling
  // --------------------------------------------------------------------------
  function showPreview(source) {
    const w = source.naturalWidth || source.width, h = source.naturalHeight || source.height;
    previewEl.width = w;
    previewEl.height = h;
    previewEl.getContext('2d').drawImage(source, 0, 0);
    previewEl.style.display = 'block';
    previewEl.style.maxWidth = '100%';
    previewEl.style.height = 'auto';
  }

  function loadImageFromBlob(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read that image')); };
      img.src = url;
    });
  }

  async function runDetection() {
    if (!lastImage) { setStatus('Paste or choose a screenshot first.'); return; }
    const run = ++detectRun;
    try {
      await modelReady;
      setStatus('Reading the screenshot…');
      outputEl.textContent = '';
      const found = await detectEnemies(lastImage, {}, enemyNames);
      if (run !== detectRun) return; // a newer screenshot replaced this one

      if (found.debugCanvas) showPreview(found.debugCanvas);
      setTeam('A', found.left);
      setTeam('B', found.right);
      resultsEl.style.display = '';

      if (found.left.length && found.right.length) {
        setStatus(`Found ${found.left.length} enemy type(s) on the left and ${found.right.length} on the right. ` +
          'Check the names and counts, then Predict again if you change anything.');
        predict();
      } else {
        setStatus('Could not find enemies on both sides. Check the overlay, or enter the teams by hand.', true);
      }
    } catch (err) {
      if (run !== detectRun) return;
      console.error(err);
      setStatus('Detection failed: ' + err.message, true);
    }
  }

  async function handleImageBlob(blob) {
    try {
      lastImage = await loadImageFromBlob(blob);
      showPreview(lastImage);
      await runDetection();
    } catch (err) {
      console.error(err);
      setStatus(err.message, true);
    }
  }

  // --------------------------------------------------------------------------
  // Wiring
  // --------------------------------------------------------------------------
  document.addEventListener('paste', (ev) => {
    const items = (ev.clipboardData && ev.clipboardData.items) || [];
    for (const item of items) {
      if (item.type && item.type.startsWith('image/')) {
        ev.preventDefault();
        handleImageBlob(item.getAsFile());
        return;
      }
    }
  });

  $('fileInput').addEventListener('change', (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (file) handleImageBlob(file);
  });

  const pasteArea = $('paste-area');
  pasteArea.addEventListener('dragover', (ev) => ev.preventDefault());
  pasteArea.addEventListener('drop', (ev) => {
    ev.preventDefault();
    const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageBlob(file);
  });

  $('redetect').addEventListener('click', runDetection);
  $('predictBtn').addEventListener('click', predict);
  for (const btn of document.querySelectorAll('.addRow')) {
    btn.addEventListener('click', () => { addRow(btn.dataset.side); outputEl.textContent = ''; });
  }
})();