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
  const ENEMIES_URL = 'data/duel_channel_enemies.json';
  const MATCHES_URL = 'data/matches.json'; // only used by the fallback

  const $ = (id) => document.getElementById(id);
  const statusEl = $('status');
  const resultsEl = $('results');
  const outputEl = $('predictionOutput');
  const detailEl = $('predictionDetail');
  const infoToggle = $('infoToggle');
  const debugToggle = $('debugToggle');
  const debugPanel = $('debugPanel');
  const previewEl = $('preview');
  const lists = { A: $('teamAList'), B: $('teamBList') };

  let model = null;
  let enemyNames = [];   // names offered in the dropdowns and given to detect.js
  let lastImage = null;  // last pasted / chosen screenshot, for Re-detect
  let infoOpen = false;  // whether "Advanced Info" is expanded

  const setStatus = (text, isError) => {
    statusEl.textContent = text;
    statusEl.style.color = isError ? '#b00020' : '';
  };

  // Shows a plain message (or nothing) in the result area and hides the details
  const showMessage = (text) => {
    outputEl.textContent = text || '';
    detailEl.textContent = '';
    detailEl.style.display = 'none';
    infoToggle.style.display = 'none';
  };

  // Makes `button` show / hide `panel`; the panel starts hidden
  const makeToggle = (button, panel) => {
    const set = (open) => {
      panel.style.display = open ? '' : 'none';
      button.setAttribute('aria-expanded', String(open));
    };
    button.addEventListener('click', () => set(panel.style.display === 'none'));
    return set;
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
      console.warn('No pretrained model found (' + err.message + '); training in the browser instead.');
    }

    if (saved) {
      model = DuelStackModel.fromJSON(saved, enemies);
    } else {
      const matches = (await fetchJson(MATCHES_URL)).filter((m) => m.winner === 'A' || m.winner === 'B');
      setStatus(`No pretrained model found. Training on ${matches.length} matches in the browser; ` +
        'the page may freeze for a minute');
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
    if (model.outdated) {
      setStatus('model.json was trained with an older stack.js, so predictions may be wrong. ' +
        'Re-run train.js and upload the new model.json.', true);
    } else {
      setStatus(`Model ready. Paste a screenshot or enter the teams below.`);
    }

    if (typeof warmUp === 'function') warmUp(enemyNames); // preload sprites and the OCR worker
    resetOcrIfBroken(); // if the text reader failed to start, let it retry later
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
    remove.addEventListener('click', () => { row.remove(); showMessage(''); });

    for (const el of [nameInput, countInput]) {
      el.addEventListener('input', () => { showMessage(''); });
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
    if (!model) { showMessage('The model is still loading.'); return null; }
    const a = readTeam('A'), b = readTeam('B');
    const unknown = a.unknown.concat(b.unknown);
    if (unknown.length) {
      showMessage('Unknown enemy: ' + unknown.join(', ') + '. Pick names from the list.');
      return null;
    }
    if (!Object.keys(a.team).length || !Object.keys(b.team).length) {
      showMessage('Enter at least one enemy on each side.');
      return null;
    }

    const t0 = performance.now();
    const d = model.predictDetailed(a.team, b.team);
    const ms = Math.round(performance.now() - t0);

    const aWins = d.p >= 0.5;
    const conf = Math.round(100 * (aWins ? d.p : 1 - d.p));
    const pct = (v) => Math.round(100 * v) + '%';

    outputEl.textContent = `${aWins ? 'Team A (left)' : 'Team B (right)'} wins: ${conf}%`;
    // Details stay behind the "Advanced Info" button; it keeps its open/closed state
    detailEl.textContent =
      `Chance left team wins: ${pct(d.p)}. Linear model ${pct(d.wide)}, trees ${pct(d.xgb)}, ` +
      `simulator: left team won ${pct(d.simWinRate)} of simulated battles. Computed in ${ms} ms.`;
    infoToggle.style.display = '';
    detailEl.style.display = infoOpen ? '' : 'none';
    return { teamA: a.team, teamB: b.team, p: d.p };
  }

  // --------------------------------------------------------------------------
  // Copy to Excel
  // --------------------------------------------------------------------------
  const SLOTS = 3; // enemy types per side in the sheet

  async function writeClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try { await navigator.clipboard.writeText(text); return; } catch (err) { /* fall back below */ }
    }
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    if (!ok) throw new Error('The browser blocked copying');
  }

  // Copies one tab-separated row (current teams + prediction) for pasting into the sheet.
  // "Winner (L/R)" is left blank to fill in after the match.
  async function copyToExcel() {
    const res = predict(); // always copy what is on screen right now
    if (!res) return;
    const clean = (v) => String(v).replace(/[\t\r\n]+/g, ' ');
    const side = (team) => {
      const cells = [];
      const names = Object.keys(team);
      for (let i = 0; i < SLOTS; i++) {
        if (i < names.length) cells.push(clean(names[i]), team[names[i]]);
        else cells.push('', '');
      }
      return cells;
    };
    const leftWins = res.p >= 0.5;
    const prob = leftWins ? res.p : 1 - res.p;
    const row = [
      ...side(res.teamA), ...side(res.teamB),
      '', leftWins ? 'L' : 'R', (100 * prob).toFixed(1) + '%',
    ];
    const lines = [];
    lines.push(row.join('\t'));

    const btn = $('copyBtn');
    const tooMany = Object.keys(res.teamA).length > SLOTS || Object.keys(res.teamB).length > SLOTS;
    try {
      await writeClipboard(lines.join('\n'));
      btn.textContent = tooMany ? 'Copied (first 3 per side)' : 'Copied!';
    } catch (err) {
      console.error(err);
      btn.textContent = 'Copy failed';
    }
    setTimeout(() => { btn.textContent = 'Copy to Excel'; }, 1800);
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

  let lastImageUrl = null;
  const isBlob = (b) => typeof Blob !== 'undefined' && b instanceof Blob;

  function imageFromUrl(url, crossOrigin) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      if (crossOrigin) img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('The browser could not decode that image'));
      img.src = url;
    });
  }

  // Blob -> <img>. If the browser's <img> decoder rejects it, re-encode it
  // through createImageBitmap + canvas and try again.
  async function loadImageFromBlob(blob) {
    if (!isBlob(blob)) throw new Error('The clipboard did not hand over an image');
    if (blob.size === 0) throw new Error('The copied image was empty');
    let url = URL.createObjectURL(blob);
    let img;
    try {
      img = await imageFromUrl(url);
    } catch (err) {
      URL.revokeObjectURL(url);
      if (typeof createImageBitmap !== 'function') throw err;
      const bmp = await createImageBitmap(blob);
      const c = document.createElement('canvas');
      c.width = bmp.width; c.height = bmp.height;
      c.getContext('2d').drawImage(bmp, 0, 0);
      const png = await new Promise((resolve) => c.toBlob(resolve, 'image/png'));
      if (!png) throw err;
      url = URL.createObjectURL(png);
      img = await imageFromUrl(url);
    }
    // Keep this image's data alive while it is in use; release the previous one
    if (lastImageUrl) URL.revokeObjectURL(lastImageUrl);
    lastImageUrl = url;
    return img;
  }

  // Everything in a paste / drop that might be an image, best candidates first.
  // Must be called synchronously inside the event handler.
  function imageCandidates(dt) {
    const blobs = [], urls = [];
    if (!dt) return { blobs, urls };
    const seen = new Set();
    const addBlob = (f) => {
      if (!isBlob(f) || f.size === 0) return;
      if (f.type && !f.type.startsWith('image/')) return;
      const key = f.type + ':' + f.size;
      if (seen.has(key)) return;
      seen.add(key);
      blobs.push(f);
    };
    for (const f of Array.from(dt.files || [])) addBlob(f);
    for (const item of Array.from(dt.items || [])) {
      if (item.kind === 'file' && (!item.type || item.type.startsWith('image/'))) addBlob(item.getAsFile());
    }
    // PNG first: it is what screenshots are, and every browser decodes it
    blobs.sort((x, y) => (y.type === 'image/png') - (x.type === 'image/png'));
    // Images copied from a web page or chat app often arrive only as HTML
    let html = '';
    try { html = dt.getData('text/html') || ''; } catch (err) { /* not available */ }
    const m = /<img[^>]+src\s*=\s*["']([^"']+)["']/i.exec(html);
    if (m) urls.push(m[1].replace(/&amp;/g, '&'));
    return { blobs, urls };
  }

  // Last resort for pastes where the event carried no usable image (the
  // clipboard was not ready yet): ask the clipboard directly, twice.
  async function readClipboardImage() {
    if (!navigator.clipboard || !navigator.clipboard.read) return null;
    for (let attempt = 0; attempt < 2; attempt++) {
      if (attempt) await new Promise((resolve) => setTimeout(resolve, 300));
      try {
        for (const item of await navigator.clipboard.read()) {
          const type = item.types.includes('image/png') ? 'image/png' : item.types.find((t) => t.startsWith('image/'));
          if (type) return await item.getType(type);
        }
      } catch (err) {
        console.warn('Direct clipboard read failed:', err);
      }
    }
    return null;
  }

  // Tries each candidate until one decodes.
  async function loadFirstImage({ blobs, urls }, allowClipboardRead) {
    let lastErr = null;
    for (const blob of blobs) {
      try { return await loadImageFromBlob(blob); } catch (err) { lastErr = err; console.warn(err); }
    }
    for (const url of urls) {
      try {
        const img = await imageFromUrl(url, !url.startsWith('data:'));
        if (lastImageUrl) { URL.revokeObjectURL(lastImageUrl); lastImageUrl = null; }
        return img;
      } catch (err) { lastErr = err; console.warn(err); }
    }
    if (allowClipboardRead) {
      const blob = await readClipboardImage();
      if (blob) {
        try { return await loadImageFromBlob(blob); } catch (err) { lastErr = err; console.warn(err); }
      }
    }
    throw new Error(lastErr
      ? 'Could not read the image (' + lastErr.message + '). Copy the screenshot again and paste once more.'
      : 'No image found in what was pasted. Copy the screenshot itself (not a file or a link) and paste again.');
  }

  const errorText = (err) => (err && err.message) || String(err || 'unknown error');
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const withTimeout = (promise, ms, what) => Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(what + ' timed out after ' + ms / 1000 + ' s')), ms)),
  ]);

  // detect.js caches its OCR worker, including a failed start, for the whole
  // session; then every count silently reads as 1. Clear a failed one so the
  // next attempt starts fresh.
  function resetOcrIfBroken() {
    try {
      if (typeof getOcrWorker !== 'function' || typeof ocrWorkerPromise === 'undefined') return Promise.resolve();
      return Promise.race([getOcrWorker(), sleep(15000).then(() => { throw new Error('OCR start timed out'); })])
        .then(() => {}, (err) => {
          console.warn('Text reader failed to start; it will be restarted.', err);
          ocrWorkerPromise = null; // eslint-disable-line no-global-assign
        });
    } catch (err) { return Promise.resolve(); }
  }

  const DETECT_TIMEOUT_MS = 45000;
  let detecting = false;   // a detection is in progress
  let queued = false;      // a newer screenshot (or Re-detect) arrived meanwhile
  let firstDetection = true;

  // One detection of `image`, retried once if it throws or stalls.
  async function detectWithRetry(image, config) {
    for (let attempt = 1; ; attempt++) {
      try {
        return await withTimeout(detectEnemies(image, config, enemyNames), DETECT_TIMEOUT_MS, 'Detection');
      } catch (err) {
        console.error(`Detection attempt ${attempt} failed:`, err);
        if (attempt >= 2) throw err;
        setStatus('Detection hit a problem (' + errorText(err) + '); trying once more…');
        await resetOcrIfBroken();
        await sleep(300);
      }
    }
  }

  // Runs detection on the latest screenshot. Only one detection runs at a time:
  // pasting again while one is running queues the newest image instead of
  // starting a second detection that competes with the first.
  async function runDetection() {
    if (!lastImage) { setStatus('Paste or choose a screenshot first.'); return; }
    if (detecting) {
      queued = true;
      setStatus('Still reading the previous screenshot; the newest one is next. No need to paste again.');
      return;
    }
    detecting = true;
    try {
      await modelReady;
      do {
        queued = false;
        const image = lastImage;
        const t0 = performance.now();
        setStatus(firstDetection
          ? 'Reading the screenshot… (the first one is slower while icons and the text reader load)'
          : 'Reading the screenshot…');
        showMessage('');
        // The "Minimum match score" box in Advanced debug sets detect.js's minScore
        const minScore = Number($('threshold').value);
        const config = minScore > 0 && minScore <= 1 ? { minScore } : {};

        let found;
        try {
          found = await detectWithRetry(image, config);
        } catch (err) {
          if (!queued) {
            setStatus('Detection failed: ' + errorText(err) +
              '. Press Re-detect in Advanced debug to try again, or enter the teams by hand.', true);
          }
          continue;
        }
        firstDetection = false;
        if (queued) continue; // a newer screenshot arrived; show that one instead

        if (found.debugCanvas) showPreview(found.debugCanvas);
        setTeam('A', found.left);
        setTeam('B', found.right);
        resultsEl.style.display = '';

        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        if (found.left.length && found.right.length) {
          setStatus(`Found ${found.left.length} enemy type(s) on the left and ${found.right.length} on the right ` +
            `in ${secs} s. Check the names and counts, then Predict again if you change anything.`);
          predict();
        } else {
          setStatus('Could not find enemies on both sides. Enter the teams by hand, or open Advanced debug to see what was detected.', true);
        }
      } while (queued);
    } catch (err) {
      console.error(err);
      setStatus('Detection failed: ' + errorText(err), true);
    } finally {
      detecting = false;
    }
  }

  async function handleImage(candidates, allowClipboardRead) {
    try {
      lastImage = await loadFirstImage(candidates, allowClipboardRead);
      showPreview(lastImage);
      await runDetection();
    } catch (err) {
      console.error(err);
      setStatus(errorText(err), true);
    }
  }

  // --------------------------------------------------------------------------
  // Wiring
  // --------------------------------------------------------------------------
  document.addEventListener('paste', (ev) => {
    const cand = imageCandidates(ev.clipboardData);
    const types = Array.from((ev.clipboardData && ev.clipboardData.types) || []);
    const typing = ev.target && /^(INPUT|TEXTAREA)$/.test(ev.target.tagName);
    // Leave ordinary text pastes into the name / count boxes alone
    if (!cand.blobs.length && !cand.urls.length && (typing || types.includes('text/plain'))) return;
    ev.preventDefault();
    handleImage(cand, true);
  });

  $('fileInput').addEventListener('change', (ev) => {
    const files = Array.from(ev.target.files || []);
    if (files.length) handleImage({ blobs: files, urls: [] }, false);
    ev.target.value = ''; // so choosing the same file again still triggers
  });

  const pasteArea = $('paste-area');
  pasteArea.addEventListener('dragover', (ev) => ev.preventDefault());
  pasteArea.addEventListener('drop', (ev) => {
    ev.preventDefault();
    handleImage(imageCandidates(ev.dataTransfer), false);
  });

  makeToggle(debugToggle, debugPanel);
  infoToggle.addEventListener('click', () => {
    infoOpen = !infoOpen;
    detailEl.style.display = infoOpen ? '' : 'none';
    infoToggle.setAttribute('aria-expanded', String(infoOpen));
  });

  $('redetect').addEventListener('click', runDetection);
  $('predictBtn').addEventListener('click', predict);
  $('copyBtn').addEventListener('click', copyToExcel);
  for (const btn of document.querySelectorAll('.addRow')) {
    btn.addEventListener('click', () => { addRow(btn.dataset.side); showMessage(''); });
  }
})();