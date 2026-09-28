// Enemy detector, v3: instead of fixed per-slot boxes (which broke once
// icon count/position varied between rounds -- more enemies push the row
// wider from center), this slides the same template-match window across a
// broad search band on each side and keeps local best-matches, similar in
// spirit to "find the circles" but reusing the matcher we already have
// instead of a separate geometric circle-detection pass.
//
// Still an MVP: search-band bounds, icon size, and OCR crop are estimated
// from a couple of example screenshots, not a large verified set.

const THUMB = 24; // template/window comparison size

// Templates are loaded in parallel (not one-at-a-time) and cached, so the
// cost is paid once per page load instead of on every detection.
let templatesPromise = null;

function getTemplates(enemyNames) {
  if (!templatesPromise) {
    templatesPromise = (async () => {
      const loaded = await Promise.all(
        enemyNames.map(async (name) => {
          try {
            const img = await loadImage(`sprites/${name}.png`);
            return { name, data: toColorThumb(img, 0.85) };
          } catch {
            return null; // sprite missing for this enemy -- skip
          }
        })
      );
      return loaded.filter(Boolean);
    })();
  }
  return templatesPromise;
}

// One shared OCR worker, created once. Tesseract.recognize() by itself
// builds a fresh worker (reloading WASM + language data) on every call,
// which is very slow when called once per detected icon.
let ocrWorkerPromise = null;

function getOcrWorker() {
  if (!ocrWorkerPromise) {
    ocrWorkerPromise = (async () => {
      const worker = await Tesseract.createWorker("eng");
      await worker.setParameters({
        tessedit_char_whitelist: "0123456789",
        tessedit_pageseg_mode: "7", // treat the crop as a single line of text
      });
      return worker;
    })();
  }
  return ocrWorkerPromise;
}

// Call once after the model loads so templates + OCR are ready (or at
// least underway) before the user pastes their first screenshot.
function warmUp(enemyNames) {
  getTemplates(enemyNames);
  getOcrWorker().catch(() => {});
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function toColorThumb(imgOrCanvas, innerCropFrac = 1.0) {
  // Crop to the central square region first (innerCropFrac < 1 strips
  // outer pixels -- used to exclude the circular border/frame that every
  // in-game queue icon has but the wiki sprite templates don't, since that
  // shared border was dominating the comparison over actual content).
  const srcW = imgOrCanvas.naturalWidth || imgOrCanvas.width;
  const srcH = imgOrCanvas.naturalHeight || imgOrCanvas.height;
  const cropW = srcW * innerCropFrac;
  const cropH = srcH * innerCropFrac;
  const cropX = (srcW - cropW) / 2;
  const cropY = (srcH - cropH) / 2;

  const c = document.createElement("canvas");
  c.width = THUMB;
  c.height = THUMB;
  const ctx = c.getContext("2d");
  ctx.drawImage(imgOrCanvas, cropX, cropY, cropW, cropH, 0, 0, THUMB, THUMB);
  const { data } = ctx.getImageData(0, 0, THUMB, THUMB);

  // Color (R,G,B), not collapsed to grayscale -- brightness alone doesn't
  // separate e.g. a light-colored enemy from a dark one if their shape
  // pattern happens to be similar, but color usually will.
  const n = THUMB * THUMB;
  const vec = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    vec[i] = data[i * 4];
    vec[n + i] = data[i * 4 + 1];
    vec[2 * n + i] = data[i * 4 + 2];
  }

  // Mean-center each channel. This is the key fix for the "everything
  // matches the same template" symptom: without it, cosine similarity is
  // dominated by whatever low-frequency pattern (like a border ring) is
  // common across all icons; centering removes that shared component and
  // leaves only the actual distinguishing variation.
  for (const offset of [0, n, 2 * n]) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += vec[offset + i];
    mean /= n;
    for (let i = 0; i < n; i++) vec[offset + i] -= mean;
  }
  return vec;
}

function cropToCanvas(sourceCanvas, x, y, w, h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  c.getContext("2d").drawImage(sourceCanvas, x, y, w, h, 0, 0, c.width, c.height);
  return c;
}

function similarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

// Reads the "xN" badge, which sits just below/overlapping the bottom of
// each icon. Crops a band starting partway down the icon and extending
// below it, then upscales for OCR. Returns 1 if unreadable.
async function readCount(sourceCanvas, x, y, size) {
  const badge = cropToCanvas(
    sourceCanvas, x, y + size * 0.75, size, size * 0.55
  );
  const upscaled = cropToCanvas(badge, 0, 0, badge.width, badge.height);
  upscaled.width = badge.width * 4;
  upscaled.height = badge.height * 4;
  upscaled.getContext("2d").drawImage(badge, 0, 0, upscaled.width, upscaled.height);

  try {
    const worker = await getOcrWorker();
    const { data: { text } } = await worker.recognize(upscaled);
    const match = text.match(/(\d+)/);
    return match ? parseInt(match[1], 10) : 1;
  } catch {
    return 1;
  }
}

// Slides an icon-sized window across `band` (a wide search area, not a
// precise slot box) at the given stride, scoring every position against
// every template. Keeps local maxima above `threshold` via simple
// non-max suppression on x-overlap, so multiple distinct icons in the
// same band are each found once regardless of exact spacing.
async function detectSide(sourceCanvas, band, templates, iconSize, threshold = 0.85) {
  const stride = iconSize / 3;
  const candidates = [];

  for (let x = band.x; x + iconSize <= band.x + band.w; x += stride) {
    for (let y = band.y; y + iconSize <= band.y + band.h; y += stride) {
      const windowCanvas = cropToCanvas(sourceCanvas, x, y, iconSize, iconSize);
      const windowThumb = toColorThumb(windowCanvas, 0.6);

      let best = null;
      for (const t of templates) {
        const score = similarity(windowThumb, t.data);
        if (score >= threshold && (!best || score > best.score)) {
          best = { name: t.name, score };
        }
      }
      if (best) candidates.push({ x, y, ...best });
    }
  }

  // Non-max suppression: sort best-first, drop anything overlapping (in x)
  // a already-kept, higher-scoring detection by more than half an icon.
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of candidates) {
    const overlaps = kept.some((k) => Math.abs(k.x - c.x) < iconSize * 0.5);
    if (!overlaps) kept.push(c);
  }
  // Duel Channel shows at most 3 enemy types per side, so anything beyond
  // the 3 best matches is a false positive (and would cost extra OCR calls).
  kept.length = Math.min(kept.length, 3);
  kept.sort((a, b) => a.x - b.x); // left-to-right, matches on-screen order

  const results = [];
  for (const k of kept) {
    const count = await readCount(sourceCanvas, k.x, k.y, iconSize);
    results.push({ name: k.name, score: k.score, count });
  }
  return results;
}

async function detectEnemies(imageEl, config, enemyNames) {
  const canvas = document.createElement("canvas");
  canvas.width = imageEl.naturalWidth;
  canvas.height = imageEl.naturalHeight;
  canvas.getContext("2d").drawImage(imageEl, 0, 0);

  console.time("detect");
  const templates = await getTemplates(enemyNames);
  console.log(`templates loaded: ${templates.length} of ${enemyNames.length}`);
  const [left, right] = await Promise.all([
    detectSide(canvas, config.leftBand, templates, config.iconSize, config.threshold),
    detectSide(canvas, config.rightBand, templates, config.iconSize, config.threshold),
  ]);
  console.timeEnd("detect");
  return { left, right };
}
