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

async function loadTemplates(enemyNames) {
  const templates = [];
  for (const name of enemyNames) {
    try {
      const img = await loadImage(`sprites/${name}.png`);
      templates.push({ name, data: toGrayThumb(img) });
    } catch {
      // Sprite not downloaded yet for this enemy -- skip silently.
    }
  }
  return templates;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function toGrayThumb(imgOrCanvas) {
  const c = document.createElement("canvas");
  c.width = THUMB;
  c.height = THUMB;
  const ctx = c.getContext("2d");
  ctx.drawImage(imgOrCanvas, 0, 0, THUMB, THUMB);
  const { data } = ctx.getImageData(0, 0, THUMB, THUMB);
  const gray = new Float32Array(THUMB * THUMB);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
  }
  return gray;
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
    const { data: { text } } = await Tesseract.recognize(upscaled, "eng", {
      tessedit_char_whitelist: "0123456789x\u00d7",
    });
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
      const windowThumb = toGrayThumb(windowCanvas);

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

  const templates = await loadTemplates(enemyNames);
  const [left, right] = await Promise.all([
    detectSide(canvas, config.leftBand, templates, config.iconSize, config.threshold),
    detectSide(canvas, config.rightBand, templates, config.iconSize, config.threshold),
  ]);
  return { left, right };
}
