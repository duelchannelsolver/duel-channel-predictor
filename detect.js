// Enemy detector, v2: targets the round-queue strip near the bottom center
// of the screen (icon + "xN" badge for each enemy type in that round),
// rather than scanning the whole battlefield. These icons are clean,
// static renders -- much closer to the wiki sprite art than in-motion
// battlefield sprites -- and the count is printed right next to each one,
// so OCR replaces the old "count defaults to 1" guess.
//
// Still an MVP: slot spacing/count and the OCR crop box are unverified
// against enough real screenshots. Detections are editable, not final.

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

// Reads the small "xN" badge in an icon slot via OCR. Crops the
// bottom-right portion of the slot (where the badge sits) and upscales it,
// since OCR does much better on larger text. Returns 1 if unreadable.
async function readCount(slotCanvas) {
  const bx = slotCanvas.width * 0.45;
  const by = slotCanvas.height * 0.6;
  const badge = cropToCanvas(
    slotCanvas, bx, by, slotCanvas.width - bx, slotCanvas.height - by
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
    return 1; // OCR unavailable or failed -- fall back to a manual-edit default.
  }
}

// region: {x, y, w, h} in source-canvas pixel coords, bounding just the
// row of enemy icons on one side of the round counter. maxSlots divides
// the width evenly; slots with no confident template match are dropped.
async function detectSide(sourceCanvas, region, templates, maxSlots, threshold = 0.85) {
  const slotW = region.w / maxSlots;
  const results = [];
  for (let i = 0; i < maxSlots; i++) {
    const slotCanvas = cropToCanvas(
      sourceCanvas, region.x + i * slotW, region.y, slotW, region.h
    );
    const slotThumb = toGrayThumb(slotCanvas);

    let best = null;
    for (const t of templates) {
      const score = similarity(slotThumb, t.data);
      if (score >= threshold && (!best || score > best.score)) {
        best = { name: t.name, score };
      }
    }
    if (best) {
      const count = await readCount(slotCanvas);
      results.push({ slot: i, ...best, count });
    }
  }
  return results;
}

async function detectEnemies(imageEl, regions, enemyNames) {
  const canvas = document.createElement("canvas");
  canvas.width = imageEl.naturalWidth;
  canvas.height = imageEl.naturalHeight;
  canvas.getContext("2d").drawImage(imageEl, 0, 0);

  const templates = await loadTemplates(enemyNames);
  const [left, right] = await Promise.all([
    detectSide(canvas, regions.left, templates, regions.leftSlots),
    detectSide(canvas, regions.right, templates, regions.rightSlots),
  ]);
  return { left, right };
}
