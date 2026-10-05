// Enemy detector, v6: Adaptive line detection + Circular Hough anchoring + Robust NCC & OCR
'use strict';

const SZ = 24;            // Comparison patch size (px)
const MASK_R = SZ * 0.42; // Circular mask radius
const SCALES = [0.65, 0.80, 0.95, 1.10];
const SHIFTS = [[0, 0], [0, -0.1], [0, 0.1], [-0.1, 0], [0.1, 0]];

// Nominal slot X offsets from screen centre / image width
const SLOT_X_NOMINAL = {
  left:  [-0.2007, -0.1450, -0.0889], // outer -> inner
  right: [ 0.0889,  0.1450,  0.2007], // inner -> outer
};

const DEFAULTS = {
  diameter: 0.054,         // Icon diameter / width (~5.4% of width)
  centerBelowLine: 0.038,  // Expected circle centre distance below line / width
  minScore: 0.52,          // Minimum NCC match threshold (real matches score 0.70+)
  emptyStd: 8.0,           // Below this luminance std-dev => empty slot
  slashMin: 18,            // Contrast required to count as diagonal empty slash
  slashFrac: 0.90,         // Fraction of line required to declare empty
  debug: true,
};

// Precompute circular evaluation mask
const CIRCLE_MASK = new Uint8Array(SZ * SZ);
let CIRCLE_PIXELS = 0;
for (let y = 0; y < SZ; y++) {
  for (let x = 0; x < SZ; x++) {
    if (Math.hypot(x + 0.5 - SZ / 2, y + 0.5 - SZ / 2) <= MASK_R) {
      CIRCLE_MASK[y * SZ + x] = 1;
      CIRCLE_PIXELS++;
    }
  }
}

// -----------------------------------------------------------------------------
// Image & Canvas Utilities
// -----------------------------------------------------------------------------
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

function cropToCanvas(src, x, y, w, h, outW = w, outH = h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(outW));
  c.height = Math.max(1, Math.round(outH));
  const ctx = c.getContext("2d");
  ctx.drawImage(src, x, y, w, h, 0, 0, c.width, c.height);
  return c;
}

function toVec(canvas) {
  const n = SZ * SZ;
  const { data } = canvas.getContext("2d").getImageData(0, 0, SZ, SZ);
  const rgb = new Float32Array(n * 3);
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    rgb[i] = data[i * 4];
    rgb[n + i] = data[i * 4 + 1];
    rgb[2 * n + i] = data[i * 4 + 2];
    mask[i] = CIRCLE_MASK[i] && data[i * 4 + 3] > 128 ? 1 : 0;
  }
  return { rgb, mask };
}

function alphaCrop(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  const { data } = g.getImageData(0, 0, w, h);

  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[(y * w + x) * 4 + 3] > 24) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return img;
  return cropToCanvas(c, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

function makeVariant(img, s, dx, dy) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const side = Math.max(iw, ih);
  const k = SZ / (s * side);
  const w = iw * k;
  const h = ih * k;
  const c = document.createElement("canvas");
  c.width = c.height = SZ;
  c.getContext("2d").drawImage(
    img, SZ / 2 - w / 2 - dx * SZ, SZ / 2 - h / 2 - dy * SZ, w, h
  );
  return toVec(c);
}

// -----------------------------------------------------------------------------
// Template Management (Per-Item Cached)
// -----------------------------------------------------------------------------
const templateCache = new Map();

async function loadTemplate(name) {
  if (templateCache.has(name)) return templateCache.get(name);

  const cleanName = name.replace(/["'\u2019]/g, '').trim();
  const candidates = [
    `sprites/${name}.png`,
    `sprites/${cleanName}.png`,
    `sprites/${encodeURIComponent(name)}.png`,
    `sprites/${name.toLowerCase()}.png`,
    `sprites/${cleanName.toLowerCase()}.png`,
  ];

  let img = null;
  for (const src of candidates) {
    try {
      img = await loadImage(src);
      if (img) break;
    } catch (_) {}
  }

  if (!img) return null;

  const cropped = alphaCrop(img);
  const variants = [];
  for (const s of SCALES) {
    for (const [dx, dy] of SHIFTS) {
      variants.push(makeVariant(cropped, s, dx, dy));
    }
  }

  const entry = { name, variants };
  templateCache.set(name, entry);
  return entry;
}

async function getTemplates(enemyNames = []) {
  let names = enemyNames && enemyNames.length ? enemyNames : [];
  if (!names.length) {
    try {
      const res = await fetch("sprites/manifest.json");
      if (res.ok) names = await res.json();
    } catch (_) {}
  }
  const loaded = await Promise.all(names.map(loadTemplate));
  return loaded.filter(Boolean);
}

function warmUp(enemyNames) {
  getTemplates(enemyNames).catch(() => {});
  getOcrWorker().catch(() => {});
}

// -----------------------------------------------------------------------------
// Step 1: Adaptive Horizontal Line Detection
// -----------------------------------------------------------------------------
function findLineY(canvas) {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext("2d");
  const yStart = Math.floor(H * 0.68);
  const yEnd = Math.floor(H * 0.92);
  const searchH = yEnd - yStart;

  const x0 = Math.floor(W * 0.28);
  const x1 = Math.floor(W * 0.72);
  const scanW = x1 - x0;

  const { data } = ctx.getImageData(x0, yStart, scanW, searchH);

  let bestY = null;
  let bestScore = -1;

  for (let y = 3; y < searchH - 3; y++) {
    let greyPixels = 0;
    let lineLum = 0;
    let flankLum = 0;

    for (let x = 0; x < scanW; x += 2) {
      const idx = (y * scanW + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const spread = Math.max(r, g, b) - Math.min(r, g, b);
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;

      // Grey tone check (~neutral color, luminance between 60 and 190)
      if (spread < 28 && lum > 60 && lum < 190) {
        greyPixels++;
        lineLum += lum;
      }

      const idxUp = ((y - 2) * scanW + x) * 4;
      const idxDn = ((y + 2) * scanW + x) * 4;
      flankLum += 0.5 * (
        (0.299 * data[idxUp] + 0.587 * data[idxUp + 1] + 0.114 * data[idxUp + 2]) +
        (0.299 * data[idxDn] + 0.587 * data[idxDn + 1] + 0.114 * data[idxDn + 2])
      );
    }

    const sampledCount = scanW / 2;
    const coverage = greyPixels / sampledCount;
    if (coverage < 0.22) continue;

    const avgLineLum = greyPixels > 0 ? lineLum / greyPixels : 0;
    const avgFlankLum = flankLum / sampledCount;
    const contrast = Math.abs(avgLineLum - avgFlankLum);
    const score = coverage * 1.5 + (contrast / 25.0);

    if (score > bestScore) {
      bestScore = score;
      bestY = yStart + y;
    }
  }

  // Fallback to ~82% screen height if no distinct line is identified
  const finalY = bestY !== null ? bestY : Math.round(H * 0.82);
  console.log(`[Line Detection] y=${finalY} (detected=${bestY !== null}, score=${bestScore.toFixed(2)})`);
  return finalY;
}

// -----------------------------------------------------------------------------
// Step 2: Radial Hough-Style Circle Center Refinement
// -----------------------------------------------------------------------------
function refineCircleCenter(canvas, initCx, initCy, D) {
  const W = canvas.width, H = canvas.height;
  const R = D / 2;
  const searchRange = Math.round(D * 0.35);

  const x0 = Math.max(0, Math.round(initCx - R - searchRange));
  const y0 = Math.max(0, Math.round(initCy - R - searchRange));
  const boxW = Math.min(W - x0, Math.round((R + searchRange) * 2));
  const boxH = Math.min(H - y0, Math.round((R + searchRange) * 2));

  if (boxW <= 0 || boxH <= 0) return { cx: initCx, cy: initCy };

  const { data } = canvas.getContext("2d").getImageData(x0, y0, boxW, boxH);
  const getLum = (px, py) => {
    const rx = Math.min(boxW - 1, Math.max(0, Math.round(px)));
    const ry = Math.min(boxH - 1, Math.max(0, Math.round(py)));
    const i = (ry * boxW + rx) * 4;
    return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  };

  const NUM_ANGLES = 16;
  const angles = [];
  for (let a = 0; a < NUM_ANGLES; a++) {
    const th = (a / NUM_ANGLES) * 2 * Math.PI;
    angles.push([Math.cos(th), Math.sin(th)]);
  }

  let bestCx = initCx, bestCy = initCy;
  let maxGrad = -1;

  for (let dy = -searchRange; dy <= searchRange; dy += 2) {
    for (let dx = -searchRange; dx <= searchRange; dx += 2) {
      const cxLocal = (initCx - x0) + dx;
      const cyLocal = (initCy - y0) + dy;
      let gradSum = 0;

      for (const [cosT, sinT] of angles) {
        const lumRim = getLum(cxLocal + R * cosT, cyLocal + R * sinT);
        const lumIn  = getLum(cxLocal + (R - 4) * cosT, cyLocal + (R - 4) * sinT);
        const lumOut = getLum(cxLocal + (R + 4) * cosT, cyLocal + (R + 4) * sinT);
        gradSum += Math.abs(lumRim - lumIn) + Math.abs(lumRim - lumOut);
      }

      if (gradSum > maxGrad) {
        maxGrad = gradSum;
        bestCx = initCx + dx;
        bestCy = initCy + dy;
      }
    }
  }

  return { cx: bestCx, cy: bestCy };
}

function getPatch(canvas, cx, cy, D) {
  const c = cropToCanvas(canvas, cx - D / 2, cy - D / 2, D, D, SZ, SZ);
  const v = toVec(c);

  let sum = 0, sum2 = 0, cnt = 0;
  const n = SZ * SZ;
  for (let i = 0; i < n; i++) {
    if (!v.mask[i]) continue;
    const l = 0.299 * v.rgb[i] + 0.587 * v.rgb[n + i] + 0.114 * v.rgb[2 * n + i];
    sum += l;
    sum2 += l * l;
    cnt++;
  }
  const mean = sum / (cnt || 1);
  v.std = Math.sqrt(Math.max(0, sum2 / (cnt || 1) - mean * mean));
  return v;
}

function slashFraction(canvas, cx, cy, D, cfg) {
  const n = Math.round(D);
  const c = cropToCanvas(canvas, cx - D / 2, cy - D / 2, n, n);
  const { data } = c.getContext("2d").getImageData(0, 0, n, n);
  const L = (x, y) => {
    const rx = Math.min(n - 1, Math.max(0, Math.round(x)));
    const ry = Math.min(n - 1, Math.max(0, Math.round(y)));
    const i = (ry * n + rx) * 4;
    return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  };

  const off = (0.07 * n) / Math.SQRT2;
  let best = 0;
  for (let o = -0.06 * n; o <= 0.06 * n; o += 1.0) {
    let ok = 0, tot = 0;
    for (let k = 0; k <= 18; k++) {
      const t = (0.28 + 0.024 * k) * n;
      const mid = L(t, t + o);
      const a = L(t + off, t + o - off);
      const b = L(t - off, t + o + off);
      if (mid - Math.max(a, b) > cfg.slashMin) ok++;
      tot++;
    }
    best = Math.max(best, ok / (tot || 1));
  }
  return best;
}

// -----------------------------------------------------------------------------
// Step 3: NCC Template Classification
// -----------------------------------------------------------------------------
function ncc(a, b) {
  const n = SZ * SZ;
  let cnt = 0;
  const ma = [0, 0, 0], mb = [0, 0, 0];

  for (let i = 0; i < n; i++) {
    if (!a.mask[i] || !b.mask[i]) continue;
    cnt++;
    for (let c = 0; c < 3; c++) {
      ma[c] += a.rgb[c * n + i];
      mb[c] += b.rgb[c * n + i];
    }
  }
  if (cnt < CIRCLE_PIXELS * 0.15) return -1;
  for (let c = 0; c < 3; c++) {
    ma[c] /= cnt;
    mb[c] /= cnt;
  }

  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    if (!a.mask[i] || !b.mask[i]) continue;
    for (let c = 0; c < 3; c++) {
      const x = a.rgb[c * n + i] - ma[c];
      const y = b.rgb[c * n + i] - mb[c];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
  }
  if (na === 0 || nb === 0) return -1;
  const covFactor = Math.min(1.0, Math.sqrt(cnt / (CIRCLE_PIXELS * 0.4)));
  return (dot / Math.sqrt(na * nb)) * covFactor;
}

function classify(patch, templates) {
  let best = { name: null, score: -1 };
  let second = { name: null, score: -1 };

  for (const t of templates) {
    let s = -1;
    for (const v of t.variants) {
      s = Math.max(s, ncc(patch, v));
    }
    if (s > best.score) {
      second = best;
      best = { name: t.name, score: s };
    } else if (s > second.score) {
      second = { name: t.name, score: s };
    }
  }
  return { ...best, runnerUp: second.name, margin: best.score - second.score };
}

// -----------------------------------------------------------------------------
// Step 4: Adaptive Number & Count OCR
// -----------------------------------------------------------------------------
let ocrWorkerPromise = null;
function getOcrWorker() {
  if (typeof Tesseract === "undefined") return Promise.reject(new Error("Tesseract not available"));
  if (!ocrWorkerPromise) {
    ocrWorkerPromise = (async () => {
      const worker = await Tesseract.createWorker("eng");
      await worker.setParameters({
        tessedit_char_whitelist: "0123456789xX\u00d7",
        tessedit_pageseg_mode: "7",
      });
      return worker;
    })();
  }
  return ocrWorkerPromise;
}

async function readCount(src, cx, cy, D, side) {
  const x0 = side === "left" ? cx + 0.05 * D : cx - 0.78 * D;
  const y0 = cy + 0.18 * D;
  const cropW = 0.75 * D;
  const cropH = 0.55 * D;

  const crop = cropToCanvas(src, x0, y0, cropW, cropH);
  const cw = crop.width, ch = crop.height;
  const { data } = crop.getContext("2d").getImageData(0, 0, cw, ch);

  let maxLum = 0;
  const lum = new Uint8ClampedArray(cw * ch);
  for (let i = 0; i < cw * ch; i++) {
    const l = Math.min(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
    lum[i] = l;
    if (l > maxLum) maxLum = l;
  }

  // If no bright white badge pixels are present, count defaults to 1
  if (maxLum < 135) return 1;

  const threshold = Math.max(130, maxLum - 50);
  let bx0 = cw, bx1 = -1, by0 = ch, by1 = -1;

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      if (lum[y * cw + x] >= threshold) {
        if (x < bx0) bx0 = x; if (x > bx1) bx1 = x;
        if (y < by0) by0 = y; if (y > by1) by1 = y;
      }
    }
  }

  if (bx1 < 0 || (bx1 - bx0 < 3)) return 1;

  // Separate leading 'x' or '×' badge if present on the left
  const glyphW = bx1 - bx0 + 1;
  const glyphH = by1 - by0 + 1;

  const grayCanvas = document.createElement("canvas");
  grayCanvas.width = glyphW + 24;
  grayCanvas.height = Math.max(48, glyphH + 24);
  const gctx = grayCanvas.getContext("2d");
  gctx.fillStyle = "#ffffff";
  gctx.fillRect(0, 0, grayCanvas.width, grayCanvas.height);

  const imgData = gctx.getImageData(0, 0, grayCanvas.width, grayCanvas.height);
  const offX = 12, offY = Math.round((grayCanvas.height - glyphH) / 2);

  for (let y = 0; y < glyphH; y++) {
    for (let x = 0; x < glyphW; x++) {
      const isInk = lum[(by0 + y) * cw + (bx0 + x)] >= threshold;
      const di = ((offY + y) * grayCanvas.width + (offX + x)) * 4;
      const v = isInk ? 0 : 255;
      imgData.data[di] = imgData.data[di + 1] = imgData.data[di + 2] = v;
    }
  }
  gctx.putImageData(imgData, 0, 0);

  try {
    const worker = await getOcrWorker();
    const { data: { text } } = await worker.recognize(grayCanvas);
    const clean = text.replace(/[^0-9xX\u00d7]/g, "");
    const match = clean.match(/[xX\u00d7](\d+)/) || clean.match(/(\d+)/);
    if (match) return parseInt(match[1], 10);
  } catch (_) {}

  return 1;
}

// -----------------------------------------------------------------------------
// Main Detection Routine
// -----------------------------------------------------------------------------
async function detectEnemies(imageEl, config = {}, enemyNames = []) {
  const cfg = { ...DEFAULTS, ...config };
  const canvas = document.createElement("canvas");
  canvas.width = imageEl.naturalWidth || imageEl.width;
  canvas.height = imageEl.naturalHeight || imageEl.height;
  canvas.getContext("2d").drawImage(imageEl, 0, 0);

  const W = canvas.width, H = canvas.height;
  const templates = await getTemplates(enemyNames);
  console.log(`[Detection] Templates active: ${templates.length}`);

  // 1. Detect thin horizontal divider line
  const lineY = findLineY(canvas);
  const D = Math.round(cfg.diameter * W);
  const nominalCy = lineY + Math.round(cfg.centerBelowLine * W);

  const out = { left: [], right: [] };
  const slotLog = [];

  // 2. Scan and refine the 3 slots per side
  for (const side of ["left", "right"]) {
    for (let i = 0; i < 3; i++) {
      const nominalCx = W / 2 + SLOT_X_NOMINAL[side][i] * W;
      const { cx, cy } = refineCircleCenter(canvas, nominalCx, nominalCy, D);

      const patch = getPatch(canvas, cx, cy, D);
      const slash = slashFraction(canvas, cx, cy, D, cfg);
      const log = {
        side, slot: i, cx: Math.round(cx), cy: Math.round(cy),
        std: +patch.std.toFixed(1), slash: +slash.toFixed(2),
      };

      // Empty check: strictly low variance or prominent slash
      if (patch.std < cfg.emptyStd || slash >= cfg.slashFrac) {
        log.result = "empty";
      } else {
        const m = classify(patch, templates);
        Object.assign(log, {
          name: m.name,
          score: +m.score.toFixed(3),
          runnerUp: m.runnerUp,
          margin: +m.margin.toFixed(3),
        });

        if (m.score >= cfg.minScore) {
          const count = await readCount(canvas, cx, cy, D, side);
          out[side].push({ name: m.name, score: m.score, count });
          log.count = count;
          log.result = "ok";
        } else {
          log.result = "low score";
        }
      }
      slotLog.push(log);
    }
  }

  if (cfg.debug) {
    console.table(slotLog);
    const dbg = document.createElement("canvas");
    dbg.width = W; dbg.height = H;
    const g = dbg.getContext("2d");
    g.drawImage(canvas, 0, 0);
    g.lineWidth = Math.max(2, W / 600);
    g.font = `${Math.round(W / 65)}px sans-serif`;

    // Draw detected line
    g.strokeStyle = "yellow";
    g.beginPath();
    g.moveTo(W * 0.25, lineY);
    g.lineTo(W * 0.75, lineY);
    g.stroke();

    // Draw detected circle slots
    for (const s of slotLog) {
      g.strokeStyle = s.result === "ok" ? "#00ff00" : s.result === "empty" ? "#888888" : "#ff3333";
      g.beginPath();
      g.arc(s.cx, s.cy, D / 2, 0, Math.PI * 2);
      g.stroke();

      g.fillStyle = g.strokeStyle;
      const text = s.result === "ok" ? `${s.name} x${s.count}` : (s.name ? `${s.name} (${s.score})` : s.result);
      g.fillText(text, s.cx - D / 2, s.cy - D / 2 - 6);
    }
    out.debugCanvas = dbg;
  }

  return out;
}

// Expose functions globally for the browser
if (typeof window !== "undefined") {
  window.detectEnemies = detectEnemies;
  window.warmUp = warmUp;
}