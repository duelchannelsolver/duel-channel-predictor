// Enemy detector, v4: geometry-anchored.
//
// The queue is always 3 slots per side (fixed positions, filled from the
// centre outwards), sitting a fixed distance below a faint horizontal line
// (~rgb(111,114,109)). The game UI scales with a blend of width and height
// (Unity-style match=0.5), so all distances are expressed in units of
// U = sqrt(width * height). Verified on 1918x1078, 1486x805 and 2400x1080
// screenshots (aspect ratios 1.78 - 2.22):
//   1. find the faint line  -> gives the vertical anchor
//   2. 6 fixed slot centres -> no sliding window
//   3. skip empty slots (low pixel variance)
//   4. classify with alpha-masked, multi-scale NCC against the sprites
//   5. OCR the "xN" badge after binarizing the white text
//
// Tune via the `config` argument if needed.

const SZ = 24;                 // comparison size (px)
const MASK_R = SZ * 0.42;      // ignore the icon's border ring
const SCALES = [0.5, 0.7, 0.9, 1.15];        // <1 = zoomed in on the sprite
const SHIFTS = [[0, 0], [0, -0.15], [0, 0.15]]; // [dx, dy] as fraction of view

const SLOT_X = {               // slot centre offset from image centre, in U
  left:  [-0.2685, -0.1930, -0.1188],  // outer -> inner
  right: [0.1188, 0.1930, 0.2685],     // inner -> outer
};

const DEFAULTS = {
  lineColor: [111, 114, 109],
  lineTol: 60,            // sum of |dR|+|dG|+|dB| allowed
  centerBelowLine: 0.0485, // slot centre y = lineY + this * U
  diameter: 0.070,         // icon diameter, in U
  fallbackCenterY: 0.90,   // * height, only if the line isn't found
  emptyStd: 28,           // luminance std-dev below this => empty slot
  minScore: 0.30,         // best NCC below this => unknown, skipped
  debug: true,
};

// Precomputed circular mask.
const CIRCLE = new Uint8Array(SZ * SZ);
let CIRCLE_COUNT = 0;
for (let y = 0; y < SZ; y++) {
  for (let x = 0; x < SZ; x++) {
    const d = Math.hypot(x + 0.5 - SZ / 2, y + 0.5 - SZ / 2);
    if (d <= MASK_R) { CIRCLE[y * SZ + x] = 1; CIRCLE_COUNT++; }
  }
}

// ---------------------------------------------------------------- helpers

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function cropToCanvas(src, x, y, w, h, outW = w, outH = h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(outW));
  c.height = Math.max(1, Math.round(outH));
  c.getContext("2d").drawImage(src, x, y, w, h, 0, 0, c.width, c.height);
  return c;
}

// canvas (SZxSZ) -> planar RGB + mask (circle AND opaque)
function toVec(canvas) {
  const n = SZ * SZ;
  const { data } = canvas.getContext("2d").getImageData(0, 0, SZ, SZ);
  const rgb = new Float32Array(n * 3);
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    rgb[i] = data[i * 4];
    rgb[n + i] = data[i * 4 + 1];
    rgb[2 * n + i] = data[i * 4 + 2];
    mask[i] = CIRCLE[i] && data[i * 4 + 3] > 128 ? 1 : 0;
  }
  return { rgb, mask };
}

// ------------------------------------------------------------- templates

// Renders the sprite zoomed by `s`, shifted by (dx,dy), onto a transparent
// SZxSZ canvas. Alpha is kept so transparent sprite background is ignored.
function makeVariant(img, s, dx, dy) {
  const side = Math.max(img.naturalWidth, img.naturalHeight);
  const k = SZ / (s * side);
  const w = img.naturalWidth * k;
  const h = img.naturalHeight * k;
  const c = document.createElement("canvas");
  c.width = c.height = SZ;
  c.getContext("2d").drawImage(
    img, SZ / 2 - w / 2 - dx * SZ, SZ / 2 - h / 2 - dy * SZ, w, h
  );
  return toVec(c);
}

let templatesPromise = null;
function getTemplates(enemyNames) {
  if (!templatesPromise) {
    templatesPromise = (async () => {
      const loaded = await Promise.all(
        enemyNames.map(async (name) => {
          try {
            const img = await loadImage(`sprites/${name}.png`);
            const variants = [];
            for (const s of SCALES)
              for (const [dx, dy] of SHIFTS)
                variants.push(makeVariant(img, s, dx, dy));
            return { name, variants };
          } catch {
            return null;
          }
        })
      );
      return loaded.filter(Boolean);
    })();
  }
  return templatesPromise;
}

// ------------------------------------------------------------------- OCR

let ocrWorkerPromise = null;
function getOcrWorker() {
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

function warmUp(enemyNames) {
  getTemplates(enemyNames);
  getOcrWorker().catch(() => {});
}

// The badge sits at the bottom, on the side facing the screen centre
// (right of left-side icons, left of right-side icons). Its text is white
// with a dark outline, so we keep only near-white pixels -> black on white.
async function readCount(src, cx, cy, D, side) {
  const x0 = side === "left" ? cx + 0.05 * D : cx - 0.75 * D;
  const crop = cropToCanvas(src, x0, cy + 0.22 * D, 0.7 * D, 0.5 * D);
  const ctx = crop.getContext("2d");
  const img = ctx.getImageData(0, 0, crop.width, crop.height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = Math.min(d[i], d[i + 1], d[i + 2]) > 200 ? 0 : 255;
    d[i] = d[i + 1] = d[i + 2] = v;
    d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  const PAD = 20, UP = 4;
  const big = document.createElement("canvas");
  big.width = crop.width * UP + PAD * 2;
  big.height = crop.height * UP + PAD * 2;
  const bctx = big.getContext("2d");
  bctx.fillStyle = "#fff";
  bctx.fillRect(0, 0, big.width, big.height);
  bctx.drawImage(crop, PAD, PAD, crop.width * UP, crop.height * UP);

  try {
    const worker = await getOcrWorker();
    const { data: { text } } = await worker.recognize(big);
    const clean = text.replace(/[^0-9xX\u00d7]/g, "");
    const m = clean.match(/[xX\u00d7](\d+)/) || clean.match(/(\d+)/);
    return m ? parseInt(m[1], 10) : 1;
  } catch {
    return 1;
  }
}

// -------------------------------------------------------------- geometry

// Finds the row (in the bottom quarter, central band) whose pixels best
// match the faint line colour. The line spans about +-0.285U from the
// centre; we sample 0.09U..0.27U on each side, skipping the middle where
// the ROUND badge / flame covers it.
function findLineY(canvas, cfg, U) {
  const W = canvas.width, H = canvas.height;
  const y0 = Math.floor(H * 0.75);
  const x0 = Math.max(0, Math.round(W / 2 - 0.27 * U));
  const x1 = Math.min(W, Math.round(W / 2 + 0.27 * U));
  const w = x1 - x0, h = H - y0;
  const { data } = canvas.getContext("2d").getImageData(x0, y0, w, h);
  const [tr, tg, tb] = cfg.lineColor;

  let bestY = null, bestFrac = 0;
  for (let y = 0; y < h; y++) {
    let hit = 0, tot = 0;
    for (let x = 0; x < w; x++) {
      if (Math.abs(x0 + x - W / 2) < 0.09 * U) continue;
      tot++;
      const i = (y * w + x) * 4;
      const diff = Math.abs(data[i] - tr) + Math.abs(data[i + 1] - tg) + Math.abs(data[i + 2] - tb);
      if (diff < cfg.lineTol) hit++;
    }
    const frac = hit / tot;
    if (frac > bestFrac) { bestFrac = frac; bestY = y0 + y; }
  }
  console.log(`line search: y=${bestY}, match fraction=${bestFrac.toFixed(2)}`);
  return bestFrac > 0.3 ? bestY : null;
}

function getPatch(canvas, cx, cy, D) {
  const c = cropToCanvas(canvas, cx - D / 2, cy - D / 2, D, D, SZ, SZ);
  const v = toVec(c);
  // luminance std-dev inside the mask, to spot empty slots
  const n = SZ * SZ;
  let sum = 0, sum2 = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    if (!v.mask[i]) continue;
    const l = 0.3 * v.rgb[i] + 0.59 * v.rgb[n + i] + 0.11 * v.rgb[2 * n + i];
    sum += l; sum2 += l * l; cnt++;
  }
  const mean = sum / cnt;
  v.std = Math.sqrt(Math.max(0, sum2 / cnt - mean * mean));
  return v;
}

// -------------------------------------------------------------- matching

// Masked normalized cross-correlation over the pixels valid in BOTH.
function ncc(a, b) {
  const n = SZ * SZ;
  let cnt = 0;
  const ma = [0, 0, 0], mb = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (!(a.mask[i] && b.mask[i])) continue;
    cnt++;
    for (let c = 0; c < 3; c++) { ma[c] += a.rgb[c * n + i]; mb[c] += b.rgb[c * n + i]; }
  }
  if (cnt < CIRCLE_COUNT * 0.15) return -1;
  for (let c = 0; c < 3; c++) { ma[c] /= cnt; mb[c] /= cnt; }

  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    if (!(a.mask[i] && b.mask[i])) continue;
    for (let c = 0; c < 3; c++) {
      const x = a.rgb[c * n + i] - ma[c];
      const y = b.rgb[c * n + i] - mb[c];
      dot += x * y; na += x * x; nb += y * y;
    }
  }
  if (na === 0 || nb === 0) return -1;
  const coverage = Math.min(1, Math.sqrt(cnt / (CIRCLE_COUNT * 0.4)));
  return (dot / Math.sqrt(na * nb)) * coverage;
}

function classify(patch, templates) {
  let best = { name: null, score: -1 }, second = { name: null, score: -1 };
  for (const t of templates) {
    let s = -1;
    for (const v of t.variants) s = Math.max(s, ncc(patch, v));
    if (s > best.score) { second = best; best = { name: t.name, score: s }; }
    else if (s > second.score) second = { name: t.name, score: s };
  }
  return { ...best, runnerUp: second.name, margin: best.score - second.score };
}

// ------------------------------------------------------------------ main

async function detectEnemies(imageEl, config = {}, enemyNames) {
  const cfg = { ...DEFAULTS, ...config };
  const canvas = document.createElement("canvas");
  canvas.width = imageEl.naturalWidth;
  canvas.height = imageEl.naturalHeight;
  canvas.getContext("2d").drawImage(imageEl, 0, 0);
  const W = canvas.width, H = canvas.height;

  console.time("detect");
  const templates = await getTemplates(enemyNames);
  console.log(`templates loaded: ${templates.length} of ${enemyNames.length}`);

  const U = Math.sqrt(W * H);
  const lineY = findLineY(canvas, cfg, U);
  const cy = lineY != null ? lineY + cfg.centerBelowLine * U : H * cfg.fallbackCenterY;
  const D = cfg.diameter * U;
  if (lineY == null) console.warn("faint line not found; using fallback y");

  const out = { left: [], right: [] };
  const slotLog = [];

  for (const side of ["left", "right"]) {
    for (let i = 0; i < 3; i++) {
      const cx = W / 2 + SLOT_X[side][i] * U;
      const patch = getPatch(canvas, cx, cy, D);
      const log = { side, slot: i, cx: Math.round(cx), cy: Math.round(cy), std: +patch.std.toFixed(1) };

      if (patch.std < cfg.emptyStd) {
        log.result = "empty";
      } else {
        const m = classify(patch, templates);
        Object.assign(log, { name: m.name, score: +m.score.toFixed(3), runnerUp: m.runnerUp, margin: +m.margin.toFixed(3) });
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
  console.timeEnd("detect");
  if (cfg.debug) console.table(slotLog);

  // Overlay so you can verify the geometry visually.
  if (cfg.debug) {
    const dbg = document.createElement("canvas");
    dbg.width = W; dbg.height = H;
    const g = dbg.getContext("2d");
    g.drawImage(canvas, 0, 0);
    g.lineWidth = Math.max(2, W / 600);
    g.font = `${Math.round(W / 60)}px sans-serif`;
    if (lineY != null) { g.strokeStyle = "yellow"; g.beginPath(); g.moveTo(W / 2 - 0.27 * U, lineY); g.lineTo(W / 2 + 0.27 * U, lineY); g.stroke(); }
    for (const s of slotLog) {
      g.strokeStyle = s.result === "ok" ? "lime" : s.result === "empty" ? "gray" : "red";
      g.beginPath(); g.arc(s.cx, s.cy, D / 2, 0, Math.PI * 2); g.stroke();
      g.fillStyle = g.strokeStyle;
      g.fillText(s.name ? `${s.name} ${s.score}` : s.result, s.cx - D / 2, s.cy - D / 2 - 6);
    }
    out.debugCanvas = dbg;
  }
  return out;
}