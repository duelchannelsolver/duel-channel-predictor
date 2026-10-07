// Enemy detector, v5: geometry-anchored (thin-line detection).
//
// The queue is always 3 slots per side (fixed positions, filled from the
// centre outwards), sitting a fixed distance below a faint horizontal line
// (~rgb(111,114,109)). All distances are fractions of image WIDTH. Verified
// on 1918x1078, 1486x807 and 1918x1198 screenshots. Ultrawide (20:9)
// screenshots are NOT supported: there the UI scales differently.
//   1. find the thin faint line (shape-based, not just colour) -> vertical anchor
//   2. 6 fixed slot centres -> no sliding window
//   3. skip empty slots (low pixel variance)
//   4. classify with alpha-masked, multi-scale NCC against the sprites
//   5. OCR the "xN" badge after binarizing the white text
//
// Tune via the `config` argument if needed.

const SZ = 24;                 // comparison size (px)
const MASK_R = SZ * 0.42;      // ignore the icon's border ring
// Scale is relative to the sprite's VISIBLE bounding box (transparent padding
// removed): 1 = longest side of the visible sprite fills the window, <1 =
// zoomed in. In-game icons are tight portraits, so we mostly want <= 1.
const SCALES = [0.55, 0.7, 0.85, 1.0, 1.25];
const SHIFTS = [[0, 0], [0, -0.15], [0, 0.15], [-0.15, 0], [0.15, 0]]; // [dx, dy]

const SLOT_X = {               // slot centre offset from image centre, / width
  left:  [-0.2007, -0.145, -0.0889],   // outer -> inner
  right: [0.0889, 0.145, 0.2007],      // inner -> outer
};

const DEFAULTS = {
  lineColor: [111, 114, 109],
  lineTol: 60,            // sum of |dR|+|dG|+|dB| allowed
  lineSpread: 20,         // max (maxChannel - minChannel): the line is neutral grey
  lineGap: 2,             // px of non-matching pixels tolerated inside a run
  lineMinCoverage: 0.5,   // a row is a line candidate if its longest run covers this much of the sampled width
  lineMinContrast: 0.35,  // peak coverage minus coverage of the rows just outside the line
  centerBelowLine: 0.036,  // slot centre y = lineY + this * width
  diameter: 0.054,         // icon diameter / width
  fallbackCenterY: 0.90,   // * height, only if the line isn't found
  emptyStd: 16,           // luminance std-dev below this => empty slot. Only a cheap pre-filter:
                          // empty slots measure ~10-12, but low-contrast icons (e.g. the grey wolf
                          // on its dark orange backdrop) measure ~23, so keep this well below that.
                          // minScore is what really rejects empties (they score ~0 vs real icons 0.85+).
  slashFrac: 0.85,        // empty slots show a thin light diagonal slash; >= this fraction of samples along it => empty
  slashMin: 6,            // luminance by which the slash must beat both flanks
  minScore: 0.50,         // best NCC below this => unknown, skipped (real icons score 0.85+)
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
// Crops a sprite to the bounding box of its non-transparent pixels.
function alphaCrop(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
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
  if (x1 < 0) return img; // fully transparent?? leave as is
  return cropToCanvas(c, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

function makeVariant(img, s, dx, dy) {
  const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
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

let templatesPromise = null;
function getTemplates(enemyNames) {
  if (!templatesPromise) {
    templatesPromise = (async () => {
      const loaded = await Promise.all(
        enemyNames.map(async (name) => {
          try {
            const img = alphaCrop(await loadImage(`sprites/${name}.png`));
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
// with a dark outline. Pipeline (each step fixed a real failure):
//   1. "whiteness" map (min of R,G,B) of the crop
//   2. drop the small leading "x" glyph (Tesseract merges it with a thin
//      "1" and reads "x1" as "4")
//   3. crop tightly to the digits and scale to a fixed glyph height
//      (a small glyph in a big blank canvas made tesseract.js return "")
//   4. threshold, then thin the fat outlined strokes by 1px (the game's
//      bold "6" was unreadable to tesseract.js until thinned)
// If OCR still returns nothing, retry with different settings before
// falling back to 1.
const OCR_ATTEMPTS = [
  { height: 64, thr: 170, erode: 1 },
  { height: 40, thr: 190, erode: 1 },
  { height: 48, thr: 150, erode: 2 },
  { height: 48, thr: 170, erode: 0 },
];

function erode3x3(ink, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let all = 1;
      for (let dy = -1; dy <= 1 && all; dy++)
        for (let dx = -1; dx <= 1; dx++)
          if (!ink[(y + dy) * w + x + dx]) { all = 0; break; }
      out[y * w + x] = all;
    }
  }
  return out;
}

// gray: canvas of the tight whiteness crop. Returns a black-on-white canvas.
function renderForOcr(gray, { height, thr, erode }) {
  const PAD = 30;
  const scale = height / gray.height;
  const tw = Math.max(1, Math.round(gray.width * scale));
  const th = height;
  const big = document.createElement("canvas");
  big.width = tw + PAD * 2;
  big.height = th + PAD * 2;
  const g = big.getContext("2d");
  g.fillStyle = "#000"; // padding becomes white after thresholding
  g.fillRect(0, 0, big.width, big.height);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(gray, PAD, PAD, tw, th);

  const id = g.getImageData(0, 0, big.width, big.height);
  let ink = new Uint8Array(big.width * big.height);
  for (let i = 0; i < ink.length; i++) ink[i] = id.data[i * 4] > thr ? 1 : 0;
  for (let k = 0; k < erode; k++) ink = erode3x3(ink, big.width, big.height);
  for (let i = 0; i < ink.length; i++) {
    const v = ink[i] ? 0 : 255; // white text -> black
    id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  g.putImageData(id, 0, 0);
  return big;
}

async function readCount(src, cx, cy, D, side) {
  const x0 = side === "left" ? cx + 0.05 * D : cx - 0.75 * D;
  const crop = cropToCanvas(src, x0, cy + 0.22 * D, 0.7 * D, 0.5 * D);
  const cw = crop.width, ch = crop.height;
  const px = crop.getContext("2d").getImageData(0, 0, cw, ch).data;

  const white = new Uint8ClampedArray(cw * ch);
  const colOn = new Uint8Array(cw);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = (y * cw + x) * 4;
      const v = Math.min(px[i], px[i + 1], px[i + 2]);
      white[y * cw + x] = v;
      if (v > 200) colOn[x] = 1;
    }
  }
  // column runs of text pixels; the first run is the "x" if it's narrow
  const runs = [];
  let start = -1;
  for (let x = 0; x <= cw; x++) {
    const on = x < cw && colOn[x];
    if (on && start < 0) start = x;
    if (!on && start >= 0) { runs.push([start, x]); start = -1; }
  }
  let skip = 0;
  if (runs.length >= 2 && runs[0][1] - runs[0][0] <= 0.15 * D) {
    skip = Math.max(0, runs[1][0] - 2);
  }

  // tight bounding box of the remaining text pixels
  let bx0 = cw, bx1 = -1, by0 = ch, by1 = -1;
  for (let y = 0; y < ch; y++) {
    for (let x = skip; x < cw; x++) {
      if (white[y * cw + x] > 200) {
        if (x < bx0) bx0 = x; if (x > bx1) bx1 = x;
        if (y < by0) by0 = y; if (y > by1) by1 = y;
      }
    }
  }
  if (bx1 < 0) return 1;
  bx0 = Math.max(skip, bx0 - 1); by0 = Math.max(0, by0 - 1);
  bx1 = Math.min(cw - 1, bx1 + 1); by1 = Math.min(ch - 1, by1 + 1);
  const gw = bx1 - bx0 + 1, gh = by1 - by0 + 1;

  const gray = document.createElement("canvas");
  gray.width = gw; gray.height = gh;
  const gctx = gray.getContext("2d");
  const gid = gctx.createImageData(gw, gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const v = white[(by0 + y) * cw + bx0 + x];
      const i = (y * gw + x) * 4;
      gid.data[i] = gid.data[i + 1] = gid.data[i + 2] = v;
      gid.data[i + 3] = 255;
    }
  }
  gctx.putImageData(gid, 0, 0);

  try {
    const worker = await getOcrWorker();
    for (const attempt of OCR_ATTEMPTS) {
      const { data: { text } } = await worker.recognize(renderForOcr(gray, attempt));
      const clean = text.replace(/[^0-9xX\u00d7]/g, "");
      const m = clean.match(/[xX\u00d7](\d+)/) || clean.match(/(\d+)/);
      if (m) return parseInt(m[1], 10);
    }
  } catch { /* fall through */ }
  return 1;
}

// -------------------------------------------------------------- geometry

// Finds the thin horizontal line. The line spans about +-0.215W from the
// centre; we sample 0.07W..0.21W on each side, skipping the middle where the
// ROUND badge / flame covers it.
//
// Matching the line's grey is NOT enough: sprite fur, stone tiles and snow
// are the same grey. What distinguishes the line is its shape, so a row only
// counts if:
//   1. its pixels are neutral grey close to lineColor (colour + low spread)
//   2. they form one long CONTINUOUS run (small gaps bridged) covering most
//      of the sampled width on both sides of the centre, not scattered hits
//   3. it is THIN: the group of adjacent qualifying rows is at most
//      ~W/320 px tall (a patch of grey floor is tens of px tall)
//   4. it has CONTRAST: the rows just above and below do not qualify
// Candidates are ranked by (peak coverage - neighbour coverage).
// Pure function over RGBA data so it can be tested outside the browser.
function findLineInData(data, W, H, cfg) {
  const y0 = Math.floor(H * 0.75);
  const x0 = Math.round(W * 0.29), x1 = Math.round(W * 0.71);
  const cx = W / 2, excl = W * 0.07;
  const h = H - y0;
  const [tr, tg, tb] = cfg.lineColor;

  // longest gap-tolerant run of matching pixels within [xa, xb) on row y
  const rowRun = (y, xa, xb) => {
    let best = 0, start = -1, last = -1;
    for (let x = xa; x < xb; x++) {
      const i = (y * W + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const diff = Math.abs(r - tr) + Math.abs(g - tg) + Math.abs(b - tb);
      const spread = Math.max(r, g, b) - Math.min(r, g, b);
      if (diff < cfg.lineTol && spread <= cfg.lineSpread) {
        if (start < 0 || x - last > cfg.lineGap + 1) start = x;
        last = x;
        if (last - start + 1 > best) best = last - start + 1;
      }
    }
    return best;
  };

  const lA = x0, lB = Math.floor(cx - excl), rA = Math.ceil(cx + excl), rB = x1;
  const tot = (lB - lA) + (rB - rA);
  const cov = new Float32Array(h);
  for (let j = 0; j < h; j++) {
    const y = y0 + j;
    cov[j] = (rowRun(y, lA, lB) + rowRun(y, rA, rB)) / tot;
  }

  const maxThick = Math.max(3, Math.round(W / 320));
  let best = null;
  for (let j = 0; j < h; ) {
    if (cov[j] < cfg.lineMinCoverage) { j++; continue; }
    let k = j;
    while (k < h && cov[k] >= cfg.lineMinCoverage) k++;
    const thick = k - j;                       // rows j..k-1
    if (thick <= maxThick) {
      let peak = 0, peakJ = j;
      for (let q = j; q < k; q++) if (cov[q] > peak) { peak = cov[q]; peakJ = q; }
      let around = 0;
      for (let d = 2; d <= maxThick + 1; d++) { // d=1 is the line's own anti-aliased edge
        if (j - d >= 0) around = Math.max(around, cov[j - d]);
        if (k - 1 + d < h) around = Math.max(around, cov[k - 1 + d]);
      }
      const score = peak - around;
      if (score >= cfg.lineMinContrast && (!best || score > best.score))
        best = { y: y0 + peakJ, score, peak, thick };
    }
    j = k;
  }
  return best;
}

function findLineY(canvas, cfg) {
  const W = canvas.width, H = canvas.height;
  const { data } = canvas.getContext("2d").getImageData(0, 0, W, H);
  const hit = findLineInData(data, W, H, cfg);
  console.log(hit
    ? `line search: y=${hit.y}, coverage=${hit.peak.toFixed(2)}, thickness=${hit.thick}px, contrast=${hit.score.toFixed(2)}`
    : "line search: no thin line found");
  return hit ? hit.y : null;
}

// Empty slots are translucent discs with a thin light "\\" slash through the
// middle. Their std-dev overlaps with low-contrast real icons, and a pale
// empty disc can correlate strongly with a pale sprite, so detect the slash
// itself: sample along the main diagonal (searching a small perpendicular
// offset) and check the line is brighter than BOTH flanks at ~every point.
// Measured: empty slots 1.00 on every sample, real icons <= 0.62.
function slashFraction(canvas, cx, cy, D, cfg) {
  const n = Math.round(D);
  const c = cropToCanvas(canvas, cx - D / 2, cy - D / 2, n, n);
  const { data } = c.getContext("2d").getImageData(0, 0, n, n);
  const L = (x, y) => {
    x = Math.min(n - 1, Math.max(0, Math.round(x)));
    y = Math.min(n - 1, Math.max(0, Math.round(y)));
    const i = (y * n + x) * 4;
    return 0.3 * data[i] + 0.59 * data[i + 1] + 0.11 * data[i + 2];
  };
  const off = 0.07 * n / Math.SQRT2;
  let best = 0;
  for (let o = -0.08 * n; o < 0.08 * n; o += 0.5) {
    let ok = 0, tot = 0;
    for (let k = 0; k <= 20; k++) {
      const t = (0.3 + 0.02 * k) * n;
      const mid = L(t, t + o);
      const a = L(t + off, t + o - off), b = L(t - off, t + o + off);
      if (mid - Math.max(a, b) > cfg.slashMin) ok++;
      tot++;
    }
    best = Math.max(best, ok / tot);
  }
  return best;
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

  const lineY = findLineY(canvas, cfg);
  const cy = lineY != null ? lineY + cfg.centerBelowLine * W : H * cfg.fallbackCenterY;
  const D = cfg.diameter * W;
  if (lineY == null) console.warn("faint line not found; using fallback y");

  const out = { left: [], right: [] };
  const slotLog = [];

  for (const side of ["left", "right"]) {
    for (let i = 0; i < 3; i++) {
      const cx = W / 2 + SLOT_X[side][i] * W;
      const patch = getPatch(canvas, cx, cy, D);
      const log = { side, slot: i, cx: Math.round(cx), cy: Math.round(cy), std: +patch.std.toFixed(1) };

      const slash = slashFraction(canvas, cx, cy, D, cfg);
      log.slash = +slash.toFixed(2);
      if (patch.std < cfg.emptyStd || slash >= cfg.slashFrac) {
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
    if (lineY != null) { g.strokeStyle = "yellow"; g.beginPath(); g.moveTo(W * 0.29, lineY); g.lineTo(W * 0.71, lineY); g.stroke(); }
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