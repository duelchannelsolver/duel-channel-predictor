#!/usr/bin/env node
'use strict';
/**
 * Duel Channel Stacked Ensemble Classifier (dependency-free, plain Node.js).
 *
 * Base Models:
 *   1. Domain Neural Network (Arknights damage & TTK feature extraction + MLP)
 *   2. Logistic Regression (Sqrt-count differential + L2 Adam)
 *   3. Random Forest (CART trees with weighted Gini splits + swap augmentation)
 *
 * Meta-Learner:
 *   - Trains on out-of-fold log-odds (logits) using k-fold cross-validation.
 *   - Anti-symmetric linear logit combination: P(A beats B) = 1 - P(B beats A).
 *
 * Usage:
 *   node stacked_duel_model.js
 *   node stacked_duel_model.js --folds 5 --seed 42 --epochs 120 --trees 150
 *   node stacked_duel_model.js predict '{"teamA":{"Snotty Slug":10},"teamB":{"Hound Pro":3},"storm":false}'
 */

const fs = require('fs');
const path = require('path');

// ----------------------------------------------------------------------------------------------
// CLI & Configuration
// ----------------------------------------------------------------------------------------------
const argv = process.argv.slice(2);
const predictIdx = argv.indexOf('predict');
const isPredict = predictIdx !== -1;

const flags = {};
for (let i = 0; i < argv.length; i++) {
  if (i === predictIdx || i === predictIdx + 1) continue;
  if (argv[i].startsWith('--')) {
    flags[argv[i].slice(2)] = argv[i + 1];
    i++;
  }
}

const CFG = {
  enemies: flags.enemies || path.join(__dirname, 'data/duel_channel_enemies.json'),
  matches: flags.matches || path.join(__dirname, 'data/matches.json'),
  out: flags.out || path.join(__dirname, 'stacked_model.json'),
  folds: Number(flags.folds || 5),
  seed: Number(flags.seed || 42),
  // NN configs
  nnHidden: (flags.hidden || '24,12').split(',').map(Number),
  nnEpochs: Number(flags.epochs || 120),
  nnLr: Number(flags.lr || 0.003),
  nnWd: Number(flags.wd || 0.003),
  nnBatch: Number(flags.batch || 32),
  // Logistic Regression configs
  lrL2: Number(flags.lrL2 || 1.0),
  lrStormWeight: Number(flags.lrStorm || 0.75),
  lrIters: Number(flags.lrIters || 800),
  // Random Forest configs
  rfTrees: Number(flags.trees || 150),
  rfDepth: Number(flags.depth || 12),
  rfMinLeaf: Number(flags.minLeaf || 3),
  rfStormWeight: Number(flags.rfStorm || 0.75),
};

// ----------------------------------------------------------------------------------------------
// Math & RNG Utilities
// ----------------------------------------------------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const logit = (p) => {
  const c = clamp(p, 1e-6, 1 - 1e-6);
  return Math.log(c / (1 - c));
};

function norm(s) {
  return String(s).toLowerCase().replace(/["'\u2019]/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
}
const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

// ----------------------------------------------------------------------------------------------
// Base Model 1: Neural Network (Arknights Stats + Domain Features)
// ----------------------------------------------------------------------------------------------
const ABILITY_FLAGS = {
  stun:      (t) => /\bstuns?\b/.test(t.replace(/immune to stun( and freeze)?/g, '')),
  revive:    (t) => /revive/.test(t),
  barrier:   (t) => /barrier|shield/.test(t),
  aoe:       (t) => /splash|adjacent|radius|up to (two|three|four|\d+) targets|jump between|entire column|every row/.test(t),
  dot:       (t) => /burn|necrosis|corruption|damage every|per second|dot\b/.test(t) && !/regenerates/.test(t),
  heal:      (t) => /regenerates|restores/.test(t),
  instakill: (t) => /instantly defeat|devour|abduct/.test(t),
  defShred:  (t) => /inflict[^.]*-\d+ def/.test(t),
  tanky:     (t) => /takes -\d+%|dodge|immune|status resistance/.test(t),
  selfBuff:  (t) => /gains|enraged|overdrive|\+\d+% atk/.test(t),
  burst:     (t) => /explodes|burst/.test(t),
};
const FLAG_NAMES = Object.keys(ABILITY_FLAGS);

function makeUnit(e) {
  const dmgType = String(e.Damage || '').toLowerCase();
  const physShare = dmgType.startsWith('phys') ? 1 : dmgType.startsWith('arts') ? 0 : dmgType === 'see below' ? 0.5 : 1;
  const text = String(e['Special Ability'] || '').toLowerCase();
  const range = e['Attack Range'] === 'Global' ? 10 : num(e['Attack Range'], 0.8);
  const atk = num(e.ATK, 0);
  return {
    name: e.Name,
    hp: num(e.HP, 0),
    atk,
    def: num(e.DEF, 0),
    res: num(e.RES, 0),
    ms: num(e['Movement Speed'], 0),
    ai: num(e['Attack Interval'], atk > 0 ? 5 : 1),
    range,
    physShare,
    flags: FLAG_NAMES.map((f) => (ABILITY_FLAGS[f](text) ? 1 : 0)),
  };
}

function loadEnemies(file) {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  const lookup = new Map();
  const add = (key, unit) => {
    if (!lookup.has(key)) lookup.set(key, []);
    const list = lookup.get(key);
    if (!list.some((u) => u.name === unit.name)) list.push(unit);
  };
  for (const e of raw) {
    const unit = makeUnit(e);
    lookup.set(norm(e.Name), [unit]);
    if (e.Name.includes(',')) lookup.set(norm(e.Name.split(',')[0]), [unit]);
    if (e.Group) add(norm(e.Group), unit);
  }
  return lookup;
}

function expandTeam(team, lookup) {
  const units = [];
  for (const [name, count] of Object.entries(team)) {
    const found = lookup.get(norm(name));
    if (!found) throw new Error(`Unknown enemy "${name}" in matches.`);
    for (const u of found) units.push({ u, n: count });
  }
  return units;
}

function teamStats(units) {
  let hp = 0, count = 0, phys = 0, arts = 0, defW = 0, resW = 0, ms = 0, rng = 0, ranged = 0, hpMax = 0;
  const flagSum = new Array(FLAG_NAMES.length).fill(0);
  for (const { u, n } of units) {
    const dps = u.atk / u.ai;
    hp += n * u.hp;
    count += n;
    phys += n * dps * u.physShare;
    arts += n * dps * (1 - u.physShare);
    defW += n * u.hp * u.def;
    resW += n * u.hp * u.res;
    ms += n * u.ms;
    rng += n * u.range;
    if (u.range >= 2.5) ranged += n;
    hpMax = Math.max(hpMax, u.hp);
    u.flags.forEach((f, i) => { flagSum[i] += f * n; });
  }
  return {
    hp, count, phys, arts, hpMax,
    avgDef: hp > 0 ? defW / hp : 0,
    avgRes: hp > 0 ? resW / hp : 0,
    avgMs: count > 0 ? ms / count : 0,
    avgRange: count > 0 ? rng / count : 0,
    rangedFrac: count > 0 ? ranged / count : 0,
    flagSum,
  };
}

function effectiveDps(units, tgtDef, tgtRes) {
  let total = 0;
  for (const { u, n } of units) {
    const phys = Math.max(u.atk - tgtDef, 0.05 * u.atk);
    const arts = Math.max(u.atk * (1 - tgtRes / 100), 0.05 * u.atk);
    total += (n * (u.physShare * phys + (1 - u.physShare) * arts)) / u.ai;
  }
  return total;
}

const TEAM_FEATURES = [
  'logHP', 'logPhysDPS', 'logArtsDPS', 'logCount', 'logAvgDef', 'avgRes', 'avgMS', 'avgRange',
  'rangedFrac', 'logStrength', 'logMaxUnitHP', ...FLAG_NAMES.map((f) => `flag_${f}`),
];
function teamFeatures(s) {
  return [
    Math.log1p(s.hp), Math.log1p(s.phys), Math.log1p(s.arts), Math.log1p(s.count),
    Math.log1p(s.avgDef), s.avgRes / 100, s.avgMs, s.avgRange, s.rangedFrac,
    Math.log1p(s.hp * (s.phys + s.arts)), Math.log1p(s.hpMax),
    ...s.flagSum.map((v) => Math.log1p(v)),
  ];
}

function buildNnFeatures(unitsA, unitsB, storm) {
  const sA = teamStats(unitsA), sB = teamStats(unitsB);
  const dpsAB = Math.max(effectiveDps(unitsA, sB.avgDef, sB.avgRes), 1e-3);
  const dpsBA = Math.max(effectiveDps(unitsB, sA.avgDef, sA.avgRes), 1e-3);
  const logTtkA = Math.log(Math.max(sB.hp, 1) / dpsAB);
  const logTtkB = Math.log(Math.max(sA.hp, 1) / dpsBA);
  return [
    ...teamFeatures(sA), ...teamFeatures(sB),
    clamp(logTtkA, -10, 15), clamp(logTtkB, -10, 15), clamp(logTtkB - logTtkA, -10, 10),
    storm ? 1 : 0,
  ];
}

class MLP {
  constructor(sizes, rng) {
    this.sizes = sizes;
    this.L = sizes.length - 1;
    this.W = []; this.b = [];
    for (let l = 0; l < this.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const w = new Float64Array(nin * nout);
      const scale = Math.sqrt(2 / nin);
      for (let i = 0; i < w.length; i++) w[i] = gauss(rng) * scale;
      this.W.push(w);
      this.b.push(new Float64Array(nout));
    }
  }

  forward(x, cache) {
    let a = x;
    if (cache) cache.a = [x];
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l];
      const z = new Float64Array(nout);
      for (let o = 0; o < nout; o++) {
        let s = b[o];
        const off = o * nin;
        for (let i = 0; i < nin; i++) s += W[off + i] * a[i];
        z[o] = s;
      }
      if (l < this.L - 1) for (let o = 0; o < nout; o++) z[o] = z[o] > 0 ? z[o] : 0;
      else z[0] = sigmoid(z[0]);
      a = z;
      if (cache) cache.a.push(a);
    }
    return a[0];
  }

  predict(x) { return this.forward(x, null); }

  backward(x, y, gW, gb) {
    const cache = {};
    const p = this.forward(x, cache);
    let dz = new Float64Array([p - y]);
    for (let l = this.L - 1; l >= 0; l--) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const aPrev = cache.a[l];
      const W = this.W[l];
      for (let o = 0; o < nout; o++) {
        const g = dz[o];
        gb[l][o] += g;
        const off = o * nin;
        for (let i = 0; i < nin; i++) gW[l][off + i] += g * aPrev[i];
      }
      if (l > 0) {
        const da = new Float64Array(nin);
        for (let o = 0; o < nout; o++) {
          const off = o * nin;
          for (let i = 0; i < nin; i++) da[i] += W[off + i] * dz[o];
        }
        for (let i = 0; i < nin; i++) da[i] = aPrev[i] > 0 ? da[i] : 0;
        dz = da;
      }
    }
    return p;
  }

  toJSON() {
    return { sizes: this.sizes, W: this.W.map((w) => Array.from(w)), b: this.b.map((b) => Array.from(b)) };
  }
  static fromJSON(j) {
    const m = new MLP(j.sizes, mulberry32(1));
    m.W = j.W.map((w) => Float64Array.from(w));
    m.b = j.b.map((b) => Float64Array.from(b));
    return m;
  }
}

function fitNormalizer(X) {
  const d = X[0].length, mean = new Array(d).fill(0), std = new Array(d).fill(0);
  for (const x of X) for (let i = 0; i < d; i++) mean[i] += x[i] / X.length;
  for (const x of X) for (let i = 0; i < d; i++) std[i] += (x[i] - mean[i]) ** 2 / X.length;
  return { mean, std: std.map((s) => Math.sqrt(s) || 1) };
}
const applyNorm = (x, nz) => x.map((v, i) => (v - nz.mean[i]) / nz.std[i]);

function trainNn(matches, lookup, cfg, rng) {
  const samples = matches.map((m) => {
    const uA = expandTeam(m.teamA, lookup), uB = expandTeam(m.teamB, lookup);
    return {
      fwd: buildNnFeatures(uA, uB, !!m.storm),
      swp: buildNnFeatures(uB, uA, !!m.storm),
      y: m.winner === 'A' ? 1 : 0,
    };
  });
  const data = [];
  for (const s of samples) {
    data.push({ x: s.fwd, y: s.y });
    data.push({ x: s.swp, y: 1 - s.y });
  }
  const nz = fitNormalizer(data.map((d) => d.x));
  const X = data.map((d) => applyNorm(d.x, nz));
  const Y = data.map((d) => d.y);

  const sizes = [X[0].length, ...cfg.nnHidden, 1];
  const net = new MLP(sizes, rng);
  const mW = net.W.map((w) => new Float64Array(w.length)), vW = net.W.map((w) => new Float64Array(w.length));
  const mB = net.b.map((b) => new Float64Array(b.length)), vB = net.b.map((b) => new Float64Array(b.length));
  const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
  let step = 0;
  const idx = X.map((_, i) => i);

  for (let epoch = 1; epoch <= cfg.nnEpochs; epoch++) {
    shuffle(idx, rng);
    for (let start = 0; start < idx.length; start += cfg.nnBatch) {
      const batch = idx.slice(start, start + cfg.nnBatch);
      const gW = net.W.map((w) => new Float64Array(w.length));
      const gb = net.b.map((b) => new Float64Array(b.length));
      for (const i of batch) net.backward(X[i], Y[i], gW, gb);

      step++;
      const c1 = 1 - Math.pow(B1, step), c2 = 1 - Math.pow(B2, step);
      for (let l = 0; l < net.L; l++) {
        for (let k = 0; k < net.W[l].length; k++) {
          const g = gW[l][k] / batch.length + cfg.nnWd * net.W[l][k];
          mW[l][k] = B1 * mW[l][k] + (1 - B1) * g;
          vW[l][k] = B2 * vW[l][k] + (1 - B2) * g * g;
          net.W[l][k] -= (cfg.nnLr * (mW[l][k] / c1)) / (Math.sqrt(vW[l][k] / c2) + EPS);
        }
        for (let k = 0; k < net.b[l].length; k++) {
          const g = gb[l][k] / batch.length;
          mB[l][k] = B1 * mB[l][k] + (1 - B1) * g;
          vB[l][k] = B2 * vB[l][k] + (1 - B2) * g * g;
          net.b[l][k] -= (cfg.nnLr * (mB[l][k] / c1)) / (Math.sqrt(vB[l][k] / c2) + EPS);
        }
      }
    }
  }
  return { net, nz };
}

function predictNn(model, match, lookup) {
  const uA = expandTeam(match.teamA, lookup);
  const uB = expandTeam(match.teamB, lookup);
  const fwd = buildNnFeatures(uA, uB, !!match.storm);
  const swp = buildNnFeatures(uB, uA, !!match.storm);
  const pf = model.net.predict(applyNorm(fwd, model.nz));
  const ps = model.net.predict(applyNorm(swp, model.nz));
  return 0.5 * (pf + (1 - ps));
}

// ----------------------------------------------------------------------------------------------
// Base Model 2: Logistic Regression (Enemy Strength Model)
// ----------------------------------------------------------------------------------------------
class LogisticRegressionModel {
  constructor(enemies, l2 = 1.0, stormWeight = 0.75) {
    this.l2 = l2;
    this.stormWeight = stormWeight;
    this.enemies = enemies;
    this.weights = new Float64Array(enemies.length);
  }

  fit(matches, iters = 800) {
    const n = matches.length, d = this.enemies.length;
    const X = new Float64Array(n * d), y = new Float64Array(n), sw = new Float64Array(n);

    for (let s = 0; s < n; s++) {
      const m = matches[s];
      y[s] = m.winner === 'A' ? 1 : 0;
      sw[s] = m.storm === true ? this.stormWeight : 1.0;
      for (let j = 0; j < d; j++) {
        const e = this.enemies[j];
        const countA = parseInt((m.teamA && m.teamA[e]) || 0, 10);
        const countB = parseInt((m.teamB && m.teamB[e]) || 0, 10);
        X[s * d + j] = Math.sqrt(countA) - Math.sqrt(countB);
      }
    }

    const weights = new Float64Array(d);
    const mAdam = new Float64Array(d), vAdam = new Float64Array(d), grad = new Float64Array(d);
    const B1 = 0.9, B2 = 0.999, EPS = 1e-8, lr = 0.1;

    for (let it = 1; it <= iters; it++) {
      grad.fill(0);
      for (let s = 0; s < n; s++) {
        let z = 0;
        const off = s * d;
        for (let j = 0; j < d; j++) z += weights[j] * X[off + j];
        const p = z > 20 ? 1 : z < -20 ? 0 : 1 / (1 + Math.exp(-z));
        const err = p - y[s];
        const weight = sw[s];
        for (let j = 0; j < d; j++) grad[j] += weight * err * X[off + j];
      }

      for (let j = 0; j < d; j++) {
        grad[j] += this.l2 * weights[j];
        mAdam[j] = B1 * mAdam[j] + (1 - B1) * grad[j];
        vAdam[j] = B2 * vAdam[j] + (1 - B2) * (grad[j] * grad[j]);
        const mHat = mAdam[j] / (1 - Math.pow(B1, it));
        const vHat = vAdam[j] / (1 - Math.pow(B2, it));
        weights[j] -= (lr * mHat) / (Math.sqrt(vHat) + EPS);
      }
    }
    this.weights = weights;
  }

  predictProba(teamA, teamB) {
    let z = 0;
    for (let i = 0; i < this.enemies.length; i++) {
      const e = this.enemies[i];
      const countA = parseInt((teamA && teamA[e]) || 0, 10);
      const countB = parseInt((teamB && teamB[e]) || 0, 10);
      z += this.weights[i] * (Math.sqrt(countA) - Math.sqrt(countB));
    }
    return sigmoid(z);
  }

  toJSON() {
    return { enemies: this.enemies, weights: Array.from(this.weights) };
  }
  static fromJSON(j) {
    const m = new LogisticRegressionModel(j.enemies);
    m.weights = Float64Array.from(j.weights);
    return m;
  }
}

// ----------------------------------------------------------------------------------------------
// Base Model 3: Random Forest (Bagged CART Trees with Weighted Gini)
// ----------------------------------------------------------------------------------------------
class RandomForestModel {
  constructor(enemies, options = {}) {
    this.enemies = enemies;
    this.nTrees = options.nTrees ?? 150;
    this.maxDepth = options.maxDepth ?? 12;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 3;
    this.maxFeatures = options.maxFeatures ?? null;
    this.stormWeight = options.stormWeight ?? 0.75;
    this.seed = options.seed ?? 42;
    this.trees = [];
  }

  fit(matches) {
    const n = matches.length, d = this.enemies.length;
    const X = new Float64Array(2 * n * d), y = new Uint8Array(2 * n), sw = new Float64Array(2 * n);

    for (let s = 0; s < n; s++) {
      const m = matches[s];
      const label = m.winner === 'A' ? 1 : 0;
      const w = m.storm === true ? this.stormWeight : 1.0;
      y[s] = label; y[s + n] = 1 - label;
      sw[s] = sw[s + n] = w;

      for (let j = 0; j < d; j++) {
        const e = this.enemies[j];
        const countA = parseInt((m.teamA && m.teamA[e]) || 0, 10);
        const countB = parseInt((m.teamB && m.teamB[e]) || 0, 10);
        const f = countA - countB;
        X[s * d + j] = f;
        X[(s + n) * d + j] = -f;
      }
    }

    const rng = mulberry32(this.seed);
    const mtry = this.maxFeatures ?? Math.max(1, Math.floor(Math.sqrt(d)));
    this.trees = [];

    for (let t = 0; t < this.nTrees; t++) {
      const rows = [];
      for (let i = 0; i < n; i++) {
        const k = Math.floor(rng() * n);
        rows.push(k, k + n);
      }
      this.trees.push(this._buildTree(X, y, sw, d, rows, 0, mtry, rng));
    }
  }

  _buildTree(X, y, sw, d, rows, depth, mtry, rng) {
    let W = 0, W1 = 0;
    for (const r of rows) {
      W += sw[r];
      if (y[r] === 1) W1 += sw[r];
    }
    const leaf = { p: W > 0 ? W1 / W : 0.5 };
    if (depth >= this.maxDepth || rows.length < 2 * this.minSamplesLeaf || W1 === 0 || W1 === W) {
      return leaf;
    }

    const parentImp = 2 * (W1 / W) * (1 - W1 / W);
    const feats = new Int32Array(d);
    for (let j = 0; j < d; j++) feats[j] = j;
    for (let j = 0; j < Math.min(mtry, d); j++) {
      const k = j + Math.floor(rng() * (d - j));
      const tmp = feats[j]; feats[j] = feats[k]; feats[k] = tmp;
    }

    let bestGain = 1e-12, bestF = -1, bestT = 0;
    for (let fi = 0; fi < Math.min(mtry, d); fi++) {
      const f = feats[fi];
      const sorted = rows.slice().sort((a, b) => X[a * d + f] - X[b * d + f]);
      let WL = 0, WL1 = 0;
      for (let i = 0; i < sorted.length - 1; i++) {
        const r = sorted[i];
        WL += sw[r];
        if (y[r] === 1) WL1 += sw[r];
        const v = X[r * d + f];
        const vNext = X[sorted[i + 1] * d + f];
        if (v === vNext) continue;

        const nL = i + 1, nR = sorted.length - nL;
        if (nL < this.minSamplesLeaf || nR < this.minSamplesLeaf) continue;

        const WR = W - WL, WR1 = W1 - WL1;
        const pL = WL1 / WL, pR = WR1 / WR;
        const impL = 2 * pL * (1 - pL), impR = 2 * pR * (1 - pR);
        const gain = W * parentImp - WL * impL - WR * impR;

        if (gain > bestGain) {
          bestGain = gain; bestF = f; bestT = (v + vNext) / 2;
        }
      }
    }

    if (bestF < 0) return leaf;
    const left = [], right = [];
    for (const r of rows) {
      (X[r * d + bestF] <= bestT ? left : right).push(r);
    }
    return {
      f: bestF,
      t: bestT,
      l: this._buildTree(X, y, sw, d, left, depth + 1, mtry, rng),
      r: this._buildTree(X, y, sw, d, right, depth + 1, mtry, rng),
    };
  }

  _predictTreeVec(node, x) {
    while (node.p === undefined) node = x[node.f] <= node.t ? node.l : node.r;
    return node.p;
  }

  _forestProba(x) {
    let sum = 0;
    for (const tree of this.trees) sum += this._predictTreeVec(tree, x);
    return sum / this.trees.length;
  }

  predictProba(teamA, teamB) {
    const d = this.enemies.length;
    const x = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      const e = this.enemies[i];
      const countA = parseInt((teamA && teamA[e]) || 0, 10);
      const countB = parseInt((teamB && teamB[e]) || 0, 10);
      x[i] = countA - countB;
    }
    const neg = x.map((v) => -v);
    return 0.5 * (this._forestProba(x) + (1 - this._forestProba(neg)));
  }

  toJSON() {
    return { enemies: this.enemies, trees: this.trees };
  }
  static fromJSON(j) {
    const m = new RandomForestModel(j.enemies);
    m.trees = j.trees;
    return m;
  }
}

// ----------------------------------------------------------------------------------------------
// Stacking Meta-Learner (Logit Stacker)
// ----------------------------------------------------------------------------------------------
class StackingMetaModel {
  constructor(weights = [0.333, 0.333, 0.333]) {
    this.weights = Float64Array.from(weights);
  }

  /**
   * Fits meta-weights w on out-of-fold logit predictions:
   * z = w_nn * logit(p_nn) + w_lr * logit(p_lr) + w_rf * logit(p_rf)
   * Guaranteed zero-bias slot symmetry: logit(1 - p) = -logit(p).
   */
  fit(oofLogits, oofTargets, iters = 600, lr = 0.05, wd = 0.001) {
    const n = oofLogits.length, k = this.weights.length;
    const mW = new Float64Array(k), vW = new Float64Array(k);
    const B1 = 0.9, B2 = 0.999, EPS = 1e-8;

    for (let it = 1; it <= iters; it++) {
      const grad = new Float64Array(k);
      for (let s = 0; s < n; s++) {
        let z = 0;
        for (let j = 0; j < k; j++) z += this.weights[j] * oofLogits[s][j];
        const p = sigmoid(z);
        const err = p - oofTargets[s];
        for (let j = 0; j < k; j++) grad[j] += (err * oofLogits[s][j]) / n;
      }

      for (let j = 0; j < k; j++) {
        const g = grad[j] + wd * this.weights[j];
        mW[j] = B1 * mW[j] + (1 - B1) * g;
        vW[j] = B2 * vW[j] + (1 - B2) * g * g;
        const mHat = mW[j] / (1 - Math.pow(B1, it));
        const vHat = vW[j] / (1 - Math.pow(B2, it));
        this.weights[j] -= (lr * mHat) / (Math.sqrt(vHat) + EPS);
      }
    }
  }

  predictProba(pNN, pLR, pRF) {
    const logits = [logit(pNN), logit(pLR), logit(pRF)];
    let z = 0;
    for (let j = 0; j < this.weights.length; j++) z += this.weights[j] * logits[j];
    return sigmoid(z);
  }

  toJSON() {
    return Array.from(this.weights);
  }
  static fromJSON(j) {
    return new StackingMetaModel(j);
  }
}

// ----------------------------------------------------------------------------------------------
// Orchestrator: Training, Cross-Validation & Inference
// ----------------------------------------------------------------------------------------------
function computeMetrics(preds, targets) {
  let correct = 0, ll = 0;
  for (let i = 0; i < preds.length; i++) {
    const p = clamp(preds[i], 1e-6, 1 - 1e-6);
    const y = targets[i];
    if ((p >= 0.5) === (y === 1)) correct++;
    ll += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }
  return { acc: correct / preds.length, logLoss: ll / preds.length };
}

function runTraining() {
  const lookup = loadEnemies(CFG.enemies);
  const matches = JSON.parse(fs.readFileSync(CFG.matches, 'utf8'));

  // Build unified enemy vocabulary for LR and RF
  const enemySet = new Set();
  for (const m of matches) {
    if (m.teamA) Object.keys(m.teamA).forEach((e) => enemySet.add(e));
    if (m.teamB) Object.keys(m.teamB).forEach((e) => enemySet.add(e));
  }
  const enemies = Array.from(enemySet).sort();

  console.log(`Loaded ${lookup.size} enemy stats, ${enemies.length} distinct match unit types.`);
  console.log(`Evaluating 3 base models & meta-learner across ${CFG.folds}-fold Cross-Validation...`);

  const rng = mulberry32(CFG.seed);
  const order = shuffle(matches.map((_, i) => i), rng);
  const oofPredsNN = new Float64Array(matches.length);
  const oofPredsLR = new Float64Array(matches.length);
  const oofPredsRF = new Float64Array(matches.length);
  const oofTargets = new Float64Array(matches.length);

  for (let f = 0; f < CFG.folds; f++) {
    const testIndices = new Set(order.filter((_, idx) => idx % CFG.folds === f));
    const trMatches = matches.filter((_, idx) => !testIndices.has(idx));
    const teMatches = matches.filter((_, idx) => testIndices.has(idx));

    // Train Fold Base Models
    const foldNN = trainNn(trMatches, lookup, CFG, mulberry32(CFG.seed + f * 17 + 1));
    const foldLR = new LogisticRegressionModel(enemies, CFG.lrL2, CFG.lrStormWeight);
    foldLR.fit(trMatches, CFG.lrIters);
    const foldRF = new RandomForestModel(enemies, {
      nTrees: CFG.rfTrees,
      maxDepth: CFG.rfDepth,
      minSamplesLeaf: CFG.rfMinLeaf,
      stormWeight: CFG.rfStormWeight,
      seed: CFG.seed + f * 23 + 1,
    });
    foldRF.fit(trMatches);

    // Predict on Holdout Fold
    for (let i = 0; i < matches.length; i++) {
      if (!testIndices.has(i)) continue;
      const m = matches[i];
      oofPredsNN[i] = predictNn(foldNN, m, lookup);
      oofPredsLR[i] = foldLR.predictProba(m.teamA, m.teamB);
      oofPredsRF[i] = foldRF.predictProba(m.teamA, m.teamB);
      oofTargets[i] = m.winner === 'A' ? 1 : 0;
    }
    process.stdout.write(`  [Fold ${f + 1}/${CFG.folds} complete]\r`);
  }
  console.log('\nCross-validation complete. Model performance (Out-Of-Fold):');

  const mNN = computeMetrics(oofPredsNN, oofTargets);
  const mLR = computeMetrics(oofPredsLR, oofTargets);
  const mRF = computeMetrics(oofPredsRF, oofTargets);

  console.log(`  1. Neural Network     : Accuracy ${(mNN.acc * 100).toFixed(1)}% | Log-Loss ${mNN.logLoss.toFixed(4)}`);
  console.log(`  2. Logistic Regression: Accuracy ${(mLR.acc * 100).toFixed(1)}% | Log-Loss ${mLR.logLoss.toFixed(4)}`);
  console.log(`  3. Random Forest      : Accuracy ${(mRF.acc * 100).toFixed(1)}% | Log-Loss ${mRF.logLoss.toFixed(4)}`);

  // Fit Meta-Learner on OOF Logits
  const oofLogits = [];
  for (let i = 0; i < matches.length; i++) {
    oofLogits.push([logit(oofPredsNN[i]), logit(oofPredsLR[i]), logit(oofPredsRF[i])]);
  }
  const metaModel = new StackingMetaModel();
  metaModel.fit(oofLogits, oofTargets);

  const oofStacked = oofLogits.map((l) =>
    sigmoid(metaModel.weights[0] * l[0] + metaModel.weights[1] * l[1] + metaModel.weights[2] * l[2])
  );
  const mStack = computeMetrics(oofStacked, oofTargets);

  console.log('-------------------------------------------------------------');
  console.log(`* STACKED ENSEMBLE      : Accuracy ${(mStack.acc * 100).toFixed(1)}% | Log-Loss ${mStack.logLoss.toFixed(4)}`);
  console.log(
    `  Learned Meta Weights  : NN=${metaModel.weights[0].toFixed(3)}, LR=${metaModel.weights[1].toFixed(3)}, RF=${metaModel.weights[2].toFixed(3)}`
  );
  console.log('-------------------------------------------------------------');

  // Refit All Base Models on Full Dataset
  console.log('Fitting all base models on the complete match dataset...');
  const finalNN = trainNn(matches, lookup, CFG, mulberry32(CFG.seed + 999));
  const finalLR = new LogisticRegressionModel(enemies, CFG.lrL2, CFG.lrStormWeight);
  finalLR.fit(matches, CFG.lrIters);
  const finalRF = new RandomForestModel(enemies, {
    nTrees: CFG.rfTrees,
    maxDepth: CFG.rfDepth,
    minSamplesLeaf: CFG.rfMinLeaf,
    stormWeight: CFG.rfStormWeight,
    seed: CFG.seed + 999,
  });
  finalRF.fit(matches);

  const bundle = {
    enemies,
    meta: metaModel.toJSON(),
    nn: { net: finalNN.net.toJSON(), nz: finalNN.nz },
    lr: finalLR.toJSON(),
    rf: finalRF.toJSON(),
    cvMetrics: { nn: mNN, lr: mLR, rf: mRF, stack: mStack },
  };

  fs.writeFileSync(CFG.out, JSON.stringify(bundle, null, 2));
  console.log(`Successfully saved complete stacked ensemble to ${CFG.out}`);
}

function runPredict() {
  const payloadStr = argv[predictIdx + 1];
  if (!payloadStr) {
    console.error('Error: Please provide a match payload JSON string, e.g.:');
    console.error('  node stacked_duel_model.js predict \'{"teamA":{"Snotty Slug":10},"teamB":{"Hound Pro":3},"storm":false}\'');
    process.exit(1);
  }

  if (!fs.existsSync(CFG.out)) {
    console.error(`Error: Saved model file not found at ${CFG.out}. Train first using:`);
    console.error('  node stacked_duel_model.js');
    process.exit(1);
  }

  const match = JSON.parse(payloadStr);
  const lookup = loadEnemies(CFG.enemies);
  const saved = JSON.parse(fs.readFileSync(CFG.out, 'utf8'));

  const modelNN = { net: MLP.fromJSON(saved.nn.net), nz: saved.nn.nz };
  const modelLR = LogisticRegressionModel.fromJSON(saved.lr);
  const modelRF = RandomForestModel.fromJSON(saved.rf);
  const meta = StackingMetaModel.fromJSON(saved.meta);

  const pNN = predictNn(modelNN, match, lookup);
  const pLR = modelLR.predictProba(match.teamA, match.teamB);
  const pRF = modelRF.predictProba(match.teamA, match.teamB);
  const pStack = meta.predictProba(pNN, pLR, pRF);

  console.log('\n--- Duel Channel Stacked Prediction ---');
  console.log(`  Neural Network       : ${(pNN * 100).toFixed(2)}% Team A`);
  console.log(`  Logistic Regression  : ${(pLR * 100).toFixed(2)}% Team A`);
  console.log(`  Random Forest        : ${(pRF * 100).toFixed(2)}% Team A`);
  console.log('----------------------------------------');
  console.log(`  FINAL STACKED RESULT : ${(pStack * 100).toFixed(2)}% Team A  |  ${((1 - pStack) * 100).toFixed(2)}% Team B`);
}

if (isPredict) {
  runPredict();
} else {
  runTraining();
}