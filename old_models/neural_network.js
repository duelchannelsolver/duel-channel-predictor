#!/usr/bin/env node
'use strict';
/**
 * Duel Channel match-outcome neural network (dependency-free, plain Node.js).
 *
 * Usage:
 *   node train_duel_net.js                                   # cross-validate, train on all data, save model.json
 *   node train_duel_net.js --epochs 150 --hidden 24,12 --lr 0.003 --wd 0.002 --seed 7
 *   node train_duel_net.js predict '{"teamA":{"Snotty Slug":10},"teamB":{"Hound Pro":3},"storm":false}'
 *
 * Files (override with --enemies / --matches / --out):
 *   duel_channel_enemies.json   enemy stats (HP, ATK, DEF, RES, etc.)
 *   matches.json                [{ teamA:{name:count}, teamB:{name:count}, winner:"A"|"B", storm:bool }]
 *   model.json                  written after training (weights + normalisation parameters)
 */

const fs = require('fs');
const path = require('path');

// ----------------------------------------------------------------------------------------------
// CLI / config
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
  enemies: flags.enemies || path.join(__dirname, 'duel_channel_enemies.json'),
  matches: flags.matches || path.join(__dirname, 'matches.json'),
  out: flags.out || path.join(__dirname, 'model.json'),
  hidden: (flags.hidden || '24,12').split(',').map(Number),
  epochs: Number(flags.epochs || 120),
  lr: Number(flags.lr || 0.003),
  wd: Number(flags.wd || 0.003),
  batch: Number(flags.batch || 32),
  folds: Number(flags.folds || 5),
  seed: Number(flags.seed || 42),
};

// ----------------------------------------------------------------------------------------------
// Math & RNG Utilities
// ----------------------------------------------------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
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

// ----------------------------------------------------------------------------------------------
// Enemy Data Extraction & Mapping
// ----------------------------------------------------------------------------------------------
function norm(s) {
  return String(s)
    .toLowerCase()
    .replace(/["'\u2019]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

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
  if (!fs.existsSync(file)) {
    throw new Error(`Enemies file not found at: ${file}`);
  }
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  const lookup = new Map();

  const add = (key, unit) => {
    if (!lookup.has(key)) lookup.set(key, []);
    const list = lookup.get(key);
    if (!list.some((u) => u.name === unit.name)) list.push(unit);
  };

  for (const e of raw) {
    const unit = makeUnit(e);
    const full = norm(e.Name);
    lookup.set(full, [unit]);

    // Handle abbreviated/pre-comma names (e.g. Qui'lon, Samivilinn)
    if (e.Name.includes(',')) {
      lookup.set(norm(e.Name.split(',')[0]), [unit]);
    }

    // Handle group encounters (e.g. Chivalrous Trio, Knights Together Strong, Will of Arbitration)
    if (e.Group) {
      add(norm(e.Group), unit);
    }
  }
  return lookup;
}

// ----------------------------------------------------------------------------------------------
// Feature Engineering
// ----------------------------------------------------------------------------------------------
function expandTeam(team, lookup) {
  const units = [];
  for (const [name, count] of Object.entries(team)) {
    const found = lookup.get(norm(name));
    if (!found) {
      throw new Error(`Unknown enemy "${name}" - not found in enemies database.`);
    }
    for (const u of found) {
      units.push({ u, n: count });
    }
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
    hp,
    count,
    phys,
    arts,
    hpMax,
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
    // Arknights damage formula with 5% ATK minimum damage floor
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
    Math.log1p(s.hp),
    Math.log1p(s.phys),
    Math.log1p(s.arts),
    Math.log1p(s.count),
    Math.log1p(s.avgDef),
    s.avgRes / 100,
    s.avgMs,
    s.avgRange,
    s.rangedFrac,
    Math.log1p(s.hp * (s.phys + s.arts)),
    Math.log1p(s.hpMax),
    ...s.flagSum.map((v) => Math.log1p(v)),
  ];
}

const MATCHUP_FEATURES = ['logTTK_A_kills_B', 'logTTK_B_kills_A', 'ttkLogRatio'];
const FEATURE_NAMES = [
  ...TEAM_FEATURES.map((n) => `A_${n}`),
  ...TEAM_FEATURES.map((n) => `B_${n}`),
  ...MATCHUP_FEATURES,
  'storm',
];

function buildFeatures(unitsA, unitsB, storm) {
  const sA = teamStats(unitsA), sB = teamStats(unitsB);
  const dpsAB = Math.max(effectiveDps(unitsA, sB.avgDef, sB.avgRes), 1e-3);
  const dpsBA = Math.max(effectiveDps(unitsB, sA.avgDef, sA.avgRes), 1e-3);

  const logTtkA = Math.log(Math.max(sB.hp, 1) / dpsAB);
  const logTtkB = Math.log(Math.max(sA.hp, 1) / dpsBA);

  return [
    ...teamFeatures(sA),
    ...teamFeatures(sB),
    clamp(logTtkA, -10, 15),
    clamp(logTtkB, -10, 15),
    clamp(logTtkB - logTtkA, -10, 10),
    storm ? 1 : 0,
  ];
}

function buildSample(match, lookup) {
  const uA = expandTeam(match.teamA, lookup);
  const uB = expandTeam(match.teamB, lookup);
  const storm = !!match.storm;
  return {
    fwd: buildFeatures(uA, uB, storm),
    swp: buildFeatures(uB, uA, storm),
    y: match.winner === 'A' ? 1 : 0,
  };
}

// ----------------------------------------------------------------------------------------------
// Neural Network (Multi-Layer Perceptron + Manual Adam Optimizer)
// ----------------------------------------------------------------------------------------------
class MLP {
  constructor(sizes, rng) {
    this.sizes = sizes;
    this.L = sizes.length - 1;
    this.W = [];
    this.b = [];
    for (let l = 0; l < this.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const w = new Float64Array(nin * nout);
      const scale = Math.sqrt(2 / nin); // He initialization
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

      if (l < this.L - 1) {
        for (let o = 0; o < nout; o++) z[o] = z[o] > 0 ? z[o] : 0; // ReLU
      } else {
        z[0] = sigmoid(z[0]);
      }

      a = z;
      if (cache) cache.a.push(a);
    }
    return a[0];
  }

  predict(x) {
    return this.forward(x, null);
  }

  backward(x, y, gW, gb) {
    const cache = {};
    const p = this.forward(x, cache);
    let dz = new Float64Array([p - y]); // d(BCE)/d(logit)

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
        for (let i = 0; i < nin; i++) da[i] = aPrev[i] > 0 ? da[i] : 0; // ReLU derivative
        dz = da;
      }
    }
    return p;
  }

  toJSON() {
    return {
      sizes: this.sizes,
      W: this.W.map((w) => Array.from(w)),
      b: this.b.map((b) => Array.from(b)),
    };
  }

  static fromJSON(j) {
    const m = new MLP(j.sizes, mulberry32(1));
    m.W = j.W.map((w) => Float64Array.from(w));
    m.b = j.b.map((b) => Float64Array.from(b));
    return m;
  }
}

function fitNormalizer(X) {
  const d = X[0].length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);

  for (const x of X) {
    for (let i = 0; i < d; i++) mean[i] += x[i] / X.length;
  }
  for (const x of X) {
    for (let i = 0; i < d; i++) std[i] += (x[i] - mean[i]) ** 2 / X.length;
  }
  return { mean, std: std.map((s) => Math.sqrt(s) || 1) };
}

const applyNorm = (x, nz) => x.map((v, i) => (v - nz.mean[i]) / nz.std[i]);

function train(samples, cfg, rng, onEpoch) {
  // Swap augmentation: balance team positions symmetrically
  const data = [];
  for (const s of samples) {
    data.push({ x: s.fwd, y: s.y });
    data.push({ x: s.swp, y: 1 - s.y });
  }

  const nz = fitNormalizer(data.map((d) => d.x));
  const X = data.map((d) => applyNorm(d.x, nz));
  const Y = data.map((d) => d.y);

  const sizes = [X[0].length, ...cfg.hidden, 1];
  const net = new MLP(sizes, rng);

  const mW = net.W.map((w) => new Float64Array(w.length));
  const vW = net.W.map((w) => new Float64Array(w.length));
  const mB = net.b.map((b) => new Float64Array(b.length));
  const vB = net.b.map((b) => new Float64Array(b.length));

  const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
  let step = 0;
  const idx = X.map((_, i) => i);

  for (let epoch = 1; epoch <= cfg.epochs; epoch++) {
    shuffle(idx, rng);
    let loss = 0;

    for (let start = 0; start < idx.length; start += cfg.batch) {
      const batch = idx.slice(start, start + cfg.batch);
      const gW = net.W.map((w) => new Float64Array(w.length));
      const gb = net.b.map((b) => new Float64Array(b.length));

      for (const i of batch) {
        const p = net.backward(X[i], Y[i], gW, gb);
        loss += -(Y[i] * Math.log(p + 1e-12) + (1 - Y[i]) * Math.log(1 - p + 1e-12));
      }

      step++;
      const c1 = 1 - Math.pow(B1, step);
      const c2 = 1 - Math.pow(B2, step);

      for (let l = 0; l < net.L; l++) {
        for (let k = 0; k < net.W[l].length; k++) {
          const g = gW[l][k] / batch.length + cfg.wd * net.W[l][k]; // L2 weight decay
          mW[l][k] = B1 * mW[l][k] + (1 - B1) * g;
          vW[l][k] = B2 * vW[l][k] + (1 - B2) * g * g;
          net.W[l][k] -= (cfg.lr * (mW[l][k] / c1)) / (Math.sqrt(vW[l][k] / c2) + EPS);
        }
        for (let k = 0; k < net.b[l].length; k++) {
          const g = gb[l][k] / batch.length;
          mB[l][k] = B1 * mB[l][k] + (1 - B1) * g;
          vB[l][k] = B2 * vB[l][k] + (1 - B2) * g * g;
          net.b[l][k] -= (cfg.lr * (mB[l][k] / c1)) / (Math.sqrt(vB[l][k] / c2) + EPS);
        }
      }
    }

    if (onEpoch) onEpoch(epoch, loss / X.length, { net, nz });
  }

  return { net, nz };
}

function predictSample(model, s) {
  const pf = model.net.predict(applyNorm(s.fwd, model.nz));
  const ps = model.net.predict(applyNorm(s.swp, model.nz));
  return 0.5 * (pf + (1 - ps));
}

function evaluate(model, samples) {
  let correct = 0, ll = 0;
  for (const s of samples) {
    const p = clamp(predictSample(model, s), 1e-6, 1 - 1e-6);
    if ((p >= 0.5) === (s.y === 1)) correct++;
    ll += -(s.y * Math.log(p) + (1 - s.y) * Math.log(1 - p));
  }
  return { acc: correct / samples.length, logLoss: ll / samples.length };
}

// ----------------------------------------------------------------------------------------------
// Run Routines
// ----------------------------------------------------------------------------------------------
function main() {
  const lookup = loadEnemies(CFG.enemies);
  const matches = JSON.parse(fs.readFileSync(CFG.matches, 'utf8'));
  const samples = matches.map((m) => buildSample(m, lookup));
  const D = samples[0].fwd.length;

  console.log(`Loaded ${lookup.size} enemy entries, ${samples.length} matches, ${D} features per matchup.`);
  console.log(`Architecture: ${D} -> ${CFG.hidden.join(' -> ')} -> 1 (epochs=${CFG.epochs}, lr=${CFG.lr}, wd=${CFG.wd}, batch=${CFG.batch})`);

  // Kill-time baseline
  const ratioIdx = FEATURE_NAMES.indexOf('ttkLogRatio');
  const base = samples.filter((s) => (s.fwd[ratioIdx] > 0) === (s.y === 1)).length / samples.length;
  console.log(`Baseline (faster Lanchester TTK wins): ${(base * 100).toFixed(1)}% accuracy`);

  // K-fold cross validation
  const rng = mulberry32(CFG.seed);
  const order = shuffle(samples.map((_, i) => i), rng);
  const accs = [], lls = [];

  for (let f = 0; f < CFG.folds; f++) {
    const testIdx = new Set(order.filter((_, k) => k % CFG.folds === f));
    const tr = samples.filter((_, i) => !testIdx.has(i));
    const te = samples.filter((_, i) => testIdx.has(i));
    const model = train(tr, CFG, mulberry32(CFG.seed + f + 1));
    const r = evaluate(model, te);
    accs.push(r.acc);
    lls.push(r.logLoss);
    console.log(`  Fold ${f + 1}/${CFG.folds}: accuracy ${(r.acc * 100).toFixed(1)}% | log-loss ${r.logLoss.toFixed(3)} (${te.length} held-out matches)`);
  }

  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  console.log(`Cross-validated Accuracy: ${(mean(accs) * 100).toFixed(1)}% | Log-loss: ${mean(lls).toFixed(3)}`);

  // Final training pass on complete dataset
  const final = train(samples, CFG, mulberry32(CFG.seed + 999), (e, loss) => {
    if (e === 1 || e % 20 === 0 || e === CFG.epochs) {
      console.log(`  Epoch ${String(e).padStart(3)} | train loss ${loss.toFixed(4)}`);
    }
  });

  const trainRes = evaluate(final, samples);
  console.log(`Final training-set accuracy: ${(trainRes.acc * 100).toFixed(1)}%`);

  fs.writeFileSync(
    CFG.out,
    JSON.stringify({
      featureNames: FEATURE_NAMES,
      normalizer: final.nz,
      network: final.net.toJSON(),
      cvAccuracy: mean(accs),
      cvLogLoss: mean(lls),
    }, null, 2)
  );
  console.log(`Saved model checkpoint to ${CFG.out}`);
}

function predictCli() {
  const matchStr = argv[predictIdx + 1];
  if (!matchStr) {
    console.error('Error: Please provide a match payload string, e.g.:');
    console.error('  node train_duel_net.js predict \'{"teamA":{"Snotty Slug":10},"teamB":{"Hound Pro":3},"storm":false}\'');
    process.exit(1);
  }

  if (!fs.existsSync(CFG.out)) {
    console.error(`Error: Saved model "${CFG.out}" does not exist. Run training first:\n  node train_duel_net.js`);
    process.exit(1);
  }

  const lookup = loadEnemies(CFG.enemies);
  const saved = JSON.parse(fs.readFileSync(CFG.out, 'utf8'));
  const model = { net: MLP.fromJSON(saved.network), nz: saved.normalizer };
  const match = JSON.parse(matchStr);

  const p = predictSample(model, buildSample({ ...match, winner: 'A' }, lookup));
  console.log(`P(team A wins) = ${(p * 100).toFixed(1)}% | P(team B wins) = ${((1 - p) * 100).toFixed(1)}%`);
}

if (isPredict) {
  predictCli();
} else {
  main();
}