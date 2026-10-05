/**
 * DuelStackModel (stack.js) - Pure Browser Version
 * No Node.js dependencies (no require, no fs, no path).
 * Mounts directly to window.DuelStackModel.
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Utilities & Math
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // Domain Stats & Feature Extraction
  // ---------------------------------------------------------------------------
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

  function teamStats(units) {
    let hp = 0, count = 0, phys = 0, arts = 0, defW = 0, resW = 0, ms = 0, rng = 0, ranged = 0, hpMax = 0;
    const flagSum = new Array(FLAG_NAMES.length).fill(0);
    for (const { u, n } of units) {
      const dps = u.atk / u.ai;
      hp += n * u.hp; count += n;
      phys += n * dps * u.physShare; arts += n * dps * (1 - u.physShare);
      defW += n * u.hp * u.def; resW += n * u.hp * u.res;
      ms += n * u.ms; rng += n * u.range;
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

  // ---------------------------------------------------------------------------
  // Base Models
  // ---------------------------------------------------------------------------
  class BrowserMLP {
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
  }

  // ---------------------------------------------------------------------------
  // Master Stacking Classifier Class
  // ---------------------------------------------------------------------------
  class DuelStackModel {
    constructor(options = {}) {
      this.rfTrees = options.rfTrees ?? 100;
      this.rfDepth = options.rfDepth ?? 12;
      this.rfMinLeaf = options.rfMinLeaf ?? 3;
      this.nnEpochs = options.nnEpochs ?? 80;
      this.nnHidden = options.nnHidden ?? [24, 12];
      this.folds = options.folds ?? 3;
      this.seed = options.seed ?? 42;

      this.enemies = [];
      this.enemyLookup = new Map();
      this.weights = [0.333, 0.333, 0.333];
    }

    _expandTeam(team) {
      const units = [];
      for (const [name, count] of Object.entries(team)) {
        const found = this.enemyLookup.get(norm(name));
        if (found) {
          for (const u of found) units.push({ u, n: count });
        }
      }
      return units;
    }

    _trainLR(matches, enemies) {
      const n = matches.length, d = enemies.length;
      const X = new Float64Array(n * d), y = new Float64Array(n), sw = new Float64Array(n);
      for (let s = 0; s < n; s++) {
        const m = matches[s];
        y[s] = m.winner === 'A' ? 1 : 0;
        sw[s] = m.storm === true ? 0.75 : 1.0;
        for (let j = 0; j < d; j++) {
          const e = enemies[j];
          X[s * d + j] = Math.sqrt((m.teamA && m.teamA[e]) || 0) - Math.sqrt((m.teamB && m.teamB[e]) || 0);
        }
      }

      const w = new Float64Array(d), mW = new Float64Array(d), vW = new Float64Array(d);
      for (let it = 1; it <= 500; it++) {
        const grad = new Float64Array(d);
        for (let s = 0; s < n; s++) {
          let z = 0; const off = s * d;
          for (let j = 0; j < d; j++) z += w[j] * X[off + j];
          const err = (z > 20 ? 1 : z < -20 ? 0 : 1 / (1 + Math.exp(-z))) - y[s];
          for (let j = 0; j < d; j++) grad[j] += sw[s] * err * X[off + j];
        }
        for (let j = 0; j < d; j++) {
          grad[j] += 1.0 * w[j];
          mW[j] = 0.9 * mW[j] + 0.1 * grad[j];
          vW[j] = 0.999 * vW[j] + 0.001 * (grad[j] * grad[j]);
          w[j] -= (0.1 * (mW[j] / (1 - Math.pow(0.9, it)))) / (Math.sqrt(vW[j] / (1 - Math.pow(0.999, it))) + 1e-8);
        }
      }

      return {
        predict: (tA, tB) => {
          let z = 0;
          for (let j = 0; j < d; j++) {
            const e = enemies[j];
            z += w[j] * (Math.sqrt(tA[e] || 0) - Math.sqrt(tB[e] || 0));
          }
          return sigmoid(z);
        },
      };
    }

    _trainRF(matches, enemies, rng) {
      const n = matches.length, d = enemies.length;
      const X = new Float64Array(2 * n * d), y = new Uint8Array(2 * n), sw = new Float64Array(2 * n);
      for (let s = 0; s < n; s++) {
        const m = matches[s];
        const lbl = m.winner === 'A' ? 1 : 0;
        const w = m.storm === true ? 0.75 : 1.0;
        y[s] = lbl; y[s + n] = 1 - lbl;
        sw[s] = sw[s + n] = w;
        for (let j = 0; j < d; j++) {
          const diff = ((m.teamA && m.teamA[enemies[j]]) || 0) - ((m.teamB && m.teamB[enemies[j]]) || 0);
          X[s * d + j] = diff;
          X[(s + n) * d + j] = -diff;
        }
      }

      const mtry = Math.max(1, Math.floor(Math.sqrt(d)));
      const buildTree = (rows, depth) => {
        let W = 0, W1 = 0;
        for (const r of rows) { W += sw[r]; if (y[r] === 1) W1 += sw[r]; }
        if (depth >= this.rfDepth || rows.length < 2 * this.rfMinLeaf || W1 === 0 || W1 === W) {
          return { p: W > 0 ? W1 / W : 0.5 };
        }
        const parentImp = 2 * (W1 / W) * (1 - W1 / W);
        const feats = new Int32Array(d);
        for (let j = 0; j < d; j++) feats[j] = j;
        for (let j = 0; j < Math.min(mtry, d); j++) {
          const k = j + Math.floor(rng() * (d - j));
          const tmp = feats[j]; feats[j] = feats[k]; feats[k] = tmp;
        }

        let bestG = 1e-12, bestF = -1, bestT = 0;
        for (let fi = 0; fi < Math.min(mtry, d); fi++) {
          const f = feats[fi];
          const sorted = rows.slice().sort((a, b) => X[a * d + f] - X[b * d + f]);
          let WL = 0, WL1 = 0;
          for (let i = 0; i < sorted.length - 1; i++) {
            const r = sorted[i]; WL += sw[r]; if (y[r] === 1) WL1 += sw[r];
            const v = X[r * d + f], vNext = X[sorted[i + 1] * d + f];
            if (v === vNext || (i + 1) < this.rfMinLeaf || (sorted.length - i - 1) < this.rfMinLeaf) continue;
            const WR = W - WL, WR1 = W1 - WL1;
            const g = W * parentImp - WL * (2 * (WL1 / WL) * (1 - WL1 / WL)) - WR * (2 * (WR1 / WR) * (1 - WR1 / WR));
            if (g > bestG) { bestG = g; bestF = f; bestT = (v + vNext) / 2; }
          }
        }
        if (bestF < 0) return { p: W > 0 ? W1 / W : 0.5 };
        const left = [], right = [];
        for (const r of rows) (X[r * d + bestF] <= bestT ? left : right).push(r);
        return { f: bestF, t: bestT, l: buildTree(left, depth + 1), r: buildTree(right, depth + 1) };
      };

      const trees = [];
      for (let t = 0; t < this.rfTrees; t++) {
        const rows = [];
        for (let i = 0; i < n; i++) { const k = Math.floor(rng() * n); rows.push(k, k + n); }
        trees.push(buildTree(rows, 0));
      }

      const predVec = (tree, x) => {
        let cur = tree;
        while (cur.p === undefined) cur = x[cur.f] <= cur.t ? cur.l : cur.r;
        return cur.p;
      };

      return {
        predict: (tA, tB) => {
          const x = new Float64Array(d);
          for (let j = 0; j < d; j++) x[j] = (tA[enemies[j]] || 0) - (tB[enemies[j]] || 0);
          const neg = x.map((v) => -v);
          let s1 = 0, s2 = 0;
          for (const tr of trees) { s1 += predVec(tr, x); s2 += predVec(tr, neg); }
          return 0.5 * (s1 / trees.length + (1 - s2 / trees.length));
        },
      };
    }

    _trainNN(matches, rng) {
      const rawX = [], rawY = [];
      for (const m of matches) {
        const uA = this._expandTeam(m.teamA), uB = this._expandTeam(m.teamB);
        rawX.push(buildNnFeatures(uA, uB, !!m.storm)); rawY.push(m.winner === 'A' ? 1 : 0);
        rawX.push(buildNnFeatures(uB, uA, !!m.storm)); rawY.push(m.winner === 'A' ? 0 : 1);
      }
      const dim = rawX[0].length;
      const mean = new Array(dim).fill(0), std = new Array(dim).fill(0);
      for (const x of rawX) for (let j = 0; j < dim; j++) mean[j] += x[j] / rawX.length;
      for (const x of rawX) for (let j = 0; j < dim; j++) std[j] += (x[j] - mean[j]) ** 2 / rawX.length;
      for (let j = 0; j < dim; j++) std[j] = Math.sqrt(std[j]) || 1;

      const normVec = (v) => v.map((x, j) => (x - mean[j]) / std[j]);
      const X = rawX.map(normVec), Y = rawY;

      const sizes = [dim, ...this.nnHidden, 1];
      const net = new BrowserMLP(sizes, rng);
      const mW = net.W.map((w) => new Float64Array(w.length)), vW = net.W.map((w) => new Float64Array(w.length));
      const mB = net.b.map((cb) => new Float64Array(cb.length)), vB = net.b.map((cb) => new Float64Array(cb.length));
      const idx = X.map((_, i) => i);
      let step = 0;

      for (let ep = 0; ep < this.nnEpochs; ep++) {
        shuffle(idx, rng);
        for (let s = 0; s < idx.length; s += 32) {
          const batch = idx.slice(s, s + 32);
          const gW = net.W.map((w) => new Float64Array(w.length));
          const gb = net.b.map((cb) => new Float64Array(cb.length));
          for (const i of batch) net.backward(X[i], Y[i], gW, gb);

          step++;
          const c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
          for (let l = 0; l < net.L; l++) {
            for (let k = 0; k < net.W[l].length; k++) {
              const g = gW[l][k] / batch.length + 0.003 * net.W[l][k];
              mW[l][k] = 0.9 * mW[l][k] + 0.1 * g;
              vW[l][k] = 0.999 * vW[l][k] + 0.001 * g * g;
              net.W[l][k] -= (0.003 * (mW[l][k] / c1)) / (Math.sqrt(vW[l][k] / c2) + 1e-8);
            }
            for (let k = 0; k < net.b[l].length; k++) {
              const g = gb[l][k] / batch.length;
              mB[l][k] = 0.9 * mB[l][k] + 0.1 * g;
              vB[l][k] = 0.999 * vB[l][k] + 0.001 * g * g;
              net.b[l][k] -= (0.003 * (mB[l][k] / c1)) / (Math.sqrt(vB[l][k] / c2) + 1e-8);
            }
          }
        }
      }

      return {
        predict: (tA, tB, storm) => {
          const uA = this._expandTeam(tA), uB = this._expandTeam(tB);
          const fwd = normVec(buildNnFeatures(uA, uB, !!storm));
          const swp = normVec(buildNnFeatures(uB, uA, !!storm));
          return 0.5 * (net.predict(fwd) + (1 - net.predict(swp)));
        },
      };
    }

    fit(matches, enemiesData) {
      if (!matches || !matches.length) throw new Error("No matches provided");

      // Setup enemy lookup
      const lookup = new Map();
      const add = (k, u) => {
        if (!lookup.has(k)) lookup.set(k, []);
        const l = lookup.get(k);
        if (!l.some((x) => x.name === u.name)) l.push(u);
      };

      for (const e of enemiesData || []) {
        const unit = makeUnit(e);
        lookup.set(norm(e.Name), [unit]);
        if (e.Name.includes(',')) lookup.set(norm(e.Name.split(',')[0]), [unit]);
        if (e.Group) add(norm(e.Group), unit);
      }
      this.enemyLookup = lookup;

      const eSet = new Set();
      for (const m of matches) {
        if (m.teamA) Object.keys(m.teamA).forEach((e) => eSet.add(e));
        if (m.teamB) Object.keys(m.teamB).forEach((e) => eSet.add(e));
      }
      this.enemies = Array.from(eSet).sort();

      // K-Fold Cross Validation for Meta-Learner
      const K = Math.min(this.folds, matches.length);
      const oofLogits = [];
      const oofTargets = [];

      for (let f = 0; f < K; f++) {
        const tr = matches.filter((_, i) => i % K !== f);
        const te = matches.filter((_, i) => i % K === f);
        const foldLR = this._trainLR(tr, this.enemies);
        const foldRF = this._trainRF(tr, this.enemies, mulberry32(this.seed + f * 17 + 1));
        const foldNN = this._trainNN(tr, mulberry32(this.seed + f * 31 + 1));

        for (const m of te) {
          const pNN = foldNN.predict(m.teamA, m.teamB, m.storm);
          const pLR = foldLR.predict(m.teamA, m.teamB);
          const pRF = foldRF.predict(m.teamA, m.teamB);
          oofLogits.push([logit(pNN), logit(pLR), logit(pRF)]);
          oofTargets.push(m.winner === 'A' ? 1 : 0);
        }
      }

      // Meta weights optimization
      const w = new Float64Array([0.333, 0.333, 0.333]);
      const mW = new Float64Array(3), vW = new Float64Array(3);
      for (let it = 1; it <= 400; it++) {
        const grad = new Float64Array(3);
        for (let s = 0; s < oofLogits.length; s++) {
          let z = w[0] * oofLogits[s][0] + w[1] * oofLogits[s][1] + w[2] * oofLogits[s][2];
          const err = sigmoid(z) - oofTargets[s];
          grad[0] += (err * oofLogits[s][0]) / oofLogits.length;
          grad[1] += (err * oofLogits[s][1]) / oofLogits.length;
          grad[2] += (err * oofLogits[s][2]) / oofLogits.length;
        }
        for (let j = 0; j < 3; j++) {
          grad[j] += 0.001 * w[j];
          mW[j] = 0.9 * mW[j] + 0.1 * grad[j];
          vW[j] = 0.999 * vW[j] + 0.001 * grad[j] * grad[j];
          w[j] -= (0.05 * (mW[j] / (1 - Math.pow(0.9, it)))) / (Math.sqrt(vW[j] / (1 - Math.pow(0.999, it))) + 1e-8);
        }
      }
      this.weights = Array.from(w);

      // Final base models trained on full dataset
      this.modelLR = this._trainLR(matches, this.enemies);
      this.modelRF = this._trainRF(matches, this.enemies, mulberry32(this.seed + 999));
      this.modelNN = this._trainNN(matches, mulberry32(this.seed + 888));
    }

    predictDetailed(teamA, teamB, storm = false) {
      const pNN = this.modelNN.predict(teamA, teamB, storm);
      const pLR = this.modelLR.predict(teamA, teamB);
      const pRF = this.modelRF.predict(teamA, teamB);

      const z = this.weights[0] * logit(pNN) + this.weights[1] * logit(pLR) + this.weights[2] * logit(pRF);
      return {
        p: sigmoid(z),
        nn: pNN,
        lr: pLR,
        rf: pRF,
        weights: this.weights,
      };
    }

    predictProba(teamA, teamB, storm = false) {
      return this.predictDetailed(teamA, teamB, storm).p;
    }
  }

  // Mount to browser global scope
  global.DuelStackModel = DuelStackModel;
  global.EnemyStrengthModel = DuelStackModel;
})(typeof window !== 'undefined' ? window : this);