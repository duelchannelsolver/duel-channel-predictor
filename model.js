/**
 * EnemyStrengthModel (Random Forest edition)
 * Predicts the probability of Team A winning a duel based on unit composition.
 *
 * Features: for every enemy type, sqrt(countA) - sqrt(countB)  (same as before)
 * Model:    bagged ensemble of CART decision trees (weighted Gini splits,
 *           random feature subsets at every split).
 *
 * Extras:
 *  - Symmetry augmentation: every match is also added with A/B swapped
 *    (features negated, label flipped), so the model can't favour "side A".
 *  - predictProba is symmetrised: P(A beats B) = 1 - P(B beats A).
 *  - Storm matches get a lower sample weight (stormWeight), as before.
 *  - Out-of-bag accuracy is reported after training.
 *  - strengths() now returns normalised feature importances (0..1, sum to 1).
 *    NOTE: these are unsigned "how much this enemy matters" scores, not
 *    signed linear weights like in the logistic regression version.
 */
class EnemyStrengthModel {
  constructor(options = {}) {
    // Backwards-compatible: new EnemyStrengthModel(l2, stormWeight) still works
    // (l2 is ignored; random forests don't use it).
    if (typeof options === "number") {
      options = { stormWeight: arguments[1] };
    }
    const o = options;
    this.nTrees = o.nTrees ?? 200;
    this.maxDepth = o.maxDepth ?? 12;
    this.minSamplesLeaf = o.minSamplesLeaf ?? 3;
    this.maxFeatures = o.maxFeatures ?? null; // null => floor(sqrt(d))
    this.stormWeight = o.stormWeight ?? 0.75;
    this.seed = o.seed ?? 42;

    this.enemies = [];
    this.trees = [];
    this.importances = [];
    this.oobAccuracy = null;
  }

  // Small, fast, seedable PRNG (mulberry32)
  _makeRng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  _feature(countA, countB) {
    return Math.sqrt(countA) - Math.sqrt(countB);
  }

  fit(matches) {
    if (!matches || matches.length === 0) {
      console.error("Training failed: No matches provided.");
      return;
    }

    // 1. Build unique enemy list
    const enemySet = new Set();
    matches.forEach(m => {
      if (m.teamA) Object.keys(m.teamA).forEach(e => enemySet.add(e));
      if (m.teamB) Object.keys(m.teamB).forEach(e => enemySet.add(e));
    });
    this.enemies = Array.from(enemySet).sort();

    const n = matches.length;
    const d = this.enemies.length;

    // 2. Typed arrays. Rows 0..n-1 are the real matches; rows n..2n-1 are the
    //    A/B-swapped mirror images (negated features, flipped label).
    const X = new Float64Array(2 * n * d);
    const y = new Uint8Array(2 * n);
    const sw = new Float64Array(2 * n);

    for (let s = 0; s < n; s++) {
      const m = matches[s];
      const label = m.winner === "A" ? 1 : 0;
      const w = m.storm === true ? this.stormWeight : 1.0;

      y[s] = label;
      y[s + n] = 1 - label;
      sw[s] = sw[s + n] = w;

      for (let j = 0; j < d; j++) {
        const enemy = this.enemies[j];
        const countA = parseInt((m.teamA && m.teamA[enemy]) || 0, 10) || 0;
        const countB = parseInt((m.teamB && m.teamB[enemy]) || 0, 10) || 0;
        const f = this._feature(countA, countB);
        X[s * d + j] = f;
        X[(s + n) * d + j] = -f;
      }
    }

    const rng = this._makeRng(this.seed);
    const mtry = this.maxFeatures ?? Math.max(1, Math.floor(Math.sqrt(d)));
    const importances = new Float64Array(d);
    this.trees = [];

    // Out-of-bag accumulators (over the original, un-swapped matches)
    const oobSum = new Float64Array(n);
    const oobCnt = new Uint32Array(n);

    // 3. Grow the forest
    for (let t = 0; t < this.nTrees; t++) {
      // Bootstrap sample of matches; each drawn match contributes both rows.
      const inBag = new Uint8Array(n);
      const rows = [];
      for (let i = 0; i < n; i++) {
        const k = Math.floor(rng() * n);
        inBag[k] = 1;
        rows.push(k, k + n);
      }

      const tree = this._buildTree(X, y, sw, d, rows, 0, mtry, rng, importances);
      this.trees.push(tree);

      // OOB predictions for matches this tree never saw
      for (let s = 0; s < n; s++) {
        if (inBag[s]) continue;
        oobSum[s] += this._predictTreeRow(tree, X, s * d);
        oobCnt[s]++;
      }
    }

    // 4. Importances + OOB accuracy
    let total = 0;
    for (let j = 0; j < d; j++) total += importances[j];
    this.importances = Array.from(importances, v => (total > 0 ? v / total : 0));

    let correct = 0, counted = 0;
    for (let s = 0; s < n; s++) {
      if (oobCnt[s] === 0) continue;
      counted++;
      if ((oobSum[s] / oobCnt[s] >= 0.5 ? 1 : 0) === y[s]) correct++;
    }
    this.oobAccuracy = counted > 0 ? correct / counted : null;
  }

  /**
   * Recursively builds one CART tree using weighted Gini impurity.
   * Leaves store the weighted fraction of "A wins".
   */
  _buildTree(X, y, sw, d, rows, depth, mtry, rng, importances) {
    let W = 0, W1 = 0;
    for (const r of rows) {
      W += sw[r];
      if (y[r] === 1) W1 += sw[r];
    }
    const leaf = { p: W > 0 ? W1 / W : 0.5 };

    if (
      depth >= this.maxDepth ||
      rows.length < 2 * this.minSamplesLeaf ||
      W1 === 0 || W1 === W
    ) {
      return leaf;
    }

    const parentImp = 2 * (W1 / W) * (1 - W1 / W); // Gini
    const minLeaf = this.minSamplesLeaf;

    // Random feature subset (partial Fisher-Yates)
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
        if (v === vNext) continue; // can only split between distinct values

        const nL = i + 1, nR = sorted.length - nL;
        if (nL < minLeaf || nR < minLeaf) continue;

        const WR = W - WL, WR1 = W1 - WL1;
        const pL = WL1 / WL, pR = WR1 / WR;
        const impL = 2 * pL * (1 - pL);
        const impR = 2 * pR * (1 - pR);
        const gain = W * parentImp - WL * impL - WR * impR;

        if (gain > bestGain) {
          bestGain = gain;
          bestF = f;
          bestT = (v + vNext) / 2;
        }
      }
    }

    if (bestF < 0) return leaf;

    importances[bestF] += bestGain;

    const left = [], right = [];
    for (const r of rows) {
      (X[r * d + bestF] <= bestT ? left : right).push(r);
    }

    return {
      f: bestF,
      t: bestT,
      l: this._buildTree(X, y, sw, d, left, depth + 1, mtry, rng, importances),
      r: this._buildTree(X, y, sw, d, right, depth + 1, mtry, rng, importances),
    };
  }

  _predictTreeRow(node, X, offset) {
    while (node.p === undefined) {
      node = X[offset + node.f] <= node.t ? node.l : node.r;
    }
    return node.p;
  }

  _predictTreeVec(node, x) {
    while (node.p === undefined) {
      node = x[node.f] <= node.t ? node.l : node.r;
    }
    return node.p;
  }

  /** Normalised feature importance per enemy (higher = matters more). */
  strengths() {
    const result = {};
    this.enemies.forEach((e, i) => {
      result[e] = this.importances[i];
    });
    return result;
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
      const enemy = this.enemies[i];
      const countA = parseInt(teamA[enemy] || 0, 10) || 0;
      const countB = parseInt(teamB[enemy] || 0, 10) || 0;
      x[i] = this._feature(countA, countB);
    }
    const neg = x.map(v => -v);

    // Symmetrise: P(A wins) = average of P(A) and 1 - P(swapped)
    return 0.5 * (this._forestProba(x) + (1 - this._forestProba(neg)));
  }
}

/**
 * Initializes the pipeline: Fetches matches.json, trains the model,
 * and makes it available to the global window for UI usage.
 */
async function loadAndTrain() {
  try {
    console.time("Total Load & Train Time");

    console.log("Fetching matches.json...");
    // Ensure the path is correct for your GitHub Pages deployment
    const response = await fetch('data/matches.json');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}. Check if matches.json exists at this URL.`);
    }

    const matches = await response.json();
    console.log(`Successfully parsed ${matches.length} matches.`);

    console.log("Starting random forest training...");
    const model = new EnemyStrengthModel({
      nTrees: 200,
      maxDepth: 12,
      minSamplesLeaf: 3,
      stormWeight: 0.5,
      seed: 42,
    });

    model.fit(matches);

    console.timeEnd("Total Load & Train Time");

    if (model.oobAccuracy !== null) {
      console.log(`Out-of-bag accuracy: ${(model.oobAccuracy * 100).toFixed(1)}%`);
    }

    // Expose the fully trained model so you can call
    // `window.duelModel.predictProba(teamA, teamB)` from your UI.
    window.duelModel = model;

    // Print the most influential enemies to the console
    const strengths = model.strengths();
    const sortedEnemies = Object.keys(strengths).sort((a, b) => strengths[b] - strengths[a]);
    console.log("Training complete. Most influential enemies (feature importance):");
    sortedEnemies.slice(0, 10).forEach(enemy => {
      console.log(`  ${enemy}: ${strengths[enemy].toFixed(3)}`);
    });

  } catch (error) {
    console.error("CRITICAL FAILURE in loadAndTrain:", error);
  }
}

// Not run automatically: app.js trains its own model on page load.
// Call loadAndTrain() manually (e.g. from the console) if you want a
// standalone model on window.duelModel.