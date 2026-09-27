// P(A wins) = sigmoid( sum_i strength_i * (count_i(A) - count_i(B)) )
// Trained by plain gradient descent -- with ~100 rows and a few dozen
// enemies this converges in well under a second, so we train fresh on
// every page load rather than shipping precomputed weights.

function trainModel(matches, { l2 = 1.0, lr = 0.1, iters = 2000 } = {}) {
  const enemySet = new Set();
  matches.forEach((m) => {
    Object.keys(m.teamA).forEach((e) => enemySet.add(e));
    Object.keys(m.teamB).forEach((e) => enemySet.add(e));
  });
  const enemies = Array.from(enemySet).sort();
  const index = new Map(enemies.map((e, i) => [e, i]));

  const X = matches.map((m) => {
    const row = new Array(enemies.length).fill(0);
    for (const [e, c] of Object.entries(m.teamA)) row[index.get(e)] += c;
    for (const [e, c] of Object.entries(m.teamB)) row[index.get(e)] -= c;
    return row;
  });
  const y = matches.map((m) => (m.winner === "A" ? 1 : 0));
  const weights = new Array(enemies.length).fill(0);
  const n = matches.length;

  for (let it = 0; it < iters; it++) {
    const grad = new Array(enemies.length).fill(0);
    for (let s = 0; s < n; s++) {
      let z = 0;
      for (let j = 0; j < enemies.length; j++) z += weights[j] * X[s][j];
      const p = 1 / (1 + Math.exp(-z));
      const err = p - y[s];
      for (let j = 0; j < enemies.length; j++) grad[j] += err * X[s][j];
    }
    for (let j = 0; j < enemies.length; j++) {
      grad[j] = grad[j] / n + (l2 / n) * weights[j]; // L2 regularization
      weights[j] -= lr * grad[j];
    }
  }

  return { enemies, weights };
}

function predictProba(model, teamA, teamB) {
  const idx = new Map(model.enemies.map((e, i) => [e, i]));
  let z = 0;
  for (const [e, c] of Object.entries(teamA)) {
    if (idx.has(e)) z += model.weights[idx.get(e)] * c;
  }
  for (const [e, c] of Object.entries(teamB)) {
    if (idx.has(e)) z -= model.weights[idx.get(e)] * c;
  }
  return 1 / (1 + Math.exp(-z));
}
