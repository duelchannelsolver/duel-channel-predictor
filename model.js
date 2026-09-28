class EnemyStrengthModel {
  constructor(l2 = 1.0, stormWeight = 0.5) {
    this.l2 = l2;
    this.stormWeight = stormWeight;
    this.enemies = [];
    this.weights = [];
  }

  fit(matches, iters = 500) {
    // 1. Build unique enemy list
    const enemySet = new Set();
    matches.forEach(m => {
      if (m.teamA) Object.keys(m.teamA).forEach(e => enemySet.add(e));
      if (m.teamB) Object.keys(m.teamB).forEach(e => enemySet.add(e));
    });
    this.enemies = Array.from(enemySet).sort();

    const n = matches.length;
    const d = this.enemies.length;
    
    // 2. Strict Typed Arrays (Float64Array) prevent V8 de-optimization cliffs
    // A 1D flattened array is used for maximum memory access speed
    const X = new Float64Array(n * d);
    const y = new Float64Array(n);
    const sw = new Float64Array(n);

    for (let s = 0; s < n; s++) {
      const m = matches[s];
      y[s] = m.winner === "A" ? 1 : 0;
      
      // Look explicitly for the "storm" key matching your JSON schema
      sw[s] = m.storm === true ? this.stormWeight : 1.0; 

      for (let j = 0; j < d; j++) {
        const enemy = this.enemies[j];
        const countA = (m.teamA && m.teamA[enemy]) ? m.teamA[enemy] : 0;
        const countB = (m.teamB && m.teamB[enemy]) ? m.teamB[enemy] : 0;
        X[s * d + j] = countA - countB;
      }
    }

    const weights = new Float64Array(d);
    const mAdam = new Float64Array(d);
    const vAdam = new Float64Array(d);
    const grad = new Float64Array(d);
    
    const beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lr = 0.1;

    for (let it = 1; it <= iters; it++) {
      grad.fill(0);

      for (let s = 0; s < n; s++) {
        let z = 0;
        const offset = s * d;
        for (let j = 0; j < d; j++) {
          z += weights[j] * X[offset + j];
        }
        
        // Mathematical bounds check to prevent Infinity/NaN cascades
        let p;
        if (z > 20) p = 1;
        else if (z < -20) p = 0;
        else p = 1 / (1 + Math.exp(-z));
        
        const err = p - y[s];
        const weight = sw[s];
        
        for (let j = 0; j < d; j++) {
          grad[j] += weight * err * X[offset + j];
        }
      }

      for (let j = 0; j < d; j++) {
        // Scikit-learn exact L2 scaling
        grad[j] += this.l2 * weights[j];

        // Adam update rules
        mAdam[j] = beta1 * mAdam[j] + (1 - beta1) * grad[j];
        vAdam[j] = beta2 * vAdam[j] + (1 - beta2) * (grad[j] * grad[j]);
        
        const mHat = mAdam[j] / (1 - Math.pow(beta1, it));
        const vHat = vAdam[j] / (1 - Math.pow(beta2, it));
        
        weights[j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
      }
    }
    
    // Store final weights as a standard array for UI interactions
    this.weights = Array.from(weights);
  }

  strengths() {
    const result = {};
    this.enemies.forEach((e, i) => {
      result[e] = this.weights[i];
    });
    return result;
  }

  predictProba(teamA, teamB) {
    let z = 0;
    for (let i = 0; i < this.enemies.length; i++) {
      const enemy = this.enemies[i];
      const countA = teamA[enemy] || 0;
      const countB = teamB[enemy] || 0;
      z += this.weights[i] * (countA - countB);
    }
    return 1 / (1 + Math.exp(-z));
  }
}