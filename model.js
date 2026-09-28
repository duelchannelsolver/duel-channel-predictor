/**
 * EnemyStrengthModel
 * Predicts the probability of Team A winning a duel based on unit composition.
 */
class EnemyStrengthModel {
  constructor(l2 = 1.0, stormWeight = 0.5) {
    this.l2 = l2;
    this.stormWeight = stormWeight;
    this.enemies = [];
    this.weights = [];
  }

  fit(matches, iters = 500) {
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
    
    // 2. Strict Typed Arrays prevent browser engine de-optimization
    const X = new Float64Array(n * d);
    const y = new Float64Array(n);
    const sw = new Float64Array(n);

    // 3. Parse and sanitize data
    for (let s = 0; s < n; s++) {
      const m = matches[s];
      y[s] = m.winner === "A" ? 1 : 0;
      
      // Match the "storm" boolean from matches.json
      sw[s] = m.storm === true ? this.stormWeight : 1.0; 

      for (let j = 0; j < d; j++) {
        const enemy = this.enemies[j];
        
        // parseInt guarantees we don't accidentally do string concatenation 
        // which causes the NaN / Infinity math cascades that freeze the browser
        const countA = parseInt((m.teamA && m.teamA[enemy]) || 0, 10);
        const countB = parseInt((m.teamB && m.teamB[enemy]) || 0, 10);
        
        X[s * d + j] = countA - countB;
      }
    }

    const weights = new Float64Array(d);
    const mAdam = new Float64Array(d);
    const vAdam = new Float64Array(d);
    const grad = new Float64Array(d);
    
    const beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lr = 0.1;

    // 4. Training Loop (Adam Optimizer)
    for (let it = 1; it <= iters; it++) {
      grad.fill(0);

      for (let s = 0; s < n; s++) {
        let z = 0;
        const offset = s * d;
        for (let j = 0; j < d; j++) {
          z += weights[j] * X[offset + j];
        }
        
        // Safe Sigmoid bounds check
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
    
    // Store final weights as a standard JavaScript array
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
      const countA = parseInt(teamA[enemy] || 0, 10);
      const countB = parseInt(teamB[enemy] || 0, 10);
      z += this.weights[i] * (countA - countB);
    }
    return 1 / (1 + Math.exp(-z));
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

    console.log("Starting model training...");
    const model = new EnemyStrengthModel(1.0, 0.5);
    
    // Train the model with 500 iterations
    model.fit(matches, 500); 
    
    console.timeEnd("Total Load & Train Time");
    
    // Expose the fully trained model to the global window object 
    // so you can use `window.duelModel.predictProba(...)` in your UI buttons.
    window.duelModel = model;
    
    // Print the sorted tier list to the console to verify it worked
    const strengths = model.strengths();
    const sortedEnemies = Object.keys(strengths).sort((a, b) => strengths[b] - strengths[a]);
    console.log("Training complete. Top enemies by strength:");
    sortedEnemies.slice(0, 10).forEach(enemy => {
      console.log(`  ${enemy}: ${strengths[enemy].toFixed(3)}`);
    });

  } catch (error) {
    console.error("CRITICAL FAILURE in loadAndTrain:", error);
  }
}

// Automatically execute when the script loads
loadAndTrain();