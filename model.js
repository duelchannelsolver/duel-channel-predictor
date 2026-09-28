const fs = require('fs');
const xlsx = require('xlsx');

class Match {
  constructor(teamA, teamB, winner, stormOccurred = false) {
    this.teamA = teamA; // object: enemy_name -> count on side A
    this.teamB = teamB; // object: enemy_name -> count on side B
    this.winner = winner; // "A" or "B"
    this.stormOccurred = stormOccurred; // true/false
  }
}

class EnemyStrengthModel {
  constructor(l2 = 1.0, stormWeight = 0.5) {
    this.l2 = l2;
    this.stormWeight = stormWeight;
    this.enemies = [];
    this.weights = [];
  }

  _featureVector(teamA, teamB) {
    // Unseen enemies contribute 0
    return this.enemies.map(e => {
      const countA = teamA[e] || 0;
      const countB = teamB[e] || 0;
      return countA - countB;
    });
  }

  fit(matches, iters = 2000) {
    const enemySet = new Set();
    matches.forEach(m => {
      Object.keys(m.teamA).forEach(e => enemySet.add(e));
      Object.keys(m.teamB).forEach(e => enemySet.add(e));
    });
    this.enemies = Array.from(enemySet).sort();

    const X = matches.map(m => this._featureVector(m.teamA, m.teamB));
    const y = matches.map(m => (m.winner === "A" ? 1 : 0));
    const sw = matches.map(m => (m.stormOccurred ? this.stormWeight : 1.0));

    this.weights = new Array(this.enemies.length).fill(0);

    // Using the Adam Optimizer to replicate the robust convergence of Python's L-BFGS
    const mAdam = new Array(this.enemies.length).fill(0);
    const vAdam = new Array(this.enemies.length).fill(0);
    const beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lr = 0.1;

    for (let it = 1; it <= iters; it++) {
      const grad = new Array(this.enemies.length).fill(0);

      for (let s = 0; s < matches.length; s++) {
        let z = 0;
        for (let j = 0; j < this.enemies.length; j++) z += this.weights[j] * X[s][j];
        
        const p = 1 / (1 + Math.exp(-z));
        const err = p - y[s];

        for (let j = 0; j < this.enemies.length; j++) {
          // Note: scikit-learn sums the loss over the dataset, it does not divide by N
          grad[j] += sw[s] * err * X[s][j]; 
        }
      }

      for (let j = 0; j < this.enemies.length; j++) {
        // Scikit-learn exact L2 formulation (C = 1 / l2)
        grad[j] += this.l2 * this.weights[j];

        // Adam update rules
        mAdam[j] = beta1 * mAdam[j] + (1 - beta1) * grad[j];
        vAdam[j] = beta2 * vAdam[j] + (1 - beta2) * (grad[j] * grad[j]);
        
        const mHat = mAdam[j] / (1 - Math.pow(beta1, it));
        const vHat = vAdam[j] / (1 - Math.pow(beta2, it));
        
        this.weights[j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
      }
    }
  }

  strengths() {
    const result = {};
    this.enemies.forEach((e, i) => {
      result[e] = this.weights[i];
    });
    return result;
  }

  predictProba(teamA, teamB) {
    const x = this._featureVector(teamA, teamB);
    let z = 0;
    for (let i = 0; i < x.length; i++) {
      z += this.weights[i] * x[i];
    }
    return 1 / (1 + Math.exp(-z));
  }
}

function evaluate(model, matches) {
  let avgProbSum = 0;
  let logLossSum = 0;

  matches.forEach(m => {
    const pA = model.predictProba(m.teamA, m.teamB);
    const winnerProb = m.winner === "A" ? pA : 1 - pA;
    
    avgProbSum += winnerProb;
    // Clip to 1e-9 to prevent log(0) - matching numpy's exact behavior
    const clipped = Math.max(1e-9, Math.min(1, winnerProb)); 
    logLossSum += -Math.log(clipped);
  });

  return {
    avg_winner_prob: avgProbSum / matches.length,
    log_loss: logLossSum / matches.length
  };
}

function parseSide(row, sideLabel) {
  const counter = {};
  // Escapes sideLabel and matches e.g. "Left Enemy 1"
  const pattern = new RegExp(`^${sideLabel} Enemy (.+)$`);

  for (const col of Object.keys(row)) {
    const match = col.match(pattern);
    if (!match) continue;

    const name = row[col];
    if (!name || String(name).trim() === "") continue;

    const countCol = `# of Enemy ${match[1]}`;
    const count = row[countCol];
    
    if (count == null || String(count).trim() === "") continue;

    const nameStr = String(name);
    counter[nameStr] = (counter[nameStr] || 0) + parseInt(count, 10);
  }
  return counter;
}

function loadMatchesFromExcel(path, sheetName = "Training") {
  if (!fs.existsSync(path)) {
    throw new Error(`File not found: ${path}`);
  }

  const workbook = xlsx.readFile(path);
  if (!workbook.Sheets[sheetName]) {
    throw new Error(`Sheet '${sheetName}' not found in ${path}`);
  }

  // Convert the sheet directly to an array of objects
  const rows = xlsx.utils.sheet_to_json(workbook.Sheets[sheetName]);
  const matches = [];

  for (const row of rows) {
    const teamA = parseSide(row, "Left");
    const teamB = parseSide(row, "Right");
    
    const winnerRaw = row["Winner (L/R)"];
    const winner = String(winnerRaw).trim().toUpperCase() === "L" ? "A" : "B";
    
    const stormRaw = row["Storm Start?"];
    const stormOccurred = String(stormRaw || "N").trim().toUpperCase() === "Y";
    
    matches.push(new Match(teamA, teamB, winner, stormOccurred));
  }
  
  return matches;
}

// Equivalent to `if __name__ == "__main__":` in Python
if (require.main === module) {
  try {
    const matches = loadMatchesFromExcel("Duel_Channel_Training_Data.xlsx", "Training");
    console.log(`Loaded ${matches.length} matches.`);

    const model = new EnemyStrengthModel(1.0, 0.5);
    model.fit(matches);

    console.log("\nEnemy strengths:");
    const strengths = model.strengths();
    
    // Sort descending by strength
    const sortedEnemies = Object.keys(strengths).sort((a, b) => strengths[b] - strengths[a]);
    
    sortedEnemies.forEach(enemy => {
      const s = strengths[enemy];
      const sign = s >= 0 ? "+" : "";
      console.log(`  ${enemy}: ${sign}${s.toFixed(3)}`);
    });

    console.log("\nEvaluation on training matches:");
    console.log(evaluate(model, matches));
  } catch (e) {
    console.error("Failed to run script:", e.message);
  }
}