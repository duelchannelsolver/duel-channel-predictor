/**
 * DuelStackModel (stack.js) - Browser & Node.js Universal Version
 * Mounts to window.DuelStackModel, global.DuelStackModel, and module.exports.
 *
 * v5
 *  - Combat simulator: a 2-D battle on a field about 10 tiles square, with the
 *    two teams starting spread along opposite edges. It models about 25 special
 *    mechanics parsed from each enemy's ability text: instant kills, one-shot
 *    attackers, on-death explosions and spawns, revives and second forms, stuns,
 *    Cold and Freeze, burn and damage-over-time, DEF shred, ammo, lifesteal,
 *    thorns, aggression levels, channelled bursts, spins and more. Each matchup
 *    is simulated 9 times and the results are fed to both models as features.
 *  - Ensemble: equal-weight logit average of a "wide" L2-regularised logistic
 *    regression (unit counts + team stats + simulator) and gradient-boosted
 *    trees. Options: { useNN: true } adds a neural net as a third model,
 *    { stacking: true } fits the ensemble weights instead of using equal ones.
 *  - Storm is not a model input (it is only known after the match). The `storm`
 *    argument of predict* is accepted and ignored; storm-flagged training rows
 *    are down-weighted (stormWeight).
 *  - predictDetailed().weights is ordered [wide, nn, xgb]; `nn` is null unless
 *    the neural net is enabled.
 *  - A saved model (toJSON / fromJSON, see train.js) only matches the stack.js
 *    that trained it. Whenever this file changes, re-run train.js. fromJSON
 *    sets `outdated = true` on a model saved by a different simulator version.
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

  // FNV-1a string hash -> 32-bit int, used to derive deterministic simulator seeds
  function hashString(s) {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h | 0;
  }

  const sigmoid = (z) => (z > 30 ? 1 : z < -30 ? 0 : 1 / (1 + Math.exp(-z)));
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
    stun: (t) => {
      const clean = t
        .replace(/immune to stun( and freeze)?/g, '')
        .replace(/immune to cold/g, '');
      if (/\bstuns?\b/.test(clean)) return 1;
      if (/inflict\w*[^.]*\bcold\b/.test(clean)) return 0.5;
      return 0;
    },
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
    suicide:   (t) => /loses (300|50)\b/.test(t),
    token:     (t) => /\bspawns\b(?!\s+by\b)/.test(t),
  };
  const FLAG_NAMES = Object.keys(ABILITY_FLAGS);
  const WORD_NUM = { two: 2, three: 3, four: 4 };

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
      flags: FLAG_NAMES.map((f) => {
        const v = ABILITY_FLAGS[f](text);
        return typeof v === 'number' ? v : (v ? 1 : 0);
      }),
      sim: parseSim(e, text, dmgType, atk),
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

  // Antisymmetric (A minus B) engineered features
  function pairDiffFeatures(unitsA, unitsB) {
    const sA = teamStats(unitsA), sB = teamStats(unitsB);
    const fA = teamFeatures(sA), fB = teamFeatures(sB);
    const dpsAB = Math.max(effectiveDps(unitsA, sB.avgDef, sB.avgRes), 1e-3);
    const dpsBA = Math.max(effectiveDps(unitsB, sA.avgDef, sA.avgRes), 1e-3);
    const logTtkA = Math.log(Math.max(sB.hp, 1) / dpsAB);
    const logTtkB = Math.log(Math.max(sA.hp, 1) / dpsBA);
    return [...fA.map((v, i) => v - fB[i]), clamp(logTtkB - logTtkA, -10, 10)];
  }

  // Per-team (not differenced) features for the neural net, plus time-to-kill terms
  function buildNnStatFeatures(unitsA, unitsB) {
    const sA = teamStats(unitsA), sB = teamStats(unitsB);
    const dpsAB = Math.max(effectiveDps(unitsA, sB.avgDef, sB.avgRes), 1e-3);
    const dpsBA = Math.max(effectiveDps(unitsB, sA.avgDef, sA.avgRes), 1e-3);
    const logTtkA = Math.log(Math.max(sB.hp, 1) / dpsAB);
    const logTtkB = Math.log(Math.max(sA.hp, 1) / dpsBA);
    return [
      ...teamFeatures(sA), ...teamFeatures(sB),
      clamp(logTtkA, -10, 15), clamp(logTtkB, -10, 15), clamp(logTtkB - logTtkA, -10, 10),
    ];
  }

  // ---------------------------------------------------------------------------
  // Neural net (small ReLU MLP with a sigmoid output)
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

    // Accumulates the gradient of (weight * log loss) for one sample into gW / gb
    backward(x, y, weight, gW, gb) {
      const cache = {};
      const p = this.forward(x, cache);
      let dz = new Float64Array([(p - y) * weight]);
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
  // Combat simulator
  // ---------------------------------------------------------------------------
  // Reads the simulator's view of a unit out of its stat block and ability text.
  function parseSim(e, text, dmgType, atk) {
    const numWord = (w) => WORD_NUM[w] || parseInt(w, 10) || 1;
    const s = {
      ms: num(e['Movement Speed'], 0.5),
      ai: num(e['Attack Interval'], 3),
      range: e['Attack Range'] === 'Global' ? 10 : num(e['Attack Range'], 0.8),
      phys: dmgType.startsWith('arts') ? 0 : 1,
      targets: 1, splash: 0, regen: 0,
    };
    let m;
    if ((m = /up to (two|three|four|\d+) targets/.exec(text))) s.targets = numWord(m[1]);
    if (/splash|adjacent 4|surrounding 8/.test(text)) s.splash = 1.0;
    if ((m = /targets within range with (\d+)% atk/.exec(text))) s.atkMult = parseFloat(m[1]) / 100;
    if ((m = /regenerates (\d+) hp/.exec(text))) s.regen += parseFloat(m[1]);
    if ((m = /loses (\d+) hp every second/.exec(text))) s.regen -= parseFloat(m[1]);

    if ((m = /has \+(\d+) aggression level/.exec(text))) s.aggro = parseFloat(m[1]);

    // One-shot / instant-kill attackers
    if (/launching itself/.test(text)) s.suicide = true;
    if (/devours the target when attacking for the first time/.test(text)) s.devourFirst = true;
    if ((m = /devours an enemy within a range of ([\d.]+)[^]*?digesting for (\d+) seconds/.exec(text))) {
      s.devourer = true; s.range = parseFloat(m[1]); s.ai = parseFloat(m[2]);
    }

    // On-death effects
    if ((m = /(?:explodes[^.]*when defeated|when defeated[^.]*burst)[^.]*?(\d+)% atk (physical|arts) damage[^.]*?radius of ([\d.]+)/.exec(text))) {
      s.deathExplode = { dmg: (parseFloat(m[1]) / 100) * atk, phys: m[2] === 'physical' ? 1 : 0, radius: parseFloat(m[3]), stun: 0, cold: 0 };
      const ms = /radius of [\d.]+ and stuns them for (\d+)/.exec(text);
      if (ms) s.deathExplode.stun = parseFloat(ms[1]);
      const mc = /inflicts cold for (\d+) seconds to nearby/.exec(text);
      if (mc) s.deathExplode.cold = parseFloat(mc[1]);
    }
    if (/when defeated|explodes/.test(text) && (m = /spawns an? ([a-z\- ]+?) on the spot/.exec(text))) s.spawnName = m[1];
    if (/when defeated, abducts an enemy/.test(text)) s.abduct = true;
    if ((m = /revives (?:with (\d+)% hp )?over (\d+) seconds/.exec(text))) {
      s.revive = { frac: m[1] ? parseFloat(m[1]) / 100 : 1, delay: parseFloat(m[2]), inv: /invulnerable for (\d+) seconds/.test(text) ? 10 : 0 };
    }

    // Crowd control
    if ((m = /after (two|three|four|\d+) attacks, the next attack[^.]*stuns (?:the target|them) for (\d+) seconds/.exec(text))) {
      s.stunEvery = { n: numWord(m[1]), dur: parseFloat(m[2]) };
    }
    if ((m = /after (two|three|four|\d+) attacks, the next attack inflicts cold for (\d+) seconds/.exec(text))) {
      s.coldEvery = { n: numWord(m[1]), dur: parseFloat(m[2]) };
    }
    if (/immune to stun/.test(text)) s.stunImmune = true;
    if (/immune to (stun and )?freeze/.test(text)) s.freezeImmune = true;
    if (/status resistance/.test(text)) s.statusRes = true;

    // Damage-over-time, thorns, debuffs
    if ((m = /when taking damage, deals (\d+) arts damage to the source/.exec(text))) s.thorns = parseFloat(m[1]);
    if ((m = /inflict (\d+)% atk burn/.exec(text))) s.burn = (parseFloat(m[1]) / 100) * atk;
    if ((m = /for (\d+) seconds that deals (\d+) arts damage every (second|[\d.]+ seconds)/.exec(text))) {
      const per = m[3] === 'second' ? 1 : parseFloat(m[3]);
      s.dot = { dur: parseFloat(m[1]), dps: parseFloat(m[2]) / (per || 1) };
    }
    if ((m = /attacks inflict -(\d+) def on the target/.exec(text))) s.shred = parseFloat(m[1]);
    if ((m = /loses (\d+) def and (\d+) res each time/.exec(text))) s.selfShred = { def: parseFloat(m[1]), res: parseFloat(m[2]) };

    // Attack modifiers
    if ((m = /initially has (\d+) broken blades[^.]*\+(\d+)% atk/.exec(text))) s.ammo = { n: parseFloat(m[1]), mult: 1 + parseFloat(m[2]) / 100 };
    if ((m = /has (\d+)\/\d+ (?:ammo|copper coin)[^]*?with (\d+)% atk/.exec(text))) s.ammo = { n: parseFloat(m[1]), mult: parseFloat(m[2]) / 100 };
    if ((m = /restores hp equal to (\d+)% of the damage dealt/.exec(text))) s.lifesteal = parseFloat(m[1]) / 100;
    if (/starts in the imprisoned state/.test(text)) {
      s.imprisoned = { ignoreDef: /ignoring 60% def/.test(text) ? 0.6 : 0, toArts: /attacks dealing arts damage/.test(text) };
    }
    if ((m = /has \+(\d+)% atk (?:and takes -(\d+)%[^.]*)?while hp is below 50%/.exec(text))) {
      s.lowHp = { atk: 1 + parseFloat(m[1]) / 100, taken: m[2] ? 1 - parseFloat(m[2]) / 100 : 1 };
    }
    if ((m = /gains (\d+)% atk and (\d+) aspd each time an ally is defeated/.exec(text))) s.avenger = { atk: parseFloat(m[1]) / 100, aspd: parseFloat(m[2]) };
    if ((m = /from a range of ([\d.]+), dealing (\d+)% atk (arts|physical) damage[^]*?once per round/.exec(text))) {
      s.opener = { range: parseFloat(m[1]), mult: parseFloat(m[2]) / 100, phys: m[3] === 'physical' ? 1 : 0 };
    }
    if (/charges forth/.test(text)) s.charger = true;
    if ((m = /while hp is below 50%: takes -(\d+)% physical and arts damage/.exec(text))) s.lowHp = { atk: 1, taken: 1 - parseFloat(m[1]) / 100 };
    if ((m = /while moving, deals (\d+)% atk physical damage to enemies/.exec(text))) s.opener = { range: s.range, mult: parseFloat(m[1]) / 100, phys: 1 };
    if ((m = /becomes enraged when the ([a-z ]+) is defeated, gaining (\d+)% atk, \d+% mspd, and (\d+) aspd/.exec(text))) {
      s.enrage = { partner: m[1], atk: 1 + parseFloat(m[2]) / 100, aspd: parseFloat(m[3]) };
    }
    // Periodic skills
    if ((m = /channels for (\d+) seconds before unleashing a burst[^]*?deals (\d+)% atk arts damage[^]*?cooldown: \((\d+)\) (\d+) sec/.exec(text))) {
      s.skill = { first: parseFloat(m[3]), cd: parseFloat(m[4]), channel: parseFloat(m[1]), mult: parseFloat(m[2]) / 100, phys: 0, radius: 1.5, stun: 0, range: s.range };
    }
    if ((m = /range of ([\d.]+), dealing (\d+)% atk physical damage and stunning the target for (\d+) seconds\. cooldown: (\d+) sec/.exec(text))) {
      s.skill = { first: parseFloat(m[4]), cd: parseFloat(m[4]), mult: parseFloat(m[2]) / 100, phys: 1, radius: 0, stun: parseFloat(m[3]), range: parseFloat(m[1]) };
    }
    // Spin: after a recharge the unit spins for a long time, hitting everything
    // in contact once a second instead of attacking normally
    if ((m = /spins for (\d+) seconds[^]*?radius of ([\d.]+) takes (\d+)% atk physical damage every second[^]*?cooldown: (\d+) sec/.exec(text))) {
      s.aura = { dur: parseFloat(m[1]), radius: parseFloat(m[2]), mult: parseFloat(m[3]) / 100, cd: parseFloat(m[4]) };
    }
    if (/has status resistance and \+100 aspd/.test(text)) s.ai = s.ai / 2;

    // Defensive
    if ((m = /shield for (\d+) seconds after spawning which grants \+(\d+) def/.exec(text))) s.tempDef = { dur: parseFloat(m[1]), def: parseFloat(m[2]) };
    if ((m = /gains 100% physical dodge for (\d+) seconds/.exec(text))) s.dodge = parseFloat(m[1]);
    if ((m = /when attacked for the first time, gains[^.]*?(\d+) aspd for (\d+) seconds/.exec(text))) s.rage = { aspd: parseFloat(m[1]), dur: parseFloat(m[2]) };
    if (/splitting into two/.test(text)) s.split = true;
    if ((m = /^starts in the first form: immune to stun and takes -(\d+)%/.exec(text))) s.taken = 1 - parseFloat(m[1]) / 100;
    if (/while present, the chivalrous trio takes -40%/.test(text)) s.trioAura = true;

    // Two-form unit described only in prose
    if (/starts in the jailer form/.test(text)) {
      s.phys = 0; s.range = 3; s.resBonus = 70;
      s.form2 = { phys: 1, range: 0.8, atk: 700, def: 1000, ai: -1.5, ms: 0.9 };
    }
    if (/enters the second form/.test(text)) s.form2 = { atkMult: 1.5, ai: -1.5, hits: 2 };
    return s;
  }

  // 1-D battle. Teams start `arenaLength` tiles apart, each unit walks to its
  // nearest enemy and attacks when in range. Physical damage = max(ATK - DEF,
  // 5% ATK); Arts = max(ATK * (1 - RES%), 5% ATK). With mech=true the special
  // abilities parsed above are also simulated. Returns (fraction of starting HP
  // left on A) - (same for B).
  function simulateBattle(unitsA, unitsB, opt, seed, mech) {
    const rng = mulberry32(seed);
    const L = opt.arenaLength, dt = opt.dt, spread = opt.spawnSpread;
    const off = opt.disabled || {};
    const E = [];
    const tot = [0, 0];

    const W = opt.arenaWidth || 0;
    const dist = (a, bx, by) => Math.hypot(a.pos - bx, a.y - by);
    const spawn = (u, side, pos, y, initial) => {
      const s = u.sim;
      const e = {
        u, s, side, pos, y, hp: u.hp, hp0: u.hp,
        atk: u.atk, def: u.def, res: u.res + (mech && s.resBonus ? s.resBonus : 0),
        ai: s.ai, ms: s.ms, range: s.range, phys: s.phys, hits: 1,
        cd: rng() * s.ai * 0.5, pend: 0, kill: false,
        stunT: 0, coldT: 0, frozenT: 0, invT: 0, dodgeT: 0, reviveT: 0, revived: false,
        nAtk: 0, ammo: 0, ammoMult: 1, imprisoned: false, ignoreDef: 0,
        burn: 0, dotT: 0, dotDps: 0, shieldT: 0, rageT: 0, raged: false,
        opener: null, devoured: false, splitDone: false, dodged: false,
        avenge: 0, age: 0, trio: false, enraged: false, skillT: s.skill ? s.skill.first : 0,
        spinT: 0, spinCd: s.aura ? s.aura.cd : 0, spinTick: 0,
        chanT: 0, chanX: 0, chanY: 0,
      };
      if (mech) {
        if (s.ammo && !off.ammo) { e.ammo = s.ammo.n; e.ammoMult = s.ammo.mult; }
        if (s.imprisoned && !off.imprisoned) { e.imprisoned = true; e.ai = s.ai * 2; }
        if (s.tempDef && !off.tempDef) { e.shieldT = s.tempDef.dur; e.def += s.tempDef.def; }
        if (s.opener && !off.opener) e.opener = s.opener;
        if (s.devourer) e.cd = 0;
      }
      if (initial) tot[side] += u.hp;
      E.push(e);
      return e;
    };

    for (const { u, n } of unitsA) {
      const c = Math.max(0, Math.round(Number(n) || 0));
      for (let i = 0; i < c; i++) spawn(u, 0, -rng() * spread, rng() * W, true);
    }
    for (const { u, n } of unitsB) {
      const c = Math.max(0, Math.round(Number(n) || 0));
      for (let i = 0; i < c; i++) spawn(u, 1, L + rng() * spread, rng() * W, true);
    }
    if (tot[0] <= 0 && tot[1] <= 0) return 0;
    if (tot[0] <= 0) return -1;
    if (tot[1] <= 0) return 1;

    const TRIO = { 'Jade Twin-Swords': 1, 'Crimson Crescent Blade': 1, 'Ash Spear': 1 };
    const isUp = (e) => e.hp > 0 && e.reviveT <= 0;
    const curAtk = (e) => {
      let a = e.atk;
      if (mech) {
        if (e.s.atkMult) a *= e.s.atkMult;
        if (e.enraged) a *= e.s.enrage.atk;
        if (e.avenge) a *= 1 + e.s.avenger.atk * e.avenge;
        if (e.s.lowHp && !off.lowHp && e.hp < 0.5 * e.hp0) a *= e.s.lowHp.atk;
      }
      return a;
    };
    const applyStun = (t, dur) => {
      if (t.s.stunImmune) return;
      t.stunT = Math.max(t.stunT, t.s.statusRes ? dur / 2 : dur);
    };
    // Cold slows attacks (-30 ASPD). Cold applied to a unit that is already Cold
    // freezes it for the duration of that second Cold; Cold applied to a Frozen
    // unit refreshes the freeze. Cold itself is never refreshed.
    const applyCold = (t, dur) => {
      if (t.s.statusRes) dur /= 2;
      if (t.s.freezeImmune) { if (t.coldT <= 0) t.coldT = dur; return; }
      if (t.frozenT > 0 || t.coldT > 0) {
        t.frozenT = dur; t.coldT = 0;
        t.stunT = Math.max(t.stunT, dur); // a frozen unit cannot move or attack
      } else {
        t.coldT = dur;
      }
    };
    // Queue damage from src (may be null) onto t; returns the amount queued.
    const strike = (src, t, raw, phys, ignoreDef) => {
      if (t.invT > 0) return 0;
      if (phys && t.dodgeT > 0) return 0;
      let d = phys
        ? Math.max(raw - t.def * (1 - (ignoreDef || 0)), 0.05 * raw)
        : Math.max(raw * (1 - Math.min(t.res, 100) / 100), 0.05 * raw);
      if (mech) {
        if (t.s.taken) d *= t.s.taken;
        if (t.trio) d *= 0.6;
        if (t.s.lowHp && !off.lowHp && t.hp < 0.5 * t.hp0) d *= t.s.lowHp.taken;
        if (t.s.selfShred && !off.selfShred) {
          t.def = Math.max(0, t.def - t.s.selfShred.def); t.res = Math.max(0, t.res - t.s.selfShred.res);
        }
        if (src && t.s.thorns && !off.thorns) src.pend += Math.max(t.s.thorns * (1 - Math.min(src.res, 100) / 100), 0.05 * t.s.thorns);
        if (t.s.rage && !off.rage && !t.raged) { t.raged = true; t.rageT = t.s.rage.dur; }
      }
      t.pend += d;
      return d;
    };

    const maxSteps = Math.ceil(opt.maxTime / dt);
    for (let step = 0; step < maxSteps; step++) {
      // --- who is standing --------------------------------------------------
      let up0 = 0, up1 = 0, pres0 = 0, pres1 = 0, jade0 = false, jade1 = false;
      for (const e of E) {
        if (e.reviveT > 0) {
          e.reviveT -= dt;
          if (e.side === 0) pres0++; else pres1++;
          continue;
        }
        if (e.hp > 0) {
          if (e.side === 0) { up0++; pres0++; } else { up1++; pres1++; }
          if (mech && e.s.trioAura) { if (e.side === 0) jade0 = true; else jade1 = true; }
        }
      }
      if (pres0 === 0 || pres1 === 0) break;
      const N = E.length;

      // --- timers, regeneration, damage over time ---------------------------
      for (let i = 0; i < N; i++) {
        const e = E[i];
        if (!isUp(e)) continue;
        e.age += dt;
        e.trio = mech && TRIO[e.u.name] === 1 && (e.side === 0 ? jade0 : jade1);
        if (e.invT > 0) e.invT -= dt;
        if (e.dodgeT > 0) e.dodgeT -= dt;
        if (e.coldT > 0) e.coldT -= dt;
        if (e.frozenT > 0) e.frozenT -= dt;
        if (e.rageT > 0) e.rageT -= dt;
        if (e.shieldT > 0) { e.shieldT -= dt; if (e.shieldT <= 0) e.def = Math.max(0, e.def - e.s.tempDef.def); }
        if (mech && e.s.regen) e.pend -= e.s.regen * dt;
        if (e.dotT > 0) { e.dotT -= dt; e.pend += Math.min(e.dotDps * dt * (1 - Math.min(e.res, 100) / 100), Math.max(e.hp - 1, 0)); }
      }

      // --- targeting, attacks, movement --------------------------------------
      for (let i = 0; i < N; i++) {
        const e = E[i];
        if (!isUp(e)) continue;

        // Spin (walks normally until the recharge ends, then spins, still walking
        // toward enemies). A stun or freeze pauses the spin's damage; it resumes
        // as soon as the stun ends, until the spin's time is up.
        let spinning = false;
        if (mech && e.s.aura && !off.aura) {
          const a = e.s.aura;
          if (e.spinT > 0) {
            spinning = true;
            e.spinT -= dt;
            if (e.stunT <= 0) e.spinTick += dt;
            if (e.stunT <= 0 && e.spinTick >= 1 - 1e-9) {
              e.spinTick -= 1;
              const raw = curAtk(e) * a.mult;
              for (let j = 0; j < N; j++) {
                const o = E[j];
                if (o.side !== e.side && isUp(o) && dist(e, o.pos, o.y) <= a.radius) strike(e, o, raw, 1, 0);
              }
            }
            if (e.spinT <= 0) e.spinCd = a.cd;
          } else {
            e.spinCd -= dt;
            if (e.spinCd <= 0) { e.spinT = a.dur; e.spinTick = 0; spinning = true; }
          }
        }

        if (e.stunT > 0) { e.stunT -= dt; continue; }
        // Target: the nearest enemy, except that an enemy with a higher aggression
        // level that is already within attack range is attacked first (taunt).
        let tg = null, bd = Infinity, tt = null, td = Infinity, ta = 0;
        const useAggro = mech && !off.aggro;
        for (let j = 0; j < N; j++) {
          const o = E[j];
          if (o.side === e.side || !isUp(o)) continue;
          const dd = dist(e, o.pos, o.y);
          if (dd < bd) { bd = dd; tg = o; }
          if (useAggro && o.s.aggro && dd <= e.range && (o.s.aggro > ta || (o.s.aggro === ta && dd < td))) {
            ta = o.s.aggro; td = dd; tt = o;
          }
        }
        if (!tg) continue;
        if (tt) { tg = tt; bd = td; }
        const s = e.s;
        let rate = 1;
        if (e.coldT > 0) rate *= 0.7;
        if (e.rageT > 0) rate *= 1 + s.rage.aspd / 100;
        if (e.avenge) rate *= 1 + (s.avenger.aspd * e.avenge) / 100;
        if (e.enraged) rate *= 1 + s.enrage.aspd / 100;
        e.cd -= dt * rate;

        // Periodic skill (channelled burst, boulder)
        if (mech && s.skill && !off.skill) {
          // A channelled burst lands where its target stood when the channel
          // began, so enemies that have walked away since are not hit.
          if (e.chanT > 0) {
            e.chanT -= dt;
            if (e.chanT <= 0) {
              const k = s.skill, raw = curAtk(e) * k.mult;
              for (let j = 0; j < N; j++) {
                const o = E[j];
                if (o.side !== e.side && isUp(o) && dist(o, e.chanX, e.chanY) <= k.radius) strike(e, o, raw, k.phys, 0);
              }
              e.skillT = k.cd;
            }
            continue; // busy channelling: no moving or attacking
          }
          e.skillT -= dt;
          if (e.skillT <= 0) {
            const k = s.skill;
            let st = null, sd = Infinity;
            for (let j = 0; j < N; j++) {
              const o = E[j];
              if (o.side === e.side || !isUp(o)) continue;
              const dd = dist(e, o.pos, o.y) + (k.stun && o.stunT > 0 ? 100 : 0);
              if (dd < sd) { sd = dd; st = o; }
            }
            if (st && dist(e, st.pos, st.y) <= k.range) {
              const raw = curAtk(e) * k.mult, cx = st.pos, cy = st.y;
              if (k.channel) {
                e.chanT = k.channel; e.chanX = cx; e.chanY = cy;
                continue;
              } else if (k.radius > 0) {
                for (let j = 0; j < N; j++) {
                  const o = E[j];
                  if (o.side !== e.side && isUp(o) && dist(o, cx, cy) <= k.radius) strike(e, o, raw, k.phys, 0);
                }
              } else {
                strike(e, st, raw, k.phys, 0);
                if (k.stun) applyStun(st, k.stun);
              }
              e.skillT = k.cd;
            }
          }
        }
        // One-off ranged opener (snowball, RPG)
        if (e.opener && bd <= e.opener.range) {
          const raw = curAtk(e) * e.opener.mult, cx = tg.pos, cy = tg.y;
          let hits = 0;
          for (let j = 0; j < N && hits < 5; j++) {
            const o = E[j];
            if (o.side === e.side || !isUp(o) || dist(o, cx, cy) > 1) continue;
            strike(e, o, raw, e.opener.phys, 0); hits++;
          }
          e.opener = null;
        }

        if (bd > e.range) {
          let sp = e.ms;
          if (mech && s.charger && !off.charger && e.age > 5 && e.nAtk === 0) sp *= Math.min(11, 1 + (e.age - 5) * 3);
          const mv = Math.min(sp * dt * opt.moveScale, bd);
          if (bd > 1e-9) { e.pos += ((tg.pos - e.pos) / bd) * mv; e.y += ((tg.y - e.y) / bd) * mv; }
          continue;
        }
        if (spinning) continue; // disarmed while spinning
        if (e.cd > 0) continue;

        // --- attack ---------------------------------------------------------
        if (mech && s.devourer && !off.devour) { tg.kill = true; e.cd = e.ai; e.nAtk++; continue; }
        if (e.atk <= 0) continue;
        e.cd = e.ai;
        if (mech && s.devourFirst && !off.devour && !e.devoured) {
          e.devoured = true; tg.kill = true; e.def += 700; e.ms *= 0.7; e.nAtk++;
          continue;
        }
        let raw = curAtk(e);
        if (e.ammo > 0) { raw *= e.ammoMult; e.ammo--; }
        if (mech && s.charger && !off.charger && e.nAtk === 0 && e.age > 5) {
          const mspd = e.ms * Math.min(11, 1 + (e.age - 5) * 3);
          strike(e, tg, 1300 * mspd, 1, 0);
        }
        e.nAtk++;
        let stun = 0, cold = 0;
        if (mech && !off.stun) {
          if (s.stunEvery && e.nAtk % (s.stunEvery.n + 1) === 0) stun = s.stunEvery.dur;
          if (s.coldEvery && e.nAtk % (s.coldEvery.n + 1) === 0) cold = s.coldEvery.dur;
        }
        const splash = mech ? s.splash : 0;
        const k = mech ? s.targets : 1;
        let dealt = 0;
        const land = (o) => {
          for (let h = 0; h < e.hits; h++) dealt += strike(e, o, raw, e.phys, e.ignoreDef);
          if (!mech) return;
          if (stun) applyStun(o, stun);
          if (cold) applyCold(o, cold);
          if (s.shred && !off.shred) o.def = Math.max(0, o.def - s.shred);
          if (s.burn && !off.burn) { o.burn += s.burn; if (o.burn >= 1000) { o.burn = 0; o.pend += 1200; } }
          if (s.dot && !off.dot) { o.dotT = s.dot.dur; o.dotDps = Math.max(o.dotDps, s.dot.dps); }
        };
        if (splash > 0) {
          const cx = tg.pos, cy = tg.y;
          let hits = 0;
          for (let j = 0; j < N && hits < 5; j++) {
            const o = E[j];
            if (o.side === e.side || !isUp(o) || dist(o, cx, cy) > splash) continue;
            land(o); hits++;
          }
        } else if (k > 1) {
          const cand = [];
          for (let j = 0; j < N; j++) {
            const o = E[j];
            if (o.side === e.side || !isUp(o)) continue;
            const dd = dist(e, o.pos, o.y);
            if (dd <= e.range) cand.push({ o, dd });
          }
          cand.sort((a, b) => a.dd - b.dd);
          for (let c = 0; c < Math.min(k, cand.length); c++) land(cand[c].o);
        } else {
          land(tg);
        }
        if (mech && s.lifesteal && !off.lifesteal) e.pend -= s.lifesteal * dealt;
        if (mech && s.suicide && !off.suicide) e.kill = true;
        if (e.imprisoned && e.nAtk >= 4) {
          e.imprisoned = false; e.ai = s.ai; e.atk *= 1.5; e.ignoreDef = s.imprisoned.ignoreDef;
          if (s.imprisoned.toArts) e.phys = 0;
        }
      }

      // --- resolve damage and deaths (explosions can chain) -------------------
      let guard = 0, again = true;
      while (again && guard++ < 20) {
        again = false;
        const M = E.length;
        for (let i = 0; i < M; i++) {
          const e = E[i];
          if (!isUp(e)) { e.pend = 0; e.kill = false; continue; }
          if (e.kill && e.invT <= 0) e.hp = 0;
          else if (e.pend !== 0) e.hp = Math.min(e.hp - e.pend, e.hp0);
          e.pend = 0; e.kill = false;
          const s = e.s;
          if (mech && e.hp > 0 && e.hp < 0.5 * e.hp0) {
            if (s.dodge && !off.dodge && !e.dodged) { e.dodged = true; e.dodgeT = s.dodge; }
            if (s.split && !off.split && !e.splitDone) {
              e.splitDone = true;
              const c = spawn(e.u, e.side, e.pos, e.y, false);
              c.hp = e.hp; c.splitDone = true; c.cd = e.ai;
            }
          }
          if (e.hp > 0) continue;

          // death
          e.hp = 0;
          if (!mech) continue;
          if (s.deathExplode && !off.deathExplode) {
            const x = s.deathExplode;
            for (let j = 0; j < E.length; j++) {
              const o = E[j];
              if (o.side === e.side || !isUp(o) || dist(o, e.pos, e.y) > x.radius) continue;
              strike(null, o, x.dmg, x.phys, 0);
              if (x.stun) applyStun(o, x.stun);
              if (x.cold) applyCold(o, x.cold);
              again = true;
            }
          }
          if (s.abduct && !off.abduct) {
            let vt = null, vd = Infinity;
            for (let j = 0; j < E.length; j++) {
              const o = E[j];
              if (o.side === e.side || !isUp(o) || o.kill) continue;
              const dd = dist(o, e.pos, e.y);
              if (dd < vd) { vd = dd; vt = o; }
            }
            if (vt) { vt.kill = true; again = true; }
          }
          if (s.spawnUnits && !off.spawn) for (const su of s.spawnUnits) spawn(su, e.side, e.pos, e.y, false);
          if (s.revive && !off.revive && !e.revived) {
            e.revived = true; e.reviveT = s.revive.delay; e.invT = s.revive.inv;
            e.hp = e.hp0 * s.revive.frac;
            e.stunT = 0; e.coldT = 0; e.frozenT = 0; e.dotT = 0; e.burn = 0;
            const f = s.form2;
            if (f) {
              if (f.phys !== undefined) e.phys = f.phys;
              if (f.range !== undefined) e.range = f.range;
              if (f.atk) e.atk += f.atk;
              if (f.atkMult) e.atk *= f.atkMult;
              if (f.def) e.def += f.def;
              if (f.ai) e.ai = Math.max(0.5, e.ai + f.ai);
              if (f.ms) e.ms = f.ms;
              if (f.hits) e.hits = f.hits;
              if (s.resBonus) e.res -= s.resBonus;
            }
          }
          for (let j = 0; j < E.length; j++) {
            const o = E[j];
            if (o.side === e.side && o !== e && o.s.enrage && !off.enrage && norm(o.s.enrage.partner) === norm(e.u.name)) o.enraged = true;
            if (o.side === e.side && o !== e && o.s.avenger && !off.avenger && isUp(o)) o.avenge = Math.min(10, o.avenge + 1);
          }
        }
      }
    }

    const rem = [0, 0];
    for (const e of E) if (e.hp > 0) rem[e.side] += e.hp;
    return Math.min(1, rem[0] / tot[0]) - Math.min(1, rem[1] / tot[1]);
  }

  // Bump whenever the simulator or the features change: saved models from an
  // older version would silently give wrong predictions with the new code.
  const SIM_VERSION = 5;

  const SIM_DIM = 4; // [mean margin, mean win sign, signed sqrt margin, margin without mechanics]

  // Simulator results are shared between model instances that use the same
  // enemies array (e.g. the folds of a cross-validation run).
  const SIM_CACHES = new WeakMap();

  const teamKey = (team) =>
    Object.keys(team || {}).sort().map((k) => k + ':' + team[k]).join('|');

  // ---------------------------------------------------------------------------
  // Generic weighted, L2-regularised logistic regression (no intercept)
  // ---------------------------------------------------------------------------
  // rows: array of Float64Array. Minimises  sum_s sw_s * logloss_s + (l2 / 2) * |w|^2
  function fitLogistic(rows, y, sw, l2, iters) {
    const n = rows.length, d = n ? rows[0].length : 0;
    const w = new Float64Array(d), mW = new Float64Array(d), vW = new Float64Array(d);
    const grad = new Float64Array(d);
    for (let it = 1; it <= iters; it++) {
      grad.fill(0);
      for (let s = 0; s < n; s++) {
        const x = rows[s];
        let z = 0;
        for (let j = 0; j < d; j++) z += w[j] * x[j];
        const err = sw[s] * (sigmoid(z) - y[s]);
        for (let j = 0; j < d; j++) grad[j] += err * x[j];
      }
      const c1 = 1 - Math.pow(0.9, it), c2 = 1 - Math.pow(0.999, it);
      const step = 0.1 / Math.sqrt(1 + it / 25); // decaying Adam step for a stable finish
      for (let j = 0; j < d; j++) {
        const g = grad[j] + l2 * w[j];
        mW[j] = 0.9 * mW[j] + 0.1 * g;
        vW[j] = 0.999 * vW[j] + 0.001 * g * g;
        w[j] -= (step * (mW[j] / c1)) / (Math.sqrt(vW[j] / c2) + 1e-8);
      }
    }
    return w;
  }

  function dot(w, x) {
    let z = 0;
    for (let j = 0; j < w.length; j++) z += w[j] * x[j];
    return z;
  }

  // ---------------------------------------------------------------------------
  // Predictors rebuilt from plain (JSON-serialisable) parameters
  // ---------------------------------------------------------------------------
  function wideModel(w) {
    return { w, predictRow: (x) => sigmoid(dot(w, x)) };
  }

  function xgbModel(trees) {
    // sign = +1 scores x, sign = -1 scores the mirrored row (-x)
    const margin = (x, sign) => {
      let m = 0;
      for (let i = 0; i < trees.length; i++) {
        let node = trees[i];
        while (node.leaf === undefined) node = sign * x[node.f] <= node.t ? node.l : node.r;
        m += node.leaf;
      }
      return m;
    };
    return {
      trees,
      predictRow: (x) => 0.5 * (sigmoid(margin(x, 1)) + (1 - sigmoid(margin(x, -1)))),
    };
  }

  // state: { sizes, W, b, mean, std, clip }
  function nnMember(state) {
    const net = new BrowserMLP(state.sizes, () => 0.5);
    net.W = state.W.map((w) => Float64Array.from(w));
    net.b = state.b.map((b) => Float64Array.from(b));
    const mean = state.mean, std = state.std, clip = state.clip, dim = mean.length;
    const normVec = (v) => {
      const out = new Float64Array(dim);
      for (let j = 0; j < dim; j++) out[j] = clamp((v[j] - mean[j]) / std[j], -clip, clip);
      return out;
    };
    // Average the two orientations so swapping the teams gives exactly 1 - p
    return {
      state,
      predictPair: (f, w) => 0.5 * (net.predict(normVec(f)) + (1 - net.predict(normVec(w)))),
    };
  }

  function nnModel(members) {
    return {
      members,
      predictPair: (f, w) => {
        let sum = 0;
        for (const mm of members) sum += mm.predictPair(f, w);
        return sum / members.length;
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Master Stacking Classifier Class
  // ---------------------------------------------------------------------------
  class DuelStackModel {
    constructor(options = {}) {
      // Gradient-boosted trees
      this.xgbTrees = options.xgbTrees ?? 100;
      this.xgbDepth = options.xgbDepth ?? 3;
      this.xgbLr = options.xgbLr ?? 0.08;
      this.xgbLambda = options.xgbLambda ?? 1.5;
      this.xgbGamma = options.xgbGamma ?? 0.1;
      this.xgbMinChildWeight = options.xgbMinChildWeight ?? 1.0;
      this.xgbSubsample = options.xgbSubsample ?? 0.85;
      this.xgbColsample = options.xgbColsample ?? 0.85;

      // Training weight of storm-flagged matches (all base models). 0 drops them.
      this.stormWeight = options.stormWeight ?? options.xgbStormWeight ?? 0.75;

      // Neural net: small, short training, L2 weight decay, clipped inputs, seed-averaged
      this.nnEpochs = options.nnEpochs ?? 30;
      this.nnHidden = options.nnHidden ?? [16, 8];
      this.nnLr = options.nnLr ?? 0.003;
      this.nnWd = options.nnWd ?? 0.01;
      this.nnClip = options.nnClip ?? 5;
      this.nnEnsemble = options.nnEnsemble ?? 3;

      // Wide logistic regression on all standardised features
      this.wideL2 = options.wideL2 ?? 16.7;
      this.wideIters = options.wideIters ?? 600;

      // Simulator
      // 9 simulated battles per matchup: fewer makes the simulator features noisy
      this.simSeeds = Math.max(1, (options.simSeeds ?? 9) | 0);
      this.simOpt = {
        arenaLength: options.simArenaLength ?? 10,
        arenaWidth: options.simArenaWidth ?? 10, // teams start spread over the full height of the field
        spawnSpread: options.simSpawnSpread ?? 3,
        dt: options.simDt ?? 0.2,
        maxTime: options.simMaxTime ?? 400,
        moveScale: options.simMoveScale ?? 0.5,
        disabled: options.simDisabled ?? {},
      };

      // Ensemble: equal-weight average of the wide model and the trees by default.
      this.useNN = options.useNN ?? false;       // add the neural net as a third model
      this.stacking = options.stacking ?? false; // fit ensemble weights on out-of-fold predictions
      this.folds = options.folds ?? 3;           // folds used when stacking = true
      this.seed = options.seed ?? 42;

      this.enemies = [];
      this.enemyLookup = new Map();
      this.simCache = new Map();
      this.weights = [0.5, 0, 0.5]; // [wide, nn, xgb]
    }

    _expandTeam(team) {
      const units = [];
      for (const [name, count] of Object.entries(team || {})) {
        const found = this.enemyLookup.get(norm(name));
        if (found) {
          for (const u of found) units.push({ u, n: count });
        }
      }
      return units;
    }

    // --- Features ------------------------------------------------------------

    // Simulator features for (teamA vs teamB). The battle is always simulated in
    // a canonical team order and the sign flipped if needed, so the features are
    // exactly antisymmetric: sim(A, B) === -sim(B, A).
    _simFeatures(tA, tB) {
      const kA = teamKey(tA), kB = teamKey(tB);
      const swap = kA > kB;
      const o = this.simOpt;
      const key = [JSON.stringify(o.disabled), this.simSeeds, o.arenaWidth, o.arenaLength, o.spawnSpread, o.dt, o.maxTime, o.moveScale,
        swap ? kB : kA, swap ? kA : kB].join('#');

      let f = this.simCache.get(key);
      if (!f) {
        const u1 = this._expandTeam(swap ? tB : tA), u2 = this._expandTeam(swap ? tA : tB);
        const base = hashString(key);
        let mg = 0, win = 0, mg0 = 0;
        for (let s = 0; s < this.simSeeds; s++) {
          const seed = (base + Math.imul(s + 1, 0x9e3779b1)) | 0;
          const a = simulateBattle(u1, u2, o, seed, true);
          const b = simulateBattle(u1, u2, o, seed, false);
          mg += a; win += Math.sign(a); mg0 += b;
        }
        mg /= this.simSeeds; win /= this.simSeeds; mg0 /= this.simSeeds;
        f = [mg, win, Math.sign(mg) * Math.sqrt(Math.abs(mg)), mg0];
        this.simCache.set(key, f);
      }
      return swap ? f.map((v) => -v) : f;
    }

    // Full feature row: [sqrt-count diffs per enemy | engineered stat diffs | simulator]
    _row(tA, tB) {
      const enemies = this.enemies, nRaw = enemies.length;
      const cnt = (team, e) => Math.max(Number((team && team[e]) || 0) || 0, 0);
      const eng = pairDiffFeatures(this._expandTeam(tA), this._expandTeam(tB));
      const sim = this._simFeatures(tA, tB);
      const x = new Float64Array(nRaw + eng.length + SIM_DIM);
      for (let j = 0; j < nRaw; j++) x[j] = Math.sqrt(cnt(tA, enemies[j])) - Math.sqrt(cnt(tB, enemies[j]));
      for (let k = 0; k < eng.length; k++) x[nRaw + k] = eng[k];
      for (let k = 0; k < SIM_DIM; k++) x[nRaw + eng.length + k] = sim[k];
      return x;
    }

    // Row for the neural net: [team A stats | team B stats | time-to-kill terms | simulator]
    // (no unit identities, no storm). Call with the teams swapped for the mirrored row.
    _nnRow(tA, tB) {
      const stat = buildNnStatFeatures(this._expandTeam(tA), this._expandTeam(tB));
      const sim = this._simFeatures(tA, tB);
      const x = new Float64Array(stat.length + SIM_DIM);
      for (let k = 0; k < stat.length; k++) x[k] = stat[k];
      for (let k = 0; k < SIM_DIM; k++) x[stat.length + k] = sim[k];
      return x;
    }

    // --- Base models (each takes precomputed rows) ----------------------------

    // fwd[s] is the (A vs B) row of match s, swp[s] the (B vs A) row.
    _trainNNOnce(fwd, swp, y, sw, rng) {
      const n = fwd.length, dim = fwd[0].length;
      const rawX = [], Y = [], Wt = [];
      for (let s = 0; s < n; s++) {
        rawX.push(fwd[s]); Y.push(y[s]); Wt.push(sw[s]);
        rawX.push(swp[s]); Y.push(1 - y[s]); Wt.push(sw[s]);
      }

      const mean = new Float64Array(dim), std = new Float64Array(dim);
      for (const x of rawX) for (let j = 0; j < dim; j++) mean[j] += x[j] / rawX.length;
      for (const x of rawX) for (let j = 0; j < dim; j++) std[j] += (x[j] - mean[j]) ** 2 / rawX.length;
      for (let j = 0; j < dim; j++) std[j] = Math.sqrt(std[j]) || 1;

      const clip = this.nnClip;
      const normVec = (v) => {
        const out = new Float64Array(dim);
        for (let j = 0; j < dim; j++) out[j] = clamp((v[j] - mean[j]) / std[j], -clip, clip);
        return out;
      };
      const X = rawX.map(normVec);

      const net = new BrowserMLP([dim, ...this.nnHidden, 1], rng);
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
          for (const i of batch) net.backward(X[i], Y[i], Wt[i], gW, gb);

          step++;
          const c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
          for (let l = 0; l < net.L; l++) {
            for (let k = 0; k < net.W[l].length; k++) {
              const g = gW[l][k] / batch.length + this.nnWd * net.W[l][k];
              mW[l][k] = 0.9 * mW[l][k] + 0.1 * g;
              vW[l][k] = 0.999 * vW[l][k] + 0.001 * g * g;
              net.W[l][k] -= (this.nnLr * (mW[l][k] / c1)) / (Math.sqrt(vW[l][k] / c2) + 1e-8);
            }
            for (let k = 0; k < net.b[l].length; k++) {
              const g = gb[l][k] / batch.length;
              mB[l][k] = 0.9 * mB[l][k] + 0.1 * g;
              vB[l][k] = 0.999 * vB[l][k] + 0.001 * g * g;
              net.b[l][k] -= (this.nnLr * (mB[l][k] / c1)) / (Math.sqrt(vB[l][k] / c2) + 1e-8);
            }
          }
        }
      }

      return nnMember({
        sizes: net.sizes,
        W: net.W.map((w) => Array.from(w)),
        b: net.b.map((b) => Array.from(b)),
        mean: Array.from(mean), std: Array.from(std), clip,
      });
    }

    _trainNN(fwd, swp, y, sw, rng) {
      const members = [];
      const k = Math.max(1, this.nnEnsemble | 0);
      for (let i = 0; i < k; i++) {
        members.push(this._trainNNOnce(fwd, swp, y, sw, mulberry32(Math.floor(rng() * 2147483647))));
      }
      return nnModel(members);
    }

    _trainWide(rows, y, sw) {
      const n = rows.length, d = rows[0].length;
      // Features are antisymmetric, so their mean under A/B swapping is zero:
      // scale by the root-mean-square only.
      const scale = new Float64Array(d);
      for (let s = 0; s < n; s++) for (let j = 0; j < d; j++) scale[j] += rows[s][j] * rows[s][j];
      for (let j = 0; j < d; j++) scale[j] = Math.sqrt(scale[j] / n) || 1;
      const scaled = rows.map((r) => {
        const x = new Float64Array(d);
        for (let j = 0; j < d; j++) x[j] = r[j] / scale[j];
        return x;
      });
      const ws = fitLogistic(scaled, y, sw, this.wideL2, this.wideIters);
      const w = new Float64Array(d);
      for (let j = 0; j < d; j++) w[j] = ws[j] / scale[j];
      return wideModel(w);
    }

    _trainXGB(rows, yIn, swIn, rng) {
      const n = rows.length, d = rows[0].length, nTotal = 2 * n;
      const lambda = this.xgbLambda, gamma = this.xgbGamma, minChild = this.xgbMinChildWeight;
      const lr = this.xgbLr, maxDepth = this.xgbDepth;

      // Mirror-augmented design matrix: row s is (A vs B), row s + n is (B vs A)
      const X = new Float64Array(nTotal * d), y = new Float64Array(nTotal), sw = new Float64Array(nTotal);
      for (let s = 0; s < n; s++) {
        y[s] = yIn[s]; y[s + n] = 1 - yIn[s];
        sw[s] = swIn[s]; sw[s + n] = swIn[s];
        const r = rows[s];
        for (let j = 0; j < d; j++) {
          X[s * d + j] = r[j];
          X[(s + n) * d + j] = -r[j];
        }
      }

      const mtry = Math.max(1, Math.floor(d * this.xgbColsample));
      const sampleSize = Math.max(1, Math.floor(n * this.xgbSubsample));
      const score = (G, H) => (G * G) / (H + lambda);

      const buildTree = (nodeRows, g, h, depth) => {
        let G = 0, H = 0;
        for (let i = 0; i < nodeRows.length; i++) { G += g[nodeRows[i]]; H += h[nodeRows[i]]; }
        const leaf = { leaf: -(G / (H + lambda)) * lr };
        if (depth >= maxDepth || nodeRows.length <= 1 || H < minChild) return leaf;

        const feats = new Int32Array(d);
        for (let j = 0; j < d; j++) feats[j] = j;
        const k = Math.min(mtry, d);
        for (let j = 0; j < k; j++) {
          const t = j + Math.floor(rng() * (d - j));
          const tmp = feats[j]; feats[j] = feats[t]; feats[t] = tmp;
        }

        let bestGain = 0, bestF = -1, bestT = 0;
        for (let fi = 0; fi < k; fi++) {
          const f = feats[fi];
          const sorted = nodeRows.slice().sort((a, b) => X[a * d + f] - X[b * d + f]);
          let GL = 0, HL = 0;
          for (let i = 0; i < sorted.length - 1; i++) {
            const r = sorted[i]; GL += g[r]; HL += h[r];
            const v = X[r * d + f], vNext = X[sorted[i + 1] * d + f];
            if (v === vNext) continue;
            const GR = G - GL, HR = H - HL;
            if (HL < minChild || HR < minChild) continue;
            const gain = 0.5 * (score(GL, HL) + score(GR, HR) - score(G, H)) - gamma;
            if (gain > bestGain) { bestGain = gain; bestF = f; bestT = (v + vNext) / 2; }
          }
        }
        if (bestF < 0 || bestGain <= 0) return leaf;

        const left = [], right = [];
        for (let i = 0; i < nodeRows.length; i++) {
          const r = nodeRows[i];
          (X[r * d + bestF] <= bestT ? left : right).push(r);
        }
        return {
          f: bestF, t: bestT,
          l: buildTree(left, g, h, depth + 1),
          r: buildTree(right, g, h, depth + 1),
        };
      };

      const rowScore = (node, off) => {
        while (node.leaf === undefined) node = X[off + node.f] <= node.t ? node.l : node.r;
        return node.leaf;
      };
      const raw = new Float64Array(nTotal);
      const g = new Float64Array(nTotal), h = new Float64Array(nTotal);
      const trees = [];
      for (let t = 0; t < this.xgbTrees; t++) {
        for (let i = 0; i < nTotal; i++) {
          const p = sigmoid(raw[i]);
          g[i] = (p - y[i]) * sw[i];
          h[i] = Math.max(p * (1 - p) * sw[i], 1e-16);
        }
        const sample = [];
        for (let k = 0; k < sampleSize; k++) {
          const idx = Math.floor(rng() * n);
          sample.push(idx, idx + n);
        }
        const tree = buildTree(sample, g, h, 0);
        trees.push(tree);
        for (let i = 0; i < nTotal; i++) raw[i] += rowScore(tree, i * d);
      }

      return xgbModel(trees);
    }

    // --- Fit / predict -------------------------------------------------------

    // Builds the unit lookup (and simulator cache) from the enemies data
    _setEnemies(enemiesData) {
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
      // Resolve units that are spawned by other units when they die
      for (const list of lookup.values()) {
        for (const u of list) {
          if (u.sim.spawnName && !u.sim.spawnUnits) u.sim.spawnUnits = lookup.get(norm(u.sim.spawnName)) || null;
        }
      }

      // Share simulator results across models built from the same enemies array
      if (enemiesData && typeof enemiesData === 'object') {
        if (!SIM_CACHES.has(enemiesData)) SIM_CACHES.set(enemiesData, new Map());
        this.simCache = SIM_CACHES.get(enemiesData);
      } else {
        this.simCache = new Map();
      }
    }

    fit(matches, enemiesData, onProgress) {
      if (!matches || !matches.length) throw new Error('No matches provided');

      this._setEnemies(enemiesData);

      const eSet = new Set();
      for (const m of matches) {
        if (m.teamA) Object.keys(m.teamA).forEach((e) => eSet.add(e));
        if (m.teamB) Object.keys(m.teamB).forEach((e) => eSet.add(e));
      }
      this.enemies = Array.from(eSet).sort();

      // Precompute every feature row once (the simulator runs here)
      const n = matches.length;
      const useNN = this.useNN;
      const rows = new Array(n), nnFwd = new Array(n), nnSwp = new Array(n);
      const y = new Float64Array(n), sw = new Float64Array(n);
      for (let s = 0; s < n; s++) {
        const m = matches[s];
        rows[s] = this._row(m.teamA, m.teamB);
        if (onProgress && (s % 25 === 0 || s === n - 1)) onProgress((s + 1) / n);
        if (useNN) {
          nnFwd[s] = this._nnRow(m.teamA, m.teamB);
          nnSwp[s] = this._nnRow(m.teamB, m.teamA);
        }
        y[s] = m.winner === 'A' ? 1 : 0;
        sw[s] = m.storm === true ? this.stormWeight : 1.0;
      }

      const trainAll = (idx, seedBase) => {
        const pick = (arr) => idx.map((i) => arr[i]);
        const yy = Float64Array.from(idx, (i) => y[i]);
        const ww = Float64Array.from(idx, (i) => sw[i]);
        return {
          wide: this._trainWide(pick(rows), yy, ww),
          nn: useNN ? this._trainNN(pick(nnFwd), pick(nnSwp), yy, ww, mulberry32(seedBase + 7919)) : null,
          xgb: this._trainXGB(pick(rows), yy, ww, mulberry32(seedBase)),
        };
      };

      // Ensemble weights, ordered [wide, nn, xgb]. Default: equal weights over the
      // active models. With stacking = true they are fitted on out-of-fold predictions.
      const active = [true, useNN, true];
      const nActive = active.filter(Boolean).length;
      const w = Float64Array.from(active, (a) => (a ? 1 / nActive : 0));
      const K = Math.max(1, Math.min(this.folds, n));
      if (this.stacking && K >= 2) {
        const oofLogits = [], oofTargets = [];
        for (let f = 0; f < K; f++) {
          const tr = [], te = [];
          for (let i = 0; i < n; i++) (i % K === f ? te : tr).push(i);
          const fold = trainAll(tr, this.seed + f * 17 + 1);
          for (const i of te) {
            oofLogits.push([
              logit(fold.wide.predictRow(rows[i])),
              useNN ? logit(fold.nn.predictPair(nnFwd[i], nnSwp[i])) : 0,
              logit(fold.xgb.predictRow(rows[i])),
            ]);
            oofTargets.push(y[i]);
          }
        }
        const mW = new Float64Array(3), vW = new Float64Array(3);
        for (let it = 1; it <= 400; it++) {
          const grad = new Float64Array(3);
          for (let s = 0; s < oofLogits.length; s++) {
            const o = oofLogits[s];
            const err = sigmoid(w[0] * o[0] + w[1] * o[1] + w[2] * o[2]) - oofTargets[s];
            for (let j = 0; j < 3; j++) grad[j] += (err * o[j]) / oofLogits.length;
          }
          for (let j = 0; j < 3; j++) {
            if (!active[j]) continue;
            grad[j] += 0.001 * w[j];
            mW[j] = 0.9 * mW[j] + 0.1 * grad[j];
            vW[j] = 0.999 * vW[j] + 0.001 * grad[j] * grad[j];
            w[j] -= (0.05 * (mW[j] / (1 - Math.pow(0.9, it)))) / (Math.sqrt(vW[j] / (1 - Math.pow(0.999, it))) + 1e-8);
          }
        }
      }
      this.weights = Array.from(w);

      // Final base models on all the data
      const all = [];
      for (let i = 0; i < n; i++) all.push(i);
      const full = trainAll(all, this.seed + 999);
      this.modelWide = full.wide;
      this.modelNN = full.nn;
      this.modelXGB = full.xgb;
      return this;
    }

    // `storm` is accepted for backward compatibility and ignored.
    predictDetailed(teamA, teamB, storm = false) { // eslint-disable-line no-unused-vars
      if (!this.modelWide) throw new Error('Model has not been fitted');
      const row = this._row(teamA, teamB);
      const pWide = this.modelWide.predictRow(row);
      const pNN = this.modelNN
        ? this.modelNN.predictPair(this._nnRow(teamA, teamB), this._nnRow(teamB, teamA))
        : null;
      const pXGB = this.modelXGB.predictRow(row);
      const sim = this._simFeatures(teamA, teamB);

      let z = this.weights[0] * logit(pWide) + this.weights[2] * logit(pXGB);
      if (pNN !== null) z += this.weights[1] * logit(pNN);
      return {
        p: sigmoid(z),
        wide: pWide,
        nn: pNN,           // null unless the model was built with { useNN: true }
        xgb: pXGB,
        simMargin: sim[0], // > 0 means the simulator favours team A
        simWinRate: (sim[1] + 1) / 2, // share of simulated battles won by team A
        lr: pWide,         // legacy alias (the wide model is the stack's logistic regression)
        rf: pXGB,          // legacy alias
        weights: this.weights,
      };
    }

    // --- Save / load ----------------------------------------------------------

    // Plain object holding everything needed to predict (use JSON.stringify on it).
    toJSON() {
      if (!this.modelWide) throw new Error('Model has not been fitted');
      return {
        format: 'DuelStackModel', version: 5, simVersion: SIM_VERSION,
        options: {
          useNN: this.useNN, simSeeds: this.simSeeds,
          simArenaLength: this.simOpt.arenaLength, simArenaWidth: this.simOpt.arenaWidth,
          simSpawnSpread: this.simOpt.spawnSpread, simDt: this.simOpt.dt,
          simMaxTime: this.simOpt.maxTime, simMoveScale: this.simOpt.moveScale,
          simDisabled: this.simOpt.disabled,
        },
        enemies: this.enemies,
        weights: this.weights,
        wide: Array.from(this.modelWide.w),
        xgb: this.modelXGB.trees,
        nn: this.modelNN ? this.modelNN.members.map((mm) => mm.state) : null,
      };
    }

    // Rebuilds a trained model without training. `saved` is the object (or JSON
    // string) from toJSON(); `enemiesData` is the same enemies array used to train.
    static fromJSON(saved, enemiesData) {
      const o = typeof saved === 'string' ? JSON.parse(saved) : saved;
      if (!o || o.format !== 'DuelStackModel') throw new Error('Not a saved DuelStackModel');
      const model = new DuelStackModel(o.options || {});
      model.outdated = o.simVersion !== SIM_VERSION;
      if (model.outdated && typeof console !== 'undefined') {
        console.warn('This saved model was trained with a different version of stack.js. Re-run train.js.');
      }
      model._setEnemies(enemiesData);
      model.enemies = o.enemies;
      model.weights = o.weights;
      model.modelWide = wideModel(Float64Array.from(o.wide));
      model.modelXGB = xgbModel(o.xgb);
      model.modelNN = o.nn ? nnModel(o.nn.map(nnMember)) : null;
      return model;
    }

    predictProba(teamA, teamB, storm = false) {
      return this.predictDetailed(teamA, teamB, storm).p;
    }
  }

  DuelStackModel.simulateBattle = simulateBattle;

  // Universal export: supports browser window, Node.js module.exports, and global
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = DuelStackModel;
    module.exports.DuelStackModel = DuelStackModel;
    module.exports.EnemyStrengthModel = DuelStackModel;
  }
  if (typeof global !== 'undefined') {
    global.DuelStackModel = DuelStackModel;
    global.EnemyStrengthModel = DuelStackModel;
  }
  if (typeof window !== 'undefined') {
    window.DuelStackModel = DuelStackModel;
    window.EnemyStrengthModel = DuelStackModel;
  }
})(typeof window !== 'undefined' ? window : (typeof global !== 'undefined' ? global : this));