// Lifecycle Assessment of ML Models under Drift: continuous per-data-point drift + energy model.
// Layout: 4 workflow tabs (01 Models, 02 Configuration, 03 Assessment, 04 Which Model to Deploy) + Methodology reference.
// Fully optimized for O(1) memory, bounded SVG rendering, and rAF debouncing.

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- inputs & state
const MACHINES = {
  small: { name: "Small", ex: "m5.large", vcpu: 2, watts: 7.9, embGPerHr: 1.427 },
  medium: { name: "Medium", ex: "m5.xlarge", vcpu: 4, watts: 15.9, embGPerHr: 2.854 },
  large: { name: "Large", ex: "m5.2xlarge", vcpu: 8, watts: 31.8, embGPerHr: 5.708 },
};
const GRIDS = [
  { id: "low",    name: "Low-carbon",     g: 50,  src: "Hydro / Nuclear (e.g. Nordics, France: 20–80 gCO₂e/kWh)" },
  { id: "clean",  name: "Cleaner grid",   g: 230, src: "Renewables-heavy mixed grid (e.g. EU average ~230 gCO₂e/kWh)" },
  { id: "global", name: "Global average", g: 450, src: "World grid average (~450 gCO₂e/kWh, IEA / Ember)" },
  { id: "coal",   name: "Carbon-heavy",   g: 650, src: "Coal-dependent grid (600–900 gCO₂e/kWh, e.g. coal regions)" },
];
const POLICIES = {
  none: { name: "No retraining", meta: "never retrain" },
  fixed: { name: "Fixed schedule", meta: "every N months" },
  absolute: { name: "Accuracy", meta: "below threshold τ, e.g. 82.0%" },
};
const PRESETS = [
  { name: "Stable data", m: 0 },
  { name: "Mild drift", m: 0.2 },
  { name: "Moderate drift", m: 0.5 },
  { name: "Severe drift", m: 0.9 },
];
const FIELDS = {
  acc: { label: "Accuracy", min: 50, max: 100, step: 0.5, fmt: v => `${+v.toFixed(1)}%` },
  train: { label: "Training time", min: 5, max: 3600, log: true, fmt: v => fmtTime(v) },
  inf: { label: "Inference time", min: 0.01, max: 50, log: true, fmt: v => `${+v.toPrecision(2)} ms / pred` },
  rob: { label: "Robustness", min: 0.1, max: 5.0, step: 0.1, fmt: v => `${+v.toFixed(1)} pts / 100k`,
    hint: "Accuracy drop per 100k data points under drift magnitude 1. Lower is more robust." },
};
const MAX_PIPELINES = 8;

const DEFAULT_STATE = {
  pipelines: [
    { slot: 1, name: "Deep neural net", acc: 93.0, train: 1200, inf: 0.50, rob: 3.2 },
    { slot: 2, name: "Random forest",   acc: 92.0, train: 180,  inf: 0.15, rob: 0.2 },
    { slot: 3, name: "Logistic regression", acc: 84.0, train: 15, inf: 0.02, rob: 0.5 },
  ],
  selSlot: 1,
  selectedDecisionRule: 0,
  months: 12,
  trafficSlider: (Math.log10(5000) - 2) / 3,
  traffic: 5000,
  drift: 0.50,
  floor: 50,
  tau: 0.82,
  subthresholdRetrain: true,
  subthresholdDays: 30,
  retrainMonths: 2,
  policy: "absolute",
  machine: "medium",
  gridIntensity: 230,
  gridPreset: "clean",
  hardwareUtil: 100,
  datacenterPue: 1.20,
};

let pipelines = JSON.parse(JSON.stringify(DEFAULT_STATE.pipelines));
let selSlot = 1;
let policy = "absolute";
let machine = "medium";
let gridPreset = "clean";
let gridIntensity = 230;
let hardwareUtil = 100;
let datacenterPue = 1.20;
let currentTab = "pipelines";
let selectedDecisionRule = 0;

function isStateModified() {
  if (pipelines.length !== DEFAULT_STATE.pipelines.length) return true;
  for (let i = 0; i < pipelines.length; i++) {
    const p = pipelines[i], d = DEFAULT_STATE.pipelines[i];
    if (p.name !== d.name ||
        Math.abs(p.acc - d.acc) > 0.01 ||
        Math.abs(p.train - d.train) > 0.01 ||
        Math.abs(p.inf - d.inf) > 0.001 ||
        Math.abs(p.rob - d.rob) > 0.01) {
      return true;
    }
  }
  if (policy !== DEFAULT_STATE.policy) return true;
  if (machine !== DEFAULT_STATE.machine) return true;
  if (gridIntensity !== DEFAULT_STATE.gridIntensity) return true;
  if (selectedDecisionRule !== DEFAULT_STATE.selectedDecisionRule) return true;

  if ($("months") && +$("months").value !== DEFAULT_STATE.months) return true;
  if ($("drift") && Math.abs(+$("drift").value - DEFAULT_STATE.drift) > 0.01) return true;
  if ($("floor") && +$("floor").value !== DEFAULT_STATE.floor) return true;
  if ($("tau") && Math.abs(+$("tau").value - DEFAULT_STATE.tau) > 0.001) return true;
  if ($("n") && +$("n").value !== DEFAULT_STATE.retrainMonths) return true;
  if ($("traffic")) {
    const curTraffic = niceRound(10 ** (2 + 3 * +$("traffic").value));
    if (curTraffic !== DEFAULT_STATE.traffic) return true;
  }
  if ($("subthreshold-retrain") && $("subthreshold-retrain").checked !== DEFAULT_STATE.subthresholdRetrain) return true;
  if ($("subthreshold-days") && +$("subthreshold-days").value !== DEFAULT_STATE.subthresholdDays) return true;
  if ($("hardware-util") && +$("hardware-util").value !== DEFAULT_STATE.hardwareUtil) return true;
  if ($("datacenter-pue") && Math.abs(+$("datacenter-pue").value - DEFAULT_STATE.datacenterPue) > 0.01) return true;

  return false;
}

function updateDefaultsButtons() {
  const modified = isStateModified();
  document.querySelectorAll(".btn-restore-defaults").forEach(btn => {
    btn.hidden = !modified;
  });
}

function restoreAllDefaults() {
  pipelines = JSON.parse(JSON.stringify(DEFAULT_STATE.pipelines));
  selSlot = DEFAULT_STATE.selSlot;
  selectedDecisionRule = DEFAULT_STATE.selectedDecisionRule;
  policy = DEFAULT_STATE.policy;
  machine = DEFAULT_STATE.machine;
  gridIntensity = DEFAULT_STATE.gridIntensity;
  gridPreset = DEFAULT_STATE.gridPreset;
  hardwareUtil = DEFAULT_STATE.hardwareUtil;
  datacenterPue = DEFAULT_STATE.datacenterPue;

  if ($("months")) $("months").value = DEFAULT_STATE.months;
  if ($("traffic")) $("traffic").value = DEFAULT_STATE.trafficSlider;
  if ($("drift")) $("drift").value = DEFAULT_STATE.drift;
  if ($("floor")) $("floor").value = DEFAULT_STATE.floor;
  if ($("tau")) $("tau").value = DEFAULT_STATE.tau;
  if ($("n")) $("n").value = DEFAULT_STATE.retrainMonths;
  if ($("subthreshold-retrain")) $("subthreshold-retrain").checked = DEFAULT_STATE.subthresholdRetrain;
  if ($("subthreshold-days")) $("subthreshold-days").value = DEFAULT_STATE.subthresholdDays;
  if ($("grid-intensity")) $("grid-intensity").value = DEFAULT_STATE.gridIntensity;
  if ($("hardware-util")) $("hardware-util").value = DEFAULT_STATE.hardwareUtil;
  if ($("datacenter-pue")) $("datacenter-pue").value = DEFAULT_STATE.datacenterPue;

  renderStaticOpts();
  const rows = pipelines.map(p => ({ p, sim: simulate(p, settings()) }));
  renderPipeOpts(rows);
  update();
  updateDefaultsButtons();
}
const color = p => `var(--s${p.slot})`;
const selected = () => pipelines.find(p => p.slot === selSlot);

const toLog = (t, f) => f.min * (f.max / f.min) ** t;
const fromLog = (v, f) => Math.log(v / f.min) / Math.log(f.max / f.min);

// ---------------------------------------------------------------- formatting
function fmtWh(j) {
  const wh = j / 3600;
  if (wh === 0) return "0 Wh";
  if (wh < 0.01) return wh.toPrecision(2) + " Wh";
  if (wh >= 1000) return (wh / 1000).toFixed(2) + " kWh";
  return wh.toFixed(2) + " Wh";
}
function fmtCO2(g) {
  if (g === 0) return "0 g";
  if (g < 0.001) return (g * 1000).toFixed(2) + " mg";
  if (g < 1) return (g < 0.01 ? g.toPrecision(2) : g.toFixed(2)) + " g";
  if (g >= 1e6) return (g / 1e6).toFixed(2) + " t CO₂e";
  if (g >= 1000) return (g / 1000).toFixed(2) + " kg CO₂e";
  return g.toFixed(1) + " g";
}
function fmtSciRate(gPerK) {
  if (gPerK === 0) return "0 g / 1k";
  if (gPerK < 0.001) return (gPerK * 1000).toFixed(2) + " mg / 1k";
  if (gPerK < 0.1) return gPerK.toFixed(3) + " g / 1k";
  if (gPerK < 10) return gPerK.toFixed(2) + " g / 1k";
  return gPerK.toFixed(1) + " g / 1k";
}
function fmtTime(s) {
  if (s < 120) return `${+s.toFixed(s < 10 ? 2 : 1)} s`;
  if (s < 7200) return `${+(s / 60).toFixed(1)} min`;
  return `${+(s / 3600).toFixed(1)} h`;
}
const fmtPct = v => v.toFixed(2) + "%";
const fmtRatio = x => `${x < 10 ? x.toFixed(1) : Math.round(x).toLocaleString("en-US")}×`;
const fmtNum = v => Math.round(v).toLocaleString("en-US");

function fmtMonths(m) {
  if (m < 12) return `${m} month${m === 1 ? "" : "s"}`;
  const y = +(m / 12).toFixed(1);
  return `${m} months (${y} yr${y === 1 ? "" : "s"})`;
}

function settings() {
  const months = +$("months").value;
  const days = months * 30;
  const traffic = niceRound(10 ** (2 + 3 * +$("traffic").value)); // 100 .. 100,000
  const gridG = $("grid-intensity") ? +$("grid-intensity").value : gridIntensity;
  const hardwareUtilVal = $("hardware-util") ? +$("hardware-util").value : hardwareUtil;
  const pueVal = $("datacenter-pue") ? +$("datacenter-pue").value : datacenterPue;
  const mach = MACHINES[machine];
  return {
    m: +$("drift").value,
    months,
    days,
    traffic,
    totalPreds: days * traffic,
    floor: +$("floor").value,
    retrainMonths: +$("n").value,
    tau: +$("tau").value * 100,
    subthresholdRetrain: $("subthreshold-retrain") ? $("subthreshold-retrain").checked : true,
    subthresholdDays: $("subthreshold-days") ? +$("subthreshold-days").value : 30,
    policy,
    watts: mach.watts,
    gridG,
    hardwareUtil: hardwareUtilVal,
    pue: pueVal,
    embGPerHr: mach.embGPerHr,
    embGPerSec: mach.embGPerHr / 3600,
  };
}

function niceRound(v) {
  const mag = 10 ** (Math.floor(Math.log10(v)) - 1);
  return Math.round(v / mag) * mag;
}

// ---------------------------------------------------------------- model (O(1) Memory & Time)
function simulate(p, s) {
  const driftPerPred = (p.rob / 100000) * s.m;
  const dailyDrop = driftPerPred * s.traffic;

  let cycle = Infinity;
  if (s.policy === "fixed") {
    cycle = s.retrainMonths * 30;
  } else if (s.policy === "absolute") {
    if (p.acc < s.tau) {
      cycle = s.subthresholdRetrain ? Math.min(s.subthresholdDays, s.days) : Infinity;
    } else if (dailyDrop > 0 && s.tau >= s.floor) {
      cycle = Math.max(1, (p.acc - s.tau) / dailyDrop);
    }
  }

  const retrains = cycle < s.days ? Math.floor((s.days - 1e-6) / cycle) : 0;
  const J = sec => sec * s.watts;

  // Development (Offline)
  const evalPreds = 10000;
  const initS = p.train;
  const evalInfS = (p.inf / 1000) * evalPreds;
  const devCpuS = initS + evalInfS;
  const initJ = J(initS);
  const evalJ = J(evalInfS);
  const devJ = initJ + evalJ;

  // Deployment (Online)
  const retrainS = p.train * retrains;
  const infS = (p.inf / 1000) * s.totalPreds;
  const deployCpuS = retrainS + infS;
  const retrainJ = J(retrainS);
  const infJ = J(infS);
  const deployJ = retrainJ + infJ;

  // Total
  const cpuS = devCpuS + deployCpuS;
  const totalJ = devJ + deployJ;

  // Exact analytical integration
  const daysToFloor = dailyDrop > 0 ? (p.acc - s.floor) / dailyDrop : Infinity;
  function segArea(dur) {
    if (dur <= 0) return { area: 0, min: p.acc };
    if (dur <= daysToFloor) {
      const end = p.acc - dailyDrop * dur;
      return { area: ((p.acc + end) / 2) * dur, min: end };
    }
    return { area: ((p.acc + s.floor) / 2) * daysToFloor + s.floor * (dur - daysToFloor), min: s.floor };
  }

  let totalArea = 0, minAcc = p.acc;
  if (retrains > 0) {
    const cStat = segArea(cycle);
    totalArea += retrains * cStat.area;
    minAcc = Math.min(minAcc, cStat.min);
    const rem = s.days - retrains * cycle;
    if (rem > 0) {
      const rStat = segArea(rem);
      totalArea += rStat.area;
      minAcc = Math.min(minAcc, rStat.min);
    }
  } else {
    const aStat = segArea(s.days);
    totalArea = aStat.area;
    minAcc = aStat.min;
  }

  const avgAcc = totalArea / s.days;
  const maxAcc = p.acc;
  const correctPreds = Math.round((avgAcc / 100) * s.totalPreds);

  // GSF SCI for AI Calculations (ISO/IEC 21031:2024)
  const pue = s.pue;
  const devFacilityKwh = (devJ / 3600 / 1000) * pue;
  const deployFacilityKwh = (deployJ / 3600 / 1000) * pue;
  const totalFacilityKwh = (totalJ / 3600 / 1000) * pue;

  // Operational Carbon (O = E * I in gCO2e)
  const opDevG = devFacilityKwh * s.gridG;
  const opDeployG = deployFacilityKwh * s.gridG;
  const opTotalG = totalFacilityKwh * s.gridG;

  // Embodied Carbon (M = CPU_sec * rate / util in gCO2e)
  const util = Math.max(0.1, s.hardwareUtil / 100);
  const embRateSec = s.embGPerSec / util;
  const embDevG = devCpuS * embRateSec;
  const embDeployG = deployCpuS * embRateSec;
  const embTotalG = cpuS * embRateSec;

  // Combined Carbon per stage (gCO2e)
  const carbonDevG = opDevG + embDevG;
  const carbonDeployG = opDeployG + embDeployG;
  const carbonTotalG = opTotalG + embTotalG;

  // Retraining carbon impact
  const retrainKwh = (retrainJ / 3600 / 1000) * pue;
  const retrainOpG = retrainKwh * s.gridG;
  const retrainEmbG = retrainS * embRateSec;
  const retrainCarbonG = retrainOpG + retrainEmbG;

  // SCI functional units
  const kPreds = s.totalPreds / 1000;
  const kCorrect = correctPreds / 1000;
  const sciConsumer = kPreds > 0 ? carbonDeployG / kPreds : 0;
  const sciEffective = kCorrect > 0 ? carbonDeployG / kCorrect : 0;
  const sciProvider = carbonDevG;
  const sciLifecycle = kPreds > 0 ? carbonTotalG / kPreds : 0;

  return {
    cycle, retrains, avgAcc, minAcc, maxAcc, correctPreds,
    dailyDrop,
    evalPreds,
    devCpuS, deployCpuS, cpuS,
    initJ, evalJ, devJ,
    retrainJ, infJ, deployJ, totalJ,
    whPerK: totalJ / 3600 / (correctPreds / 1000),
    opDevG, opDeployG, opTotalG,
    embDevG, embDeployG, embTotalG,
    carbonDevG, carbonDeployG, carbonTotalG,
    retrainCarbonG,
    sciConsumer, sciEffective, sciProvider, sciLifecycle,
  };
}

function expectedRetrains(p, s) {
  if (s.policy === "none") return 0;
  if (s.policy === "fixed") {
    const cycle = s.retrainMonths * 30;
    return Math.max(0, Math.floor((s.days - 1e-6) / cycle));
  }
  const driftPerPred = (p.rob / 100000) * s.m;
  const dailyDrop = driftPerPred * s.traffic;
  if (s.policy === "absolute") {
    if (p.acc < s.tau) {
      if (!s.subthresholdRetrain) return 0;
      const c = Math.min(s.subthresholdDays, s.days);
      return Math.max(0, Math.floor((s.days - 1e-6) / c));
    }
    if (dailyDrop <= 0 || s.tau < s.floor) return 0;
    const cycle = Math.max(1, (p.acc - s.tau) / dailyDrop);
    return Math.max(0, Math.floor((s.days - 1e-6) / cycle));
  }
  return 0;
}

const expectedCostJ = (p, s) => (expectedRetrains(p, s) * p.train + (p.inf / 1000) * s.totalPreds) * s.watts;

// ---------------------------------------------------------------- step 1: candidate models
function archetypeDesc(p) {
  if (p.train >= 600) return "High-capacity deep model";
  if (p.train >= 60) return "Balanced ensemble model";
  return "Lightweight fast baseline";
}

function renderPipeOpts(rows) {
  const optsEl = $("pipe-opts");
  if (!optsEl) return;
  if ($("pipe-add")) $("pipe-add").disabled = pipelines.length >= MAX_PIPELINES;

  // If user is actively typing or dragging an input inside #pipe-opts, do not overwrite innerHTML
  const active = document.activeElement;
  if (active && optsEl.contains(active)) {
    optsEl.querySelectorAll(".bill-card").forEach(c => {
      const slot = +c.dataset.slot;
      c.classList.toggle("sel", slot === selSlot);
    });
    return;
  }

  optsEl.innerHTML = pipelines.map(p => {
    const isSel = p.slot === selSlot;
    return `
      <div class="bill-card model-card${isSel ? " sel" : ""}" data-slot="${p.slot}" tabindex="0" role="region" aria-label="Model ${esc(p.name)}">
        <div class="bill">
          <!-- Header -->
          <div class="r-head">
            <div class="t">
              <span class="dot" style="background:${color(p)}"></span>
              <input class="model-name-input" data-slot="${p.slot}" value="${esc(p.name)}" aria-label="Model name" title="Click to rename model" spellcheck="false">
              ${pipelines.length > 1 ? `<button class="model-remove-btn" type="button" data-remove="${p.slot}" title="Remove ${esc(p.name)}" aria-label="Remove model">×</button>` : ""}
            </div>
            <div class="s">Model #${p.slot} · Candidate Profile</div>
          </div>

          <!-- Part 1: Performance Specs -->
          <div class="bill-part">
            <div class="part-title"><span>Accuracy &amp; Drift</span></div>
            
            <div class="model-param-group">
              <div class="rline">
                <span class="k">Baseline Accuracy</span>
                <span class="v model-val" id="disp-acc-${p.slot}">${+p.acc.toFixed(1)}%</span>
              </div>
              <div class="model-slider-wrap">
                <input type="range" class="model-slider" data-slot="${p.slot}" data-k="acc" min="50" max="100" step="0.5" value="${p.acc}" aria-label="Baseline Accuracy for ${esc(p.name)}">
              </div>
              <div class="model-hint">Test set evaluation accuracy (50–100%)</div>
            </div>

            <div class="model-param-group">
              <div class="rline">
                <span class="k">Drift Robustness</span>
                <span class="v model-val" id="disp-rob-${p.slot}">${+p.rob.toFixed(1)} pts / 100k</span>
              </div>
              <div class="model-slider-wrap">
                <input type="range" class="model-slider" data-slot="${p.slot}" data-k="rob" min="0.1" max="5.0" step="0.1" value="${p.rob}" aria-label="Robustness for ${esc(p.name)}">
              </div>
              <div class="model-hint">Accuracy drop per 100k data points under drift magnitude 1 (lower = more robust)</div>
            </div>
          </div>

          <!-- Part 2: Compute Specs -->
          <div class="bill-part">
            <div class="part-title"><span>Compute &amp; Latency</span></div>

            <div class="model-param-group">
              <div class="rline">
                <span class="k">Training Time</span>
                <span class="v model-val" id="disp-train-${p.slot}">${fmtTime(p.train)}</span>
              </div>
              <div class="model-slider-wrap">
                <input type="range" class="model-slider" data-slot="${p.slot}" data-k="train" min="0" max="1" step="0.001" value="${fromLog(p.train, FIELDS.train)}" aria-label="Training time for ${esc(p.name)}">
              </div>
              <div class="model-hint">Initial train &amp; continuous retrains</div>
            </div>

            <div class="model-param-group">
              <div class="rline">
                <span class="k">Inference Latency</span>
                <span class="v model-val" id="disp-inf-${p.slot}">${+p.inf.toPrecision(2)} ms / pred</span>
              </div>
              <div class="model-slider-wrap">
                <input type="range" class="model-slider" data-slot="${p.slot}" data-k="inf" min="0" max="1" step="0.001" value="${fromLog(p.inf, FIELDS.inf)}" aria-label="Inference latency for ${esc(p.name)}">
              </div>
              <div class="model-hint">Per-prediction query cost (0.01–50ms)</div>
            </div>
          </div>

          <!-- Part 3: Architecture Profile Stat Box (No Energy) -->
          <div class="bill-stat-box" style="margin-top: auto;">
            <div class="stat-meta">
              <div class="stat-label">Model Archetype</div>
              <div class="stat-sub" id="stat-sub-${p.slot}">${archetypeDesc(p)}</div>
            </div>
            <div class="stat-val" id="stat-acc-${p.slot}">${+p.acc.toFixed(1)}%</div>
          </div>
        </div>
      </div>`;
  }).join("");
}

function selectPipeline(slot) {
  selSlot = slot;
  document.querySelectorAll("#pipe-opts .bill-card").forEach(c => {
    c.classList.toggle("sel", +c.dataset.slot === selSlot);
  });
  scheduleUpdate();
  setTimeout(() => {
    const card = document.querySelector(`.bill-card[data-slot="${slot}"]`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
  }, 40);
}

$("pipe-opts").addEventListener("click", ev => {
  const rm = ev.target.closest("[data-remove]");
  if (rm) {
    const slot = +rm.dataset.remove;
    pipelines = pipelines.filter(p => p.slot !== slot);
    if (slot === selSlot) selSlot = pipelines[0].slot;
    renderPipeOpts(last ? last.rows : pipelines.map(p => ({ p })));
    scheduleUpdate();
    return;
  }
  const card = ev.target.closest("[data-slot]");
  if (card) {
    const slot = +card.dataset.slot;
    if (slot !== selSlot) {
      selectPipeline(slot);
    }
  }
});

$("pipe-opts").addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const card = ev.target.closest("[data-slot]");
  if (card && (ev.key === "Enter" || ev.key === " ")) {
    ev.preventDefault();
    selectPipeline(+card.dataset.slot);
  }
});

$("pipe-opts").addEventListener("input", ev => {
  const el = ev.target;
  const slot = +el.dataset.slot;
  if (!slot) return;
  const p = pipelines.find(item => item.slot === slot);
  if (!p) return;

  if (el.classList.contains("model-name-input")) {
    p.name = el.value || "Unnamed";
    scheduleUpdate();
    return;
  }

  const k = el.dataset.k;
  const f = FIELDS[k];
  if (!f) return;

  p[k] = f.log ? toLog(+el.value, f) : +el.value;
  const disp = $(`disp-${k}-${slot}`);
  if (disp) disp.textContent = f.fmt(p[k]);

  if (k === "acc") {
    const statAcc = $(`stat-acc-${slot}`);
    if (statAcc) statAcc.textContent = `${+p.acc.toFixed(1)}%`;
  }
  if (k === "train" || k === "inf") {
    const statSub = $(`stat-sub-${slot}`);
    if (statSub) statSub.textContent = archetypeDesc(p);
  }

  scheduleUpdate();
});

$("pipe-add").addEventListener("click", () => {
  const used = new Set(pipelines.map(p => p.slot));
  const slot = [1, 2, 3, 4, 5, 6, 7, 8].find(s => !used.has(s));
  if (!slot) return;
  pipelines.push({ slot, name: `Model ${slot}`, acc: 88, train: 150, inf: 0.1, rob: 1.0 });
  selSlot = slot;
  renderPipeOpts(last ? last.rows : pipelines.map(p => ({ p })));
  scheduleUpdate();
  setTimeout(() => {
    const newCard = document.querySelector(`#pipe-opts .bill-card[data-slot="${slot}"]`);
    if (newCard) {
      newCard.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
      const nameIn = newCard.querySelector(".model-name-input");
      if (nameIn) nameIn.select();
    }
  }, 40);
});

// ---------------------------------------------------------------- steps 2-4: option cards
function renderStaticOpts() {
  $("policy-opts").innerHTML = Object.entries(POLICIES).map(([k, o]) =>
    `<button class="opt" type="button" data-policy="${k}" aria-pressed="${k === policy}">
      <div class="nm">${o.name}</div><div class="meta">${o.meta}</div></button>`).join("");
  $("machine-opts").innerHTML = Object.entries(MACHINES).map(([k, m]) =>
    `<button class="opt" type="button" data-machine="${k}" aria-pressed="${k === machine}">
      <div class="nm">${m.name}</div><div class="meta">${m.ex} · ${m.vcpu} vCPU<br>${m.watts} J/s · ${m.embGPerHr} g/h emb</div></button>`).join("");
  if ($("grid-opts")) {
    $("grid-opts").innerHTML = GRIDS.map(g =>
      `<button class="opt" type="button" data-grid="${g.id}" aria-pressed="${g.id === gridPreset}">
        <div class="nm">${g.name}</div><div class="meta">${g.g} gCO₂e/kWh<br>${g.src.split("(")[0].trim()}</div></button>`).join("");
  }
  $("ctl-n").hidden = policy !== "fixed";
  if ($("ctl-delta")) $("ctl-delta").hidden = true;
  $("ctl-tau").hidden = policy !== "absolute";
  if ($("ctl-subthreshold")) $("ctl-subthreshold").hidden = policy !== "absolute";
}
$("policy-opts").addEventListener("click", ev => {
  const b = ev.target.closest("[data-policy]");
  if (b) { policy = b.dataset.policy; renderStaticOpts(); scheduleUpdate(); }
});
$("machine-opts").addEventListener("click", ev => {
  const b = ev.target.closest("[data-machine]");
  if (b) { machine = b.dataset.machine; renderStaticOpts(); scheduleUpdate(); }
});
if ($("grid-opts")) {
  $("grid-opts").addEventListener("click", ev => {
    const b = ev.target.closest("[data-grid]");
    if (b) {
      gridPreset = b.dataset.grid;
      const gObj = GRIDS.find(g => g.id === gridPreset);
      if (gObj && $("grid-intensity")) $("grid-intensity").value = gObj.g;
      renderStaticOpts();
      scheduleUpdate();
    }
  });
}
$("presets").addEventListener("click", ev => {
  const b = ev.target.closest("[data-m]");
  if (b) { $("drift").value = b.dataset.m; scheduleUpdate(); }
});

// ---------------------------------------------------------------- bills grid (side-by-side round boxes)
const BAND = [["A", "#2f7d4f"], ["B", "#6a9a3c"], ["C", "#e8a013"], ["D", "#d9772a"], ["E", "#c24a2b"]];
let allDetailsOpen = false;
const openDetails = new Set();

function updateToggleBtnText() {
  const btn = $("btn-toggle-details");
  if (btn) btn.textContent = allDetailsOpen ? "Collapse details" : "Expand details";
}

function renderMiniBreakdown(sim) {
  const tot = sim.totalJ || 1;
  const parts = [
    ["Train", sim.initJ, "#17231d"],
    ["Eval", sim.evalJ, "#555f58"],
    ["Retrain", sim.retrainJ, "#8e9b92"],
    ["Infer", sim.infJ, "#d4ded6"],
  ];
  const segs = parts.map(([n, j, c]) => {
    const pct = (j / tot) * 100;
    return pct > 0.4 ? `<div class="mini-seg" style="width:${pct.toFixed(1)}%;background:${c}" title="${n}: ${pct.toFixed(1)}% (${fmtWh(j)})"></div>` : "";
  }).join("");

  const legend = parts.map(([n, j, c]) => {
    const pct = (j / tot) * 100;
    return pct >= 1 ? `<span class="mini-leg-item"><i style="background:${c}"></i>${n} <b>${pct.toFixed(0)}%</b></span>` : "";
  }).join("");

  return `
    <div class="mini-breakdown">
      <div class="mini-bar">${segs}</div>
      <div class="mini-legend">${legend}</div>
    </div>`;
}

function renderMiniCarbonBreakdown(sim) {
  const tot = sim.carbonTotalG || 1;
  const parts = [
    ["Operational (O)", sim.opTotalG, "#17231d"],
    ["Embodied (M)", sim.embTotalG, "#8e9b92"],
  ];
  const segs = parts.map(([n, g, c]) => {
    const pct = (g / tot) * 100;
    return pct > 0.4 ? `<div class="mini-seg" style="width:${pct.toFixed(1)}%;background:${c}" title="${n}: ${pct.toFixed(1)}% (${fmtCO2(g)})"></div>` : "";
  }).join("");

  const legend = parts.map(([n, g, c]) => {
    const pct = (g / tot) * 100;
    return pct >= 1 ? `<span class="mini-leg-item"><i style="background:${c}"></i>${n} <b>${pct.toFixed(0)}%</b></span>` : "";
  }).join("");

  return `
    <div class="mini-breakdown">
      <div class="mini-bar">${segs}</div>
      <div class="mini-legend">${legend}</div>
    </div>`;
}

function renderAccViolin(sim, p, scale) {
  const scaleMin = scale.min;
  const scaleMax = scale.max;
  const range = scaleMax - scaleMin || 1;
  const toX = v => 10 + Math.max(0, Math.min(1, (v - scaleMin) / range)) * 260;

  const xMin = toX(sim.minAcc);
  const xMax = toX(sim.maxAcc);
  const L = Math.max(4, xMax - xMin);
  const y0 = 15;
  const Hmax = 9.5;

  const span = Math.max(0.001, sim.maxAcc - sim.minAcc);
  const uAvg = Math.max(0.05, Math.min(0.95, (sim.avgAcc - sim.minAcc) / span));

  const N = 20;
  const rawH = [];
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    const d = u - uAvg;
    const gauss = Math.exp(-(d * d) / (2 * 0.32 * 0.32));
    const sinTaper = Math.pow(Math.sin(Math.PI * Math.max(0, Math.min(1, u))), 0.6);
    rawH.push(gauss * sinTaper);
  }
  const peak = Math.max(...rawH) || 1;

  const topPts = [];
  const botPts = [];
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    const px = xMin + u * L;
    const h = (rawH[i] / peak) * Hmax;
    topPts.push(`${px.toFixed(1)},${(y0 - h).toFixed(1)}`);
    botPts.push(`${px.toFixed(1)},${(y0 + h).toFixed(1)}`);
  }

  const pathD = `M ${xMin.toFixed(1)} ${y0} ` +
    topPts.map(pt => `L ${pt}`).join(" ") +
    ` L ${xMax.toFixed(1)} ${y0} ` +
    botPts.slice().reverse().map(pt => `L ${pt}`).join(" ") +
    ` Z`;

  return `
    <div class="acc-violin-wrap">
      <svg class="acc-violin-svg" viewBox="0 0 280 30" preserveAspectRatio="none" role="img" aria-label="Accuracy distribution for ${esc(p.name)}" title="Accuracy distribution (${scaleMin}% – ${scaleMax}% scale)">
        <!-- Scale axis guide -->
        <line x1="10" y1="15" x2="270" y2="15" stroke="var(--line)" stroke-width="1" stroke-dasharray="2 3"/>
        <!-- Pure violin distribution envelope in neutral greyscale -->
        <path d="${pathD}" fill="#555f58" fill-opacity="0.32" stroke="#222c26" stroke-width="1.6" stroke-linejoin="round"/>
        <!-- Centerline spine -->
        <line x1="${xMin.toFixed(1)}" y1="15" x2="${xMax.toFixed(1)}" y2="15" stroke="#17231d" stroke-width="1.6" stroke-linecap="round"/>
      </svg>
      <div class="acc-gauge-labels">
        <span>${scaleMin}%</span>
        <span>${scaleMax}%</span>
      </div>
    </div>`;
}

function renderBillsGrid(rows, s) {
  if (!$("bills-grid") || !rows.length) return;

  const minJ = Math.min(...rows.map(o => o.sim.totalJ));
  const minCarbon = Math.min(...rows.map(o => o.sim.carbonTotalG));
  const greenest = rows.find(o => o.sim.totalJ === minJ);
  const line = (k, v, cls = "") => `<div class="rline ${cls}"><span class="k">${k}</span><span class="v">${v}</span></div>`;

  // rank on Wh per 1k correct predictions
  const order = [...rows].sort((a, b) => a.sim.whPerK - b.sim.whPerK);
  const n = rows.length;

  // Dynamically zoom accuracy scale based on performance across all pipelines
  const allMin = Math.min(...rows.map(r => r.sim.minAcc));
  const allMax = Math.max(...rows.map(r => r.sim.maxAcc));
  let scaleMin = Math.max(0, Math.floor((allMin - 2) / 5) * 5);
  let scaleMax = Math.min(100, Math.ceil((allMax + 2) / 5) * 5);
  if (scaleMax - scaleMin < 10) {
    if (scaleMax <= 95) scaleMax += 5;
    else scaleMin = Math.max(0, scaleMin - 5);
  }
  const accScale = { min: scaleMin, max: scaleMax };

  $("bills-grid").innerHTML = rows.map(({ p, sim }) => {
    const isSel = p.slot === selSlot;

    const rank = order.findIndex(o => o.p.slot === p.slot);
    const grade = n === 1 ? 0 : Math.round((rank / (n - 1)) * 4);

    const isEnergyOpen = allDetailsOpen || openDetails.has(`${p.slot}-energy`);
    const isPerfOpen = allDetailsOpen || openDetails.has(`${p.slot}-perf`);
    const isSciOpen = allDetailsOpen || openDetails.has(`${p.slot}-sci`);

    return `
      <div class="bill-card${isSel ? " sel" : ""}" data-slot="${p.slot}">
        <div class="bill">
          <div class="r-head">
            <div class="t"><span class="dot" style="background:${color(p)}"></span>${esc(p.name)}</div>
          </div>

          <!-- Part 1: ENERGY (Equal Prominence) -->
          <div class="bill-part">
            <div class="part-title"><span>Energy</span></div>
            <details class="bill-details" data-key="${p.slot}-energy"${isEnergyOpen ? " open" : ""}>
              <summary class="bill-summary">Detailed listing</summary>
              <div class="bill-details-body">
                <div class="rsub">Development</div>
                ${line("Initial Training Energy", fmtWh(sim.initJ))}
                ${line(`Evaluation Energy (${fmtNum(sim.evalPreds)} preds)`, fmtWh(sim.evalJ))}
                ${line("Development CPU Time", fmtTime(sim.devCpuS))}
                <div class="rsub">Deployment</div>
                ${line(`Retraining Energy (${sim.retrains}×)`, fmtWh(sim.retrainJ))}
                ${line(`Inference Energy (${fmtNum(s.totalPreds)} preds)`, fmtWh(sim.infJ))}
                ${line("Deployment CPU Time", fmtTime(sim.deployCpuS))}
                <div class="rsub">Energy Share</div>
                ${renderMiniBreakdown(sim)}
              </div>
            </details>
            <div class="bill-stat-box">
              <div class="stat-meta">
                <div class="stat-label">Total Energy</div>
                <div class="stat-sub">${fmtTime(sim.cpuS)} Total CPU Time</div>
              </div>
              <div class="stat-val">${fmtWh(sim.totalJ)}</div>
            </div>
          </div>

          <!-- Part 2: PERFORMANCE (Equal Prominence) -->
          <div class="bill-part">
            <div class="part-title"><span>Performance</span></div>
            <details class="bill-details" data-key="${p.slot}-perf"${isPerfOpen ? " open" : ""}>
              <summary class="bill-summary">Detailed listing</summary>
              <div class="bill-details-body">
                <div class="rsub">Accuracy Distribution</div>
                ${renderAccViolin(sim, p, accScale)}
                <div class="rsub">Operational Accuracy</div>
                ${line("Max Accuracy", fmtPct(sim.maxAcc))}
                ${line("Min Accuracy", fmtPct(sim.minAcc))}
                ${line("Operational Accuracy", fmtPct(sim.avgAcc))}
                ${line("Correct Predictions", fmtNum(sim.correctPreds))}
              </div>
            </details>
            <div class="bill-stat-box">
              <div class="stat-meta">
                <div class="stat-label">Operational Accuracy</div>
                <div class="stat-sub">${fmtPct(sim.minAcc)} – ${fmtPct(sim.maxAcc)} range</div>
              </div>
              <div class="stat-val">${fmtPct(sim.avgAcc)}</div>
            </div>
          </div>

          <!-- Part 3: SCI FOR AI (Equal Prominence) -->
          <div class="bill-part">
            <div class="part-title"><span>SCI for AI</span></div>
            <details class="bill-details" data-key="${p.slot}-sci"${isSciOpen ? " open" : ""}>
              <summary class="bill-summary">Detailed listing</summary>
              <div class="bill-details-body">
                <div class="rsub">Lifecycle Emissions (O &amp; M)</div>
                ${line("Operational Carbon (O)", fmtCO2(sim.opTotalG))}
                ${line("Embodied Carbon (M)", fmtCO2(sim.embTotalG))}
                ${line("Lifecycle Carbon", fmtCO2(sim.carbonTotalG))}
                ${sim.retrains > 0 ? line(`Retraining Carbon (${sim.retrains}×)`, fmtCO2(sim.retrainCarbonG)) : ""}
                <div class="rsub">Carbon Share (O vs M)</div>
                ${renderMiniCarbonBreakdown(sim)}
                <div class="rsub">Functional Units (R)</div>
                ${line("Consumer SCI (QA)", fmtSciRate(sim.sciEffective) + " (per 1k correct)")}
                ${line("Consumer SCI", fmtSciRate(sim.sciConsumer) + " (per 1k preds)")}
                ${line("Provider SCI (per model)", fmtCO2(sim.sciProvider) + " / model")}
              </div>
            </details>
            <div class="bill-stat-box">
              <div class="stat-meta">
                <div class="stat-label">Consumer SCI (QA)</div>
                <div class="stat-sub">per 1,000 correct preds</div>
              </div>
              <div class="stat-val">${fmtSciRate(sim.sciEffective)}</div>
            </div>
            <div class="sci-units-mini">
              <div class="sci-mini-row"><span class="k">Consumer SCI:</span><span class="v">${fmtSciRate(sim.sciConsumer)} (per 1k preds)</span></div>
              <div class="sci-mini-row"><span class="k">Provider SCI:</span><span class="v">${fmtCO2(sim.sciProvider)} / model</span></div>
            </div>
          </div>

          <!-- Card Footer -->
          <div class="bill-foot">
            ${line("Energy / 1k Correct", sim.whPerK < 0.001 ? sim.whPerK.toExponential(2) : sim.whPerK.toPrecision(3))}
            ${line("Consumer SCI (QA)", fmtSciRate(sim.sciEffective))}
            <div class="band">${BAND.map(([l, c], i) => `<span style="background:${c}" class="${i === grade ? "on" : ""}">${l}</span>`).join("")}</div>
            <div class="band-note">Grade relative to candidate models</div>
          </div>
        </div>
      </div>`;
  }).join("");
}

$("bills-grid").addEventListener("click", ev => {
  if (ev.target.closest("summary") || ev.target.closest(".bill-details")) return;
  const card = ev.target.closest("[data-slot]");
  if (card) selectPipeline(+card.dataset.slot);
});

$("bills-grid").addEventListener("toggle", ev => {
  const details = ev.target;
  if (!details || !details.dataset.key) return;
  if (details.open) {
    openDetails.add(details.dataset.key);
  } else {
    openDetails.delete(details.dataset.key);
    allDetailsOpen = false;
    updateToggleBtnText();
  }
}, true);

if ($("btn-toggle-details")) {
  $("btn-toggle-details").addEventListener("click", () => {
    allDetailsOpen = !allDetailsOpen;
    if (allDetailsOpen) {
      pipelines.forEach(p => {
        openDetails.add(`${p.slot}-energy`);
        openDetails.add(`${p.slot}-perf`);
        openDetails.add(`${p.slot}-sci`);
      });
    } else {
      openDetails.clear();
    }
    updateToggleBtnText();
    if (last) renderBillsGrid(last.rows, last.s);
  });
}

// ---------------------------------------------------------------- charts (x = continuous time)
const niceStep = (range, n = 5) => {
  const raw = range / n, mag = 10 ** Math.floor(Math.log10(raw || 1));
  return [1, 2, 2.5, 5, 10].map(f => f * mag).find(st => st >= raw) || raw;
};

function drawTimeChart(svg, cfg) {
  const W = svg.clientWidth, H = svg.clientHeight;
  if (!W) return;
  const M = { l: 58, r: 14, t: 10, b: 36 }, iw = W - M.l - M.r, ih = H - M.t - M.b;
  const x = t => M.l + (t / cfg.tMax) * iw;
  const y = v => M.t + ih - ((v - cfg.yLo) / (cfg.yHi - cfg.yLo)) * ih;

  let g = "";
  for (let v = cfg.yLo; v <= cfg.yHi + 1e-9; v += cfg.yStep) {
    g += `<line x1="${M.l}" x2="${M.l + iw}" y1="${y(v)}" y2="${y(v)}" stroke="#d8ded2" stroke-dasharray="2 3"/>` +
      `<text x="${M.l - 8}" y="${y(v)}" text-anchor="end" dominant-baseline="middle">${cfg.yFmt(v)}</text>`;
  }
  const xs = niceStep(cfg.tMax, Math.max(3, Math.floor(iw / 90)));
  for (let v = 0; v <= cfg.tMax + 1e-9; v += xs) {
    g += `<text x="${x(v)}" y="${M.t + ih + 16}" text-anchor="middle">${+v.toFixed(1)}</text>`;
  }
  g += `<line x1="${M.l}" x2="${M.l + iw}" y1="${M.t + ih + 0.5}" y2="${M.t + ih + 0.5}" stroke="#17231d" stroke-width="1.5"/>`;
  g += `<text x="${M.l + iw / 2}" y="${H - 3}" text-anchor="middle">time (${cfg.uName})</text>`;
  g += `<g clip-path="none">${cfg.body(x, y, { M, iw, ih })}</g><g class="hover"></g>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = g;
  svg._frame = { cfg, x, y, M, iw, ih };

  if (!svg._bound) {
    svg._bound = true;
    const tip = svg.parentElement.querySelector(".tooltip"), hov = () => svg.querySelector(".hover");
    const hide = () => { tip.classList.remove("on"); if (hov()) hov().innerHTML = ""; };
    svg.addEventListener("mouseleave", hide);
    svg.addEventListener("mousemove", ev => {
      const f = svg._frame;
      if (!f) return;
      const mx = ev.clientX - svg.getBoundingClientRect().left;
      if (mx < f.M.l - 6 || mx > f.M.l + f.iw + 6) return hide();
      const t = Math.max(0, Math.min(f.cfg.tMax, ((mx - f.M.l) / f.iw) * f.cfg.tMax));
      const h = f.cfg.hover(t);
      hov().innerHTML = `<line x1="${f.x(t)}" x2="${f.x(t)}" y1="${f.M.t}" y2="${f.M.t + f.ih}" stroke="#17231d" stroke-dasharray="3 3"/>` +
        h.marks.map(m => `<circle cx="${f.x(t)}" cy="${f.y(m.v)}" r="4.5" style="fill:${m.color}" stroke="#fff" stroke-width="2"/>`).join("");
      tip.innerHTML = `<div class="t">${h.title}</div>` + h.rows.map(r =>
        `<div class="r"><span class="dot" style="background:${r.color}"></span><span>${esc(r.name)}</span><b>${r.value}</b></div>`).join("");
      tip.classList.add("on");
      const ww = svg.parentElement.clientWidth, tw = tip.offsetWidth;
      let left = f.x(t) + 14;
      if (left + tw > ww) left = f.x(t) - tw - 14;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = f.M.t + "px";
    });
  }
}

function accAt(r, s, tDay) {
  const cycle = r.sim.cycle;
  const daysSinceLast = cycle < s.days ? tDay % cycle : tDay;
  return Math.max(s.floor, r.p.acc - r.sim.dailyDrop * daysSinceLast);
}

function energyAt(r, s, tDay) {
  const devWh = r.sim.devJ / 3600;
  const trainWh = (r.p.train * s.watts) / 3600;
  const dailyInfWh = (r.p.inf / 1000) * s.traffic * s.watts / 3600;
  const retrainsBefore = r.sim.cycle < s.days ? Math.floor(tDay / r.sim.cycle) : 0;
  return devWh + dailyInfWh * tDay + retrainsBefore * trainWh;
}

function carbonAt(r, s, tDay) {
  const wh = energyAt(r, s, tDay);
  const kwh = (wh / 1000) * s.pue;
  const opG = kwh * s.gridG;
  const retrainsBefore = r.sim.cycle < s.days ? Math.floor(tDay / r.sim.cycle) : 0;
  const devCpuS = r.sim.devCpuS;
  const dailyInfCpuS = (r.p.inf / 1000) * s.traffic;
  const totalCpuS = devCpuS + dailyInfCpuS * tDay + retrainsBefore * r.p.train;
  const util = Math.max(0.1, s.hardwareUtil / 100);
  const embG = totalCpuS * (s.embGPerSec / util);
  return opG + embG;
}

function renderAccChart(rows, s) {
  const isMonths = s.months > 2;
  const tMax = isMonths ? s.months : s.days;
  const uName = isMonths ? "months" : "days";
  const toT = d => isMonths ? d / 30 : d;
  const fromT = t => isMonths ? t * 30 : t;

  const sel = rows.find(r => r.p.slot === selSlot) || rows[0];
  const thr = s.policy === "absolute" ? s.tau : null;
  const all = rows.map(r => r.p.acc).concat(rows.map(r => r.sim.minAcc)).concat(thr == null ? [] : [thr]);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = Math.max(1, (hi - lo) * 0.06), step = niceStep(Math.max(1, hi - lo + 2 * pad));
  lo = Math.max(0, Math.floor((lo - pad) / step) * step);
  hi = Math.min(100, Math.ceil((hi + pad) / step) * step);

  drawTimeChart($("chart"), {
    tMax, uName, yLo: lo, yHi: hi, yStep: step, yFmt: v => `${+v.toFixed(1)}%`,
    body: (x, y, { M, iw }) => {
      let g = "";
      if (s.floor > lo && s.floor < hi) {
        g += `<line x1="${M.l}" x2="${M.l + iw}" y1="${y(s.floor)}" y2="${y(s.floor)}" stroke="#5a685e" stroke-dasharray="1 4" stroke-linecap="round"/>`;
      }
      if (thr != null) {
        const c = "#5a685e";
        g += `<line x1="${M.l}" x2="${M.l + iw}" y1="${y(thr)}" y2="${y(thr)}" style="stroke:${c}" stroke-width="1.5" stroke-dasharray="6 4"/>`;
      }

      const path = (r, w, op) => {
        const p = r.p, sim = r.sim;
        let d = "";
        if (sim.retrains <= 60) {
          const cycle = sim.cycle;
          const daysToFloor = sim.dailyDrop > 0 ? (p.acc - s.floor) / sim.dailyDrop : Infinity;
          for (let k = 0; k < sim.retrains; k++) {
            const d0 = k * cycle, d1 = (k + 1) * cycle;
            const x0 = x(toT(d0)), x1 = x(toT(d1));
            if (k === 0) d += `M${x0.toFixed(1)},${y(p.acc).toFixed(1)}`;
            else d += `L${x0.toFixed(1)},${y(p.acc).toFixed(1)}`;
            if (cycle <= daysToFloor) {
              const accEnd = p.acc - sim.dailyDrop * cycle;
              d += `L${x1.toFixed(1)},${y(accEnd).toFixed(1)}V${y(p.acc).toFixed(1)}`;
            } else {
              const xFloor = x(toT(d0 + daysToFloor));
              d += `L${xFloor.toFixed(1)},${y(s.floor).toFixed(1)}L${x1.toFixed(1)},${y(s.floor).toFixed(1)}V${y(p.acc).toFixed(1)}`;
            }
          }
          const remStart = sim.retrains * cycle;
          const xRem0 = x(toT(remStart)), xRem1 = x(toT(s.days));
          const remDur = s.days - remStart;
          if (sim.retrains === 0) d += `M${xRem0.toFixed(1)},${y(p.acc).toFixed(1)}`;
          if (remDur <= daysToFloor) {
            const accEnd = p.acc - sim.dailyDrop * remDur;
            d += `L${xRem1.toFixed(1)},${y(accEnd).toFixed(1)}`;
          } else {
            const xFloor = x(toT(remStart + daysToFloor));
            d += `L${xFloor.toFixed(1)},${y(s.floor).toFixed(1)}L${xRem1.toFixed(1)},${y(s.floor).toFixed(1)}`;
          }
        } else {
          const N = 250;
          for (let i = 0; i <= N; i++) {
            const curDay = (i / N) * s.days;
            const curAcc = accAt(r, s, curDay);
            const px = x(toT(curDay)), py = y(curAcc);
            d += (i === 0 ? `M` : `L`) + `${px.toFixed(1)},${py.toFixed(1)}`;
          }
        }
        return `<path d="${d}" fill="none" style="stroke:${color(p)}" stroke-width="${w}" stroke-linejoin="round" opacity="${op}"/>`;
      };

      rows.filter(r => r !== sel).forEach(r => { g += path(r, 1.5, 0.45); });
      g += path(sel, 2.75, 1);

      if (sel.sim.retrains > 0 && sel.sim.retrains <= 30) {
        for (let k = 1; k <= sel.sim.retrains; k++) {
          const d = k * sel.sim.cycle;
          const accEnd = Math.max(s.floor, sel.p.acc - sel.sim.dailyDrop * sel.sim.cycle);
          g += `<circle cx="${x(toT(d))}" cy="${y(accEnd)}" r="4" style="fill:${color(sel.p)}" stroke="#fff" stroke-width="2"/>`;
        }
      }
      return g;
    },
    hover: t => {
      const tDay = fromT(t);
      const sorted = [...rows].sort((a, c) => accAt(c, s, tDay) - accAt(a, s, tDay));
      const title = isMonths
        ? `Month ${+(tDay / 30).toFixed(1)} (${Math.round(tDay)}d) · ${fmtNum(tDay * s.traffic)} preds`
        : `Day ${Math.round(tDay)} · ${fmtNum(tDay * s.traffic)} preds`;
      return {
        title,
        rows: sorted.map(r => {
          const cur = accAt(r, s, tDay);
          const nearRetrain = r.sim.retrains > 0 && r.sim.cycle < s.days && (tDay % r.sim.cycle <= (isMonths ? 3 : 0.5));
          return { color: color(r.p), name: r.p.name + (nearRetrain ? " ↺" : ""), value: fmtPct(cur) };
        }),
        marks: rows.map(r => ({ v: accAt(r, s, tDay), color: color(r.p) })),
      };
    },
  });

  $("legend").innerHTML = rows.map(r => `<span class="item"><span class="ln${r === sel ? " sel" : ""}" style="background:${color(r.p)};${r === sel ? "" : "opacity:.55"}"></span>${esc(r.p.name)}</span>`).join("") +
    (thr != null ? `<span class="item"><span class="dash"></span>threshold ${+s.tau.toFixed(1)}%</span>` : "");
  const under = s.policy === "absolute" ? rows.filter(r => r.p.acc < s.tau).map(r => r.p.name) : [];
  $("chart-sub").textContent = `Service lifecycle over ${fmtMonths(s.months)} at ${fmtNum(s.traffic)} predictions/day. ` +
    `Accuracy drifts continuously with every data point. The selected candidate model is drawn with a bold stroke.` +
    (under.length ? ` ${under.join(", ")} start${under.length === 1 ? "s" : ""} below the threshold` +
      (s.subthresholdRetrain ? ` (retraining every ${s.subthresholdDays} days).` : ` (sub-threshold retraining disabled).`) : "");
}

function renderEnergyChart(rows, s) {
  if (!$("echart")) return;
  const isMonths = s.months > 2;
  const tMax = isMonths ? s.months : s.days;
  const uName = isMonths ? "months" : "days";
  const toT = d => isMonths ? d / 30 : d;
  const fromT = t => isMonths ? t * 30 : t;

  const sel = rows.find(r => r.p.slot === selSlot) || rows[0];
  const maxWh = Math.max(...rows.map(r => r.sim.totalJ)) / 3600;
  const step = niceStep(maxWh * 1.04 || 1);
  const hi = Math.ceil((maxWh * 1.04) / step) * step || step;
  const whFmt = v => v === 0 ? "0" : v < 0.1 ? +v.toPrecision(2) + "" : +v.toFixed(v < 10 ? 1 : 0) + "";

  drawTimeChart($("echart"), {
    tMax, uName, yLo: 0, yHi: hi, yStep: step, yFmt: v => `${whFmt(v)} Wh`,
    body: (x, y) => {
      const path = (r, w, op) => {
        const p = r.p, sim = r.sim;
        const devWh = sim.devJ / 3600;
        const trainWh = p.train * s.watts / 3600;
        const dailyInfWh = (p.inf / 1000) * s.traffic * s.watts / 3600;
        let d = "";
        if (sim.retrains <= 60) {
          let cumWh = devWh;
          d = `M${x(0).toFixed(1)},${y(0).toFixed(1)}V${y(cumWh).toFixed(1)}`;
          const cycle = sim.cycle;
          for (let k = 0; k < sim.retrains; k++) {
            cumWh += cycle * dailyInfWh;
            d += `L${x(toT((k + 1) * cycle)).toFixed(1)},${y(cumWh).toFixed(1)}`;
            cumWh += trainWh;
            d += `V${y(cumWh).toFixed(1)}`;
          }
          const remStart = sim.retrains * cycle;
          const remDur = s.days - remStart;
          if (remDur > 0) {
            cumWh += remDur * dailyInfWh;
            d += `L${x(toT(s.days)).toFixed(1)},${y(cumWh).toFixed(1)}`;
          }
        } else {
          const N = 250;
          for (let i = 0; i <= N; i++) {
            const curDay = (i / N) * s.days;
            const curWh = energyAt(r, s, curDay);
            const px = x(toT(curDay)), py = y(curWh);
            d += (i === 0 ? `M` : `L`) + `${px.toFixed(1)},${py.toFixed(1)}`;
          }
        }
        return `<path d="${d}" fill="none" style="stroke:${color(p)}" stroke-width="${w}" stroke-linejoin="round" opacity="${op}"/>`;
      };

      let g = "";
      rows.filter(r => r !== sel).forEach(r => { g += path(r, 1.5, 0.45); });
      return g + path(sel, 2.75, 1);
    },
    hover: t => {
      const tDay = fromT(t);
      const vals = rows.map(r => ({ r, wh: energyAt(r, s, tDay) })).sort((a, c) => c.wh - a.wh);
      const title = isMonths
        ? `Month ${+(tDay / 30).toFixed(1)} (${Math.round(tDay)}d) · energy so far`
        : `Day ${Math.round(tDay)} · energy so far`;
      return {
        title,
        rows: vals.map(({ r, wh }) => ({ color: color(r.p), name: r.p.name, value: fmtWh(wh * 3600) })),
        marks: vals.map(({ r, wh }) => ({ v: wh, color: color(r.p) })),
      };
    },
  });

  $("elegend").innerHTML = rows.map(r => `<span class="item"><span class="ln${r === sel ? " sel" : ""}" style="background:${color(r.p)};${r === sel ? "" : "opacity:.55"}"></span>${esc(r.p.name)} · ${fmtWh(r.sim.totalJ)}</span>`).join("");
  $("echart-sub").textContent = `Cumulative electrical energy across lifecycle. Starts at initial training, climbs with daily inference, and steps up by training cost on retrain. Over ${fmtMonths(s.months)}.`;
}

function renderCarbonChart(rows, s) {
  if (!$("cchart")) return;
  const isMonths = s.months > 2;
  const tMax = isMonths ? s.months : s.days;
  const uName = isMonths ? "months" : "days";
  const toT = d => isMonths ? d / 30 : d;
  const fromT = t => isMonths ? t * 30 : t;

  const sel = rows.find(r => r.p.slot === selSlot) || rows[0];
  const maxG = Math.max(...rows.map(r => r.sim.carbonTotalG));
  const step = niceStep(maxG * 1.04 || 1);
  const hi = Math.ceil((maxG * 1.04) / step) * step || step;
  const gFmt = v => v === 0 ? "0" : v < 0.1 ? +v.toPrecision(2) + "" : +v.toFixed(v < 10 ? 1 : 0) + "";

  drawTimeChart($("cchart"), {
    tMax, uName, yLo: 0, yHi: hi, yStep: step, yFmt: v => `${gFmt(v)} g`,
    body: (x, y) => {
      const path = (r, w, op) => {
        const p = r.p;
        let d = "";
        const N = 250;
        for (let i = 0; i <= N; i++) {
          const curDay = (i / N) * s.days;
          const curG = carbonAt(r, s, curDay);
          const px = x(toT(curDay)), py = y(curG);
          d += (i === 0 ? `M` : `L`) + `${px.toFixed(1)},${py.toFixed(1)}`;
        }
        return `<path d="${d}" fill="none" style="stroke:${color(p)}" stroke-width="${w}" stroke-linejoin="round" opacity="${op}"/>`;
      };

      let g = "";
      rows.filter(r => r !== sel).forEach(r => { g += path(r, 1.5, 0.45); });
      return g + path(sel, 2.75, 1);
    },
    hover: t => {
      const tDay = fromT(t);
      const vals = rows.map(r => ({ r, gVal: carbonAt(r, s, tDay) })).sort((a, c) => c.gVal - a.gVal);
      const title = isMonths
        ? `Month ${+(tDay / 30).toFixed(1)} (${Math.round(tDay)}d) · carbon so far`
        : `Day ${Math.round(tDay)} · carbon so far`;
      return {
        title,
        rows: vals.map(({ r, gVal }) => ({ color: color(r.p), name: r.p.name, value: fmtCO2(gVal) })),
        marks: vals.map(({ r, gVal }) => ({ v: gVal, color: color(r.p) })),
      };
    },
  });

  $("clegend").innerHTML = rows.map(r => `<span class="item"><span class="ln${r === sel ? " sel" : ""}" style="background:${color(r.p)};${r === sel ? "" : "opacity:.55"}"></span>${esc(r.p.name)} · ${fmtCO2(r.sim.carbonTotalG)}</span>`).join("");
  $("cchart-sub").textContent = `Cumulative carbon emissions (operational electrical emissions + embodied hardware manufacturing) at ${s.gridG} gCO₂e/kWh and ${s.hardwareUtil}% hardware utilization. Over ${fmtMonths(s.months)}.`;
}

function renderCharts(rows, s) {
  renderAccChart(rows, s);
  renderEnergyChart(rows, s);
  renderCarbonChart(rows, s);
}

// ---------------------------------------------------------------- comparison table (03 Analysis)
function renderComparisonTable(rows, s) {
  if (!$("results")) return;

  const maxDevAcc = Math.max(...rows.map(r => r.p.acc));
  const minRetrains = Math.min(...rows.map(r => r.sim.retrains));
  const maxAvgAcc = Math.max(...rows.map(r => r.sim.avgAcc));
  const minTotal = Math.min(...rows.map(r => r.sim.totalJ));
  const minCarbon = Math.min(...rows.map(r => r.sim.carbonTotalG));
  const minSciEffective = Math.min(...rows.map(r => r.sim.sciEffective));

  const sorted = [...rows].sort((a, b) => a.sim.totalJ - b.sim.totalJ);

  $("results").innerHTML = `
    <thead>
      <tr>
        <th>Model</th>
        <th>Development Accuracy</th>
        <th>Operational Accuracy</th>
        <th>Retrains</th>
        <th>Total Energy</th>
        <th>Lifecycle Carbon</th>
        <th>Consumer SCI (QA)</th>
      </tr>
    </thead>
    <tbody>
      ${sorted.map(r => {
        const isBestDevAcc = Math.abs(r.p.acc - maxDevAcc) < 1e-6;
        const isBestRetrains = r.sim.retrains === minRetrains;
        const isBestAcc = Math.abs(r.sim.avgAcc - maxAvgAcc) < 1e-6;
        const isBestEnergy = Math.abs(r.sim.totalJ - minTotal) < 1e-6;
        const isBestCarbon = Math.abs(r.sim.carbonTotalG - minCarbon) < 1e-6;
        const isBestSciEffective = Math.abs(r.sim.sciEffective - minSciEffective) < 1e-6;

        return `
          <tr class="${r.p.slot === selSlot ? "sel" : ""}" data-slot="${r.p.slot}" style="cursor:pointer">
            <td><span class="cell-name"><span class="dot" style="background:${color(r.p)}"></span>${esc(r.p.name)}</span></td>
            <td class="${isBestDevAcc ? "best" : ""}">${isBestDevAcc ? `<b>${fmtPct(r.p.acc)}</b>` : fmtPct(r.p.acc)}</td>
            <td class="${isBestAcc ? "best" : ""}">${isBestAcc ? `<b>${fmtPct(r.sim.avgAcc)}</b>` : fmtPct(r.sim.avgAcc)}</td>
            <td class="${isBestRetrains ? "best" : ""}">${isBestRetrains ? `<b>${r.sim.retrains}</b>` : r.sim.retrains}</td>
            <td class="${isBestEnergy ? "best" : ""}">${isBestEnergy ? `<b>${fmtWh(r.sim.totalJ)}</b>` : fmtWh(r.sim.totalJ)}</td>
            <td class="${isBestCarbon ? "best" : ""}">${isBestCarbon ? `<b>${fmtCO2(r.sim.carbonTotalG)}</b>` : fmtCO2(r.sim.carbonTotalG)}</td>
            <td class="${isBestSciEffective ? "best" : ""}">${isBestSciEffective ? `<b>${fmtSciRate(r.sim.sciEffective)}</b>` : fmtSciRate(r.sim.sciEffective)}</td>
          </tr>`;
      }).join("")}
    </tbody>`;

  $("table-sub").textContent = `Side-by-side lifecycle impact of candidate models under drift. Notice how a model with lower initial benchmark accuracy can achieve lower total lifecycle impact if its robustness avoids costly retraining. The winning value for each metric is bolded in green. Click any row to highlight.`;
}

// ---------------------------------------------------------------- decision rules (04 Decision)
function renderDecisionTab(rows, s) {
  if (!$("decision-rule-opts") || !rows.length) return;

  // Baseline selection: Always the pipeline according to the first rule (Highest development accuracy)
  const sortedByDevAcc = [...rows].sort((a, b) => b.p.acc - a.p.acc);
  const baseline = sortedByDevAcc[0];
  const baseTotalJ = baseline.sim.totalJ;
  const baseAvgAcc = baseline.sim.avgAcc;

  const DECISION_RULES = [
    {
      num: 1,
      name: "Highest development accuracy",
      tag: "AutoML Baseline",
      isBaseline: true,
      desc: "Selects the model with the highest test accuracy on static validation benchmarks prior to deployment (the conventional AutoML heuristic, blind to operational drift and retraining).",
      pick: () => sortedByDevAcc[0]
    },
    {
      num: 2,
      name: "Highest operational accuracy",
      tag: "Operational Accuracy",
      isBaseline: false,
      desc: "Selects the model that maintains the highest operational accuracy across all production inferences under continuous drift.",
      pick: () => [...rows].sort((a, b) => b.sim.avgAcc - a.sim.avgAcc)[0]
    },
    {
      num: 3,
      name: "Lowest Consumer SCI (QA)",
      tag: "Consumer SCI (QA)",
      isBaseline: false,
      desc: "Selects the model with the lowest quality-adjusted carbon cost per 1,000 correct predictions delivered (R = 1,000 correct preds), directly penalizing model mistakes and accuracy loss under drift.",
      pick: () => [...rows].sort((a, b) => a.sim.sciEffective - b.sim.sciEffective)[0]
    }
  ];

  if (selectedDecisionRule < 0 || selectedDecisionRule >= DECISION_RULES.length) {
    selectedDecisionRule = 0;
  }

  // 1. Render rule list with definitions only (no results inside each rule)
  $("decision-rule-opts").innerHTML = DECISION_RULES.map((rule, idx) => `
    <button class="opt" type="button" data-rule="${idx}" aria-pressed="${idx === selectedDecisionRule}">
      <div class="rule-head">
        <span class="rule-title">Rule ${rule.num} · ${rule.name}</span>
        <span class="rule-badge${rule.isBaseline ? " base" : ""}">${rule.tag}</span>
      </div>
      <div class="rule-desc">${rule.desc}</div>
    </button>`).join("");

  // 2. Render the single result box based on the selected rule
  const rule = DECISION_RULES[selectedDecisionRule];
  const chosen = rule.pick();
  const isSameAsBase = chosen.p.slot === baseline.p.slot;
  const dJ = chosen.sim.totalJ - baseTotalJ;
  const dAcc = chosen.sim.avgAcc - baseAvgAcc;
  const pctJ = baseTotalJ > 0 ? (Math.abs(dJ) / baseTotalJ) * 100 : 0;
  const baseCarbonG = baseline.sim.carbonTotalG;
  const dCarbon = chosen.sim.carbonTotalG - baseCarbonG;
  const pctCarbon = baseCarbonG > 0 ? (Math.abs(dCarbon) / baseCarbonG) * 100 : 0;
  const baseSciEff = baseline.sim.sciEffective;
  const dSciEff = chosen.sim.sciEffective - baseSciEff;
  const pctSciEff = baseSciEff > 0 ? (Math.abs(dSciEff) / baseSciEff) * 100 : 0;

  let energyCompHtml;
  let accCompHtml;
  let sciCompHtml;

  if (rule.isBaseline) {
    energyCompHtml = `<span class="dsingle-comp base">★ AutoML Baseline</span>`;
    accCompHtml = `<span class="dsingle-comp base">★ AutoML Baseline</span>`;
    sciCompHtml = `<span class="dsingle-comp base">★ AutoML Baseline</span>`;
  } else if (isSameAsBase) {
    energyCompHtml = `<span class="dsingle-comp same">= Same model as AutoML baseline</span>`;
    accCompHtml = `<span class="dsingle-comp same">= Same model as AutoML baseline</span>`;
    sciCompHtml = `<span class="dsingle-comp same">= Same model as AutoML baseline</span>`;
  } else {
    // Energy comparison vs baseline
    if (dJ < -1e-6) {
      energyCompHtml = `<span class="dsingle-comp pos">↓ Saves ${fmtWh(-dJ)} (−${pctJ.toFixed(1)}%) vs baseline</span>`;
    } else if (dJ > 1e-6) {
      energyCompHtml = `<span class="dsingle-comp neg">↑ +${fmtWh(dJ)} (+${pctJ.toFixed(1)}%) more energy vs baseline</span>`;
    } else {
      energyCompHtml = `<span class="dsingle-comp same">= Equal energy to baseline</span>`;
    }

    // Consumer SCI (QA) comparison vs baseline
    if (dSciEff < -1e-6) {
      sciCompHtml = `<span class="dsingle-comp pos">↓ Saves ${pctSciEff.toFixed(1)}% vs baseline</span>`;
    } else if (dSciEff > 1e-6) {
      sciCompHtml = `<span class="dsingle-comp neg">↑ +${pctSciEff.toFixed(1)}% vs baseline</span>`;
    } else {
      sciCompHtml = `<span class="dsingle-comp same">= Equal to baseline</span>`;
    }

    // Accuracy comparison vs baseline
    if (dAcc > 1e-6) {
      accCompHtml = `<span class="dsingle-comp pos">↑ +${dAcc.toFixed(2)} pts higher vs baseline</span>`;
    } else if (dAcc < -1e-6) {
      accCompHtml = `<span class="dsingle-comp neg">↓ −${Math.abs(dAcc).toFixed(2)} pts lower vs baseline</span>`;
    } else {
      accCompHtml = `<span class="dsingle-comp same">= Equal accuracy to baseline</span>`;
    }
  }

  // Best models in each aspect across all candidates
  const bestEnergyRow = [...rows].sort((a, b) => a.sim.totalJ - b.sim.totalJ)[0];
  const bestAccRow = [...rows].sort((a, b) => b.sim.avgAcc - a.sim.avgAcc)[0];
  const bestSciRow = [...rows].sort((a, b) => a.sim.sciEffective - b.sim.sciEffective)[0];

  let energyCompBestHtml;
  if (Math.abs(chosen.sim.totalJ - bestEnergyRow.sim.totalJ) < 1e-6) {
    energyCompBestHtml = `<span class="dsingle-comp pos">★ #1 Lowest energy in fleet</span>`;
  } else {
    const dBestJ = chosen.sim.totalJ - bestEnergyRow.sim.totalJ;
    const pctBestJ = bestEnergyRow.sim.totalJ > 0 ? (dBestJ / bestEnergyRow.sim.totalJ) * 100 : 0;
    energyCompBestHtml = `<span class="dsingle-comp neg">↑ +${fmtWh(dBestJ)} (+${pctBestJ.toFixed(0)}%) vs ${esc(bestEnergyRow.p.name)}</span>`;
  }

  let accCompBestHtml;
  if (Math.abs(chosen.sim.avgAcc - bestAccRow.sim.avgAcc) < 1e-6) {
    accCompBestHtml = `<span class="dsingle-comp pos">★ #1 Highest accuracy in fleet</span>`;
  } else {
    const dBestAcc = chosen.sim.avgAcc - bestAccRow.sim.avgAcc;
    accCompBestHtml = `<span class="dsingle-comp neg">↓ −${Math.abs(dBestAcc).toFixed(2)} pts vs ${esc(bestAccRow.p.name)} (${fmtPct(bestAccRow.sim.avgAcc)})</span>`;
  }

  let sciCompBestHtml;
  if (Math.abs(chosen.sim.sciEffective - bestSciRow.sim.sciEffective) < 1e-6) {
    sciCompBestHtml = `<span class="dsingle-comp pos">★ #1 Lowest carbon / QA in fleet</span>`;
  } else {
    const dBestSci = chosen.sim.sciEffective - bestSciRow.sim.sciEffective;
    const pctBestSci = bestSciRow.sim.sciEffective > 0 ? (dBestSci / bestSciRow.sim.sciEffective) * 100 : 0;
    sciCompBestHtml = `<span class="dsingle-comp neg">↑ +${pctBestSci.toFixed(0)}% vs ${esc(bestSciRow.p.name)} (${fmtSciRate(bestSciRow.sim.sciEffective)})</span>`;
  }

  // Summary sentences: What the selected model optimizes for and how it relates to best performing models
  const isTopAcc = chosen.p.slot === bestAccRow.p.slot;
  const isTopSci = chosen.p.slot === bestSciRow.p.slot;

  let optimizesFor = "";
  let relationsText = "";
  let conclusionText = "";

  if (rule.num === 1) {
    optimizesFor = `<b>Optimizes for:</b> Offline benchmark accuracy on static pre-deployment validation data (${fmtPct(chosen.p.acc)}), assuming that development test performance persists indefinitely in production.`;
    
    relationsText = `<b>Relation to top performers:</b> ${isTopAcc 
      ? `In this scenario, it also achieves the highest operational accuracy (${fmtPct(chosen.sim.avgAcc)}).` 
      : `Under real-world drift, its live accuracy drops to an average of <b>${fmtPct(chosen.sim.avgAcc)}</b>, falling behind the accuracy leader (<b>${esc(bestAccRow.p.name)}</b> at <b>${fmtPct(bestAccRow.sim.avgAcc)}</b>).`} ${isTopSci 
      ? `It also matches the best Consumer SCI (QA) score (${fmtSciRate(chosen.sim.sciEffective)}).` 
      : `In carbon efficiency, it incurs a Consumer SCI (QA) of <b>${fmtSciRate(chosen.sim.sciEffective)}</b>, which is significantly higher than the greenest model (<b>${esc(bestSciRow.p.name)}</b> at <b>${fmtSciRate(bestSciRow.sim.sciEffective)}</b>, an efficiency penalty of +${((chosen.sim.sciEffective / bestSciRow.sim.sciEffective - 1) * 100).toFixed(0)}%).`}`;

    conclusionText = `<b>AutoML Blind Spot:</b> Relying exclusively on pre-deployment validation scores frequently selects high-capacity models that are brittle to drift, triggering repeated retraining runs (${chosen.sim.retrains} retrain${chosen.sim.retrains === 1 ? "" : "s"} here) that heavily inflate total lifecycle carbon.`;
  } else if (rule.num === 2) {
    optimizesFor = `<b>Optimizes for:</b> Maximum operational quality in production, sustaining the highest operational accuracy (${fmtPct(chosen.sim.avgAcc)}) across all predictions delivered throughout the ${fmtMonths(s.months)} lifecycle under drift.`;

    relationsText = `<b>Relation to top performers:</b> This model is the <b>#1 leader in operational accuracy</b> (${fmtPct(chosen.sim.avgAcc)}, outperforming the AutoML baseline by ${dAcc >= 0 ? `+${dAcc.toFixed(2)} pts` : `${dAcc.toFixed(2)} pts`}). ${isTopSci 
      ? `Remarkably, its drift robustness also makes it the <b>#1 leader in Consumer SCI (QA)</b> (${fmtSciRate(chosen.sim.sciEffective)})—a win-win where high resilience eliminates costly retraining overhead.` 
      : `However, sustaining this top accuracy requires <b>${fmtWh(chosen.sim.totalJ)}</b> of total energy with a Consumer SCI (QA) of <b>${fmtSciRate(chosen.sim.sciEffective)}</b>, compared to <b>${fmtSciRate(bestSciRow.sim.sciEffective)}</b> for the greenest candidate (<b>${esc(bestSciRow.p.name)}</b>).`}`;

    conclusionText = `<b>Deployment Insight:</b> Prioritizing live operational accuracy protects user-facing reliability, but teams should assess whether the accuracy margin over resilient alternatives justifies the extra compute and retraining footprint.`;
  } else if (rule.num === 3) {
    optimizesFor = `<b>Optimizes for:</b> Sustainable utility delivery, achieving the lowest lifecycle carbon cost per 1,000 correct predictions delivered (${fmtSciRate(chosen.sim.sciEffective)}). It directly penalizes model inaccuracy and mistakes under drift.`;

    relationsText = `<b>Relation to top performers:</b> This model is the <b>#1 leader in Consumer SCI (QA)</b> across all candidates (${fmtSciRate(chosen.sim.sciEffective)}${dCarbon < -1e-6 ? `, saving ${fmtCO2(-dCarbon)} (${pctCarbon.toFixed(1)}%) in lifecycle carbon vs the AutoML baseline` : ""}). ${isTopAcc 
      ? `Furthermore, it matches the <b>highest operational accuracy</b> across models (${fmtPct(chosen.sim.avgAcc)}), proving that drift robustness can eliminate the trade-off between green compute and model accuracy.` 
      : `In operational accuracy, it achieves <b>${fmtPct(chosen.sim.avgAcc)}</b>—trailing the operational accuracy leader (<b>${esc(bestAccRow.p.name)}</b> at ${fmtPct(bestAccRow.sim.avgAcc)}) by only ${Math.abs(bestAccRow.sim.avgAcc - chosen.sim.avgAcc).toFixed(2)} pts, while saving <b>${fmtWh(bestAccRow.sim.totalJ - chosen.sim.totalJ)}</b> of total energy.`}`;

    conclusionText = `<b>Lifecycle Recommendation:</b> By adjusting emissions by delivered accurate predictions, Consumer SCI (QA) guards against both energy-wasteful overparameterized models and deceptively low-power models that suffer unacceptable prediction error.`;
  }

  const takeaway = `
    <div style="margin-bottom:8px;">${optimizesFor}</div>
    <div style="margin-bottom:8px;">${relationsText}</div>
    <div>${conclusionText}</div>
  `;

  if ($("decision-single-result")) {
    $("decision-single-result").innerHTML = `
      <div class="dsingle-head">
        <div>
          <div class="dsingle-rule-context">Selection outcome for Rule ${rule.num} · ${rule.name}</div>
          <div class="dsingle-pick">
            <span class="dot" style="background:${color(chosen.p)}"></span>
            <span>${esc(chosen.p.name)}</span>
            ${rule.isBaseline ? `<span class="badge-base">AutoML Baseline</span>` : (isSameAsBase ? `<span class="badge-base">AutoML Baseline</span>` : "")}
          </div>
        </div>
        <div class="dsingle-meta">${fmtPct(chosen.p.acc)} baseline acc · ${chosen.sim.retrains} retrain${chosen.sim.retrains === 1 ? "" : "s"} · ${fmtTime(chosen.sim.cpuS)} Total CPU Time</div>
      </div>
      <div class="dsingle-metrics">
        <div class="dsingle-metric-box">
          <div class="dsingle-metric-label">Total Energy</div>
          <div class="dsingle-metric-val">${fmtWh(chosen.sim.totalJ)}</div>
          <div class="dsingle-comps-stack">
            <div class="dsingle-comp-row">
              <span class="dsingle-comp-scope">vs Baseline:</span>
              ${energyCompHtml}
            </div>
            <div class="dsingle-comp-row">
              <span class="dsingle-comp-scope">vs Best:</span>
              ${energyCompBestHtml}
            </div>
          </div>
          <div class="dsingle-metric-sub">Development: ${fmtWh(chosen.sim.devJ)} · Deployment: ${fmtWh(chosen.sim.deployJ)}</div>
        </div>
        <div class="dsingle-metric-box">
          <div class="dsingle-metric-label">Operational Accuracy</div>
          <div class="dsingle-metric-val">${fmtPct(chosen.sim.avgAcc)}</div>
          <div class="dsingle-comps-stack">
            <div class="dsingle-comp-row">
              <span class="dsingle-comp-scope">vs Baseline:</span>
              ${accCompHtml}
            </div>
            <div class="dsingle-comp-row">
              <span class="dsingle-comp-scope">vs Best:</span>
              ${accCompBestHtml}
            </div>
          </div>
          <div class="dsingle-metric-sub">Operational range: ${fmtPct(chosen.sim.minAcc)} – ${fmtPct(chosen.sim.maxAcc)}</div>
        </div>
        <div class="dsingle-metric-box">
          <div class="dsingle-metric-label">Consumer SCI (QA)</div>
          <div class="dsingle-metric-val">${fmtSciRate(chosen.sim.sciEffective)}</div>
          <div class="dsingle-comps-stack">
            <div class="dsingle-comp-row">
              <span class="dsingle-comp-scope">vs Baseline:</span>
              ${sciCompHtml}
            </div>
            <div class="dsingle-comp-row">
              <span class="dsingle-comp-scope">vs Best:</span>
              ${sciCompBestHtml}
            </div>
          </div>
          <div class="dsingle-metric-sub">${chosen.sim.correctPreds.toLocaleString()} correct preds · ${fmtCO2(chosen.sim.carbonDeployG)} deploy CO₂e</div>
        </div>
      </div>
      <div class="dsingle-takeaway">${takeaway}</div>
      <div class="dsingle-foot">
        <button class="gbtn" type="button" data-pick="${chosen.p.slot}">Focus ${esc(chosen.p.name)} in Assessment &amp; Charts →</button>
      </div>`;
  }
}

if ($("results")) {
  $("results").addEventListener("click", ev => {
    const tr = ev.target.closest("tr[data-slot]");
    if (tr) selectPipeline(+tr.dataset.slot);
  });
}

if ($("decision-rule-opts")) {
  $("decision-rule-opts").addEventListener("click", ev => {
    const b = ev.target.closest("[data-rule]");
    if (b) {
      selectedDecisionRule = +b.dataset.rule;
      if (last) renderDecisionTab(last.rows, last.s);
      updateDefaultsButtons();
    }
  });
}

if ($("decision-single-result")) {
  $("decision-single-result").addEventListener("click", ev => {
    const b = ev.target.closest("[data-pick]");
    if (b) {
      selectPipeline(+b.dataset.pick);
      switchTab("results");
    }
  });
}

// ---------------------------------------------------------------- methodology live trace
let methodInspectSlot = 1;

function renderMethodology(rows, s) {
  const container = $("method-live-inspector");
  if (!container || !rows || rows.length === 0) return;

  if (!rows.some(r => r.p.slot === methodInspectSlot)) {
    methodInspectSlot = rows[0].p.slot;
  }
  const curr = rows.find(r => r.p.slot === methodInspectSlot) || rows[0];
  const { p, sim } = curr;

  const selectorHtml = rows.map(r => {
    const isAct = r.p.slot === methodInspectSlot;
    return `<button class="live-pipe-btn${isAct ? " active" : ""}" type="button" data-slot="${r.p.slot}">
      <span class="dot" style="background:${color(r.p)}"></span>
      <span>${esc(r.p.name)}</span>
    </button>`;
  }).join("");

  const devKwh = (sim.devJ / 3600 / 1000) * s.pue;
  const deployKwh = (sim.deployJ / 3600 / 1000) * s.pue;
  const penaltyRatio = sim.sciConsumer > 0 ? (sim.sciEffective / sim.sciConsumer) : 1;

  container.innerHTML = `
    <div class="live-inspector-card">
      <div class="live-pipe-selector" id="method-pipe-btns">
        ${selectorHtml}
      </div>

      <div class="live-step-grid">
        <!-- Stage 1: Continuous Drift -->
        <div class="live-step-box">
          <div class="live-step-title">
            <span>1. Drift &amp; Retraining</span>
            <span class="dot" style="background:${color(p)}"></span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Initial Benchmark</span>
            <span class="live-metric-v">${p.acc.toFixed(1)}%</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Drift Drop / Day (Δ)</span>
            <span class="live-metric-v">${sim.dailyDrop.toFixed(3)}%</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Cycle to Floor τ (${s.tau.toFixed(1)}%)</span>
            <span class="live-metric-v">${isFinite(sim.cycle) ? sim.cycle.toFixed(1) + " days" + (p.acc < s.tau ? " (fallback)" : "") : "Never"}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Retraining Runs</span>
            <span class="live-metric-v ${sim.retrains > 0 ? "amber" : ""}">${sim.retrains} event${sim.retrains === 1 ? "" : "s"}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Avg Operating Acc</span>
            <span class="live-metric-v hl">${fmtPct(sim.avgAcc)}</span>
          </div>
        </div>

        <!-- Stage 2: Compute & Energy -->
        <div class="live-step-box">
          <div class="live-step-title">
            <span>2. Compute &amp; Energy</span>
            <span style="font:11px var(--mono);color:var(--muted);">${s.watts}W</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Development CPU Time</span>
            <span class="live-metric-v">${fmtTime(sim.devCpuS)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Deploy Inferences (${fmtNum(s.totalPreds)})</span>
            <span class="live-metric-v">${fmtTime((p.inf / 1000) * s.totalPreds)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Deploy Retraining (${sim.retrains}×)</span>
            <span class="live-metric-v ${sim.retrains > 0 ? "amber" : ""}">${fmtTime(p.train * sim.retrains)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Total Deploy Energy</span>
            <span class="live-metric-v">${fmtWh(sim.deployJ)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Facility Electricity (PUE ${s.pue.toFixed(2)})</span>
            <span class="live-metric-v hl">${deployKwh.toFixed(3)} kWh</span>
          </div>
        </div>

        <!-- Stage 3: Carbon Accounting -->
        <div class="live-step-box">
          <div class="live-step-title">
            <span>3. Carbon Accounting</span>
            <span style="font:11px var(--mono);color:var(--muted);">${s.gridG} g/kWh</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Operational Carbon (O)</span>
            <span class="live-metric-v">${fmtCO2(sim.opDeployG)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Hardware Embodied (M)</span>
            <span class="live-metric-v">${fmtCO2(sim.embDeployG)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Total Deploy Carbon (O+M)</span>
            <span class="live-metric-v hl">${fmtCO2(sim.carbonDeployG)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Retrain Overhead Footprint</span>
            <span class="live-metric-v ${sim.retrains > 0 ? "amber" : ""}">${fmtCO2(sim.retrainCarbonG)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Offline Dev Carbon (Provider)</span>
            <span class="live-metric-v">${fmtCO2(sim.carbonDevG)}</span>
          </div>
        </div>

        <!-- Stage 4: Functional Units & Quality Adaptation -->
        <div class="live-step-box" style="border-color:var(--amber);">
          <div class="live-step-title" style="color:var(--ink);">
            <span>4. SCI Functional Units</span>
            <span class="method-tag amber" style="font-size:9.5px;padding:1px 5px;">R Units</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Total Inferences (R_tot)</span>
            <span class="live-metric-v">${fmtNum(s.totalPreds)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Correct Inferences (R_cor)</span>
            <span class="live-metric-v hl">${fmtNum(sim.correctPreds)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Consumer SCI (per 1k)</span>
            <span class="live-metric-v">${fmtSciRate(sim.sciConsumer)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Consumer SCI (QA)</span>
            <span class="live-metric-v hl">${fmtSciRate(sim.sciEffective)}</span>
          </div>
          <div class="live-metric-row">
            <span class="live-metric-k">Mistake Penalty Factor</span>
            <span class="live-metric-v ${penaltyRatio > 1.15 ? "ember" : ""}">${penaltyRatio.toFixed(2)}× (+${((penaltyRatio - 1) * 100).toFixed(1)}%)</span>
          </div>
        </div>
      </div>

      <div class="live-summary-banner">
        <div class="lsb-text">
          <b>Quality Adaptation Proof for ${esc(p.name)}:</b> Serving ${fmtNum(s.totalPreds)} total inferences emitted <b>${fmtCO2(sim.carbonDeployG)}</b>. Because continuous data drift degraded accuracy to an average of <b>${fmtPct(sim.avgAcc)}</b>, only <b>${fmtNum(sim.correctPreds)}</b> predictions were correct. Dividing by delivered accurate utility increases the SCI from <b>${fmtSciRate(sim.sciConsumer)}</b> to <b>${fmtSciRate(sim.sciEffective)}</b>, directly penalizing operational errors.
        </div>
        <div class="lsb-val">Consumer SCI (QA): ${fmtSciRate(sim.sciEffective)}</div>
      </div>
    </div>
  `;

  const selector = $("method-pipe-btns");
  if (selector) {
    selector.querySelectorAll(".live-pipe-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        methodInspectSlot = +btn.dataset.slot;
        renderMethodology(rows, s);
      });
    });
  }
}

// ---------------------------------------------------------------- tab navigation
function switchTab(viewId) {
  currentTab = viewId;
  document.querySelectorAll(".tab").forEach(o => o.setAttribute("aria-selected", String(o.dataset.view === viewId)));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === "view-" + viewId));
  if (viewId === "results" && last) {
    // Re-render charts once container is visible
    setTimeout(() => renderCharts(last.rows, last.s), 20);
  }
  if (viewId === "method" && last) {
    renderMethodology(last.rows, last.s);
  }
}

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => switchTab(t.dataset.view)));
if ($("btn-goto-scenario")) $("btn-goto-scenario").addEventListener("click", () => switchTab("scenario"));
if ($("btn-scenario-back-pipe")) $("btn-scenario-back-pipe").addEventListener("click", () => switchTab("pipelines"));
if ($("btn-goto-results")) $("btn-goto-results").addEventListener("click", () => switchTab("results"));
if ($("btn-goto-decision")) $("btn-goto-decision").addEventListener("click", () => switchTab("decision"));
if ($("btn-goto-method")) $("btn-goto-method").addEventListener("click", () => switchTab("method"));

// ---------------------------------------------------------------- wiring & rAF throttling
let last = null;
let rafId = null;

function scheduleUpdate() {
  if (rafId) return;
  rafId = requestAnimationFrame(() => {
    rafId = null;
    update();
  });
}

function update() {
  const s = settings();
  $("drift-v").textContent = s.m.toFixed(2);
  $("months-v").textContent = fmtMonths(s.months);
  $("period-hint").textContent = `${s.months} months (${fmtNum(s.days)} days) · Total traffic: ${fmtNum(s.totalPreds)} predictions.`;
  $("traffic-v").textContent = `${fmtNum(s.traffic)} / day`;
  $("floor-v").textContent = `${s.floor}%`;
  $("n-v").textContent = `${s.retrainMonths} month${s.retrainMonths === 1 ? "" : "s"} (${s.retrainMonths * 30} days)`;
  if ($("delta-v")) $("delta-v").textContent = "5 pts";
  if ($("tau-v")) $("tau-v").textContent = `${+s.tau.toFixed(1)}%`;
  if ($("subthreshold-days-v")) $("subthreshold-days-v").textContent = `${s.subthresholdDays} days`;
  if ($("subthreshold-days")) $("subthreshold-days").disabled = !s.subthresholdRetrain;
  $("presets").innerHTML = PRESETS.map(pr =>
    `<button class="chip" type="button" data-m="${pr.m}" aria-pressed="${Math.abs(pr.m - s.m) < 1e-9}">${pr.name}</button>`).join("");

  // Summaries & Results header
  if ($("grid-intensity-v")) $("grid-intensity-v").textContent = `${s.gridG} gCO₂e/kWh`;
  if ($("hardware-util-v")) $("hardware-util-v").textContent = `${s.hardwareUtil}%`;
  if ($("datacenter-pue-v")) $("datacenter-pue-v").textContent = `${s.pue.toFixed(2)}`;
  if ($("pipe-summary")) $("pipe-summary").textContent = `${pipelines.length} candidate model${pipelines.length === 1 ? "" : "s"} configured`;
  if ($("setup-summary")) $("setup-summary").textContent = `${fmtMonths(s.months)} · ${fmtNum(s.traffic)} preds/day · ${MACHINES[machine].name} machine · ${s.gridG} gCO₂e/kWh`;
  $("res-duration").textContent = fmtMonths(s.months);
  $("res-traffic").textContent = `${fmtNum(s.traffic)}`;

  const rows = pipelines.map(p => ({ p, sim: simulate(p, s) }));
  last = { rows, s };
  renderPipeOpts(rows);
  renderBillsGrid(rows, s);
  if (currentTab === "results") {
    renderCharts(rows, s);
  }
  renderComparisonTable(rows, s);
  renderDecisionTab(rows, s);
  renderMethodology(rows, s);
  updateDefaultsButtons();
}

["drift", "months", "traffic", "floor", "n", "tau", "grid-intensity", "hardware-util", "datacenter-pue", "subthreshold-days"]
  .forEach(id => { if ($(id)) $(id).addEventListener("input", scheduleUpdate); });

if ($("subthreshold-retrain")) {
  $("subthreshold-retrain").addEventListener("change", scheduleUpdate);
}

if ($("grid-intensity")) {
  $("grid-intensity").addEventListener("input", () => {
    const val = +$("grid-intensity").value;
    const match = GRIDS.find(g => g.g === val);
    gridPreset = match ? match.id : null;
    renderStaticOpts();
  });
}



let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (currentTab === "results" && last) renderCharts(last.rows, last.s);
  }, 120);
});

// default 5,000 predictions per day, 12 months, cleaner grid (230 gCO2e/kWh), accuracy threshold 82.0%
$("traffic").value = (Math.log10(5000) - 2) / 3;
$("months").value = 12;
if ($("tau")) $("tau").value = 0.82;
if ($("grid-intensity")) $("grid-intensity").value = 230;
if ($("hardware-util")) $("hardware-util").value = 100;
if ($("datacenter-pue")) $("datacenter-pue").value = 1.20;
renderStaticOpts();
update();
updateDefaultsButtons();

document.addEventListener("click", ev => {
  if (ev.target.closest(".btn-restore-defaults")) {
    restoreAllDefaults();
  }
});
