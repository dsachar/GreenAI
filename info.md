# Green AI: Lifecycle Assessment of ML Models — System Documentation

## 1. Overview & Purpose

The **Green AI: Lifecycle Assessment of ML Models** is an interactive, zero-build web application designed to advance the core principles of **Green AI** in production machine learning and Automated Machine Learning (**AutoML**). As formulated by Schwartz et al. (2020), **Green AI** advocates that environmental footprint, energy consumption, and compute efficiency must be evaluated as primary criteria alongside predictive accuracy—countering the unsustainable "Red AI" trend of purchasing minor accuracy gains with exponential compute.

This system demonstrates the critical importance of evaluating the *entire lifecycle carbon footprint* of ML models when selecting candidate architectures for production deployment in platforms such as AutoML and MLOps model registries.

### Motivation & The AutoML Selection Blind Spot
Traditional AutoML and model selection heuristics exemplify Red AI practices: they rank candidate models based solely on offline validation accuracy over static test benchmarks. However, once deployed into production services:
- **Models respond differently to data drift**: Real-world input data distributions shift continuously. Brittle architectures suffer rapid accuracy degradation, while robust architectures maintain stability.
- **Drift dictates retraining overhead**: Whenever live accuracy falls below acceptable operational thresholds, retraining must be triggered. Each retraining event repeatedly incurs the model's full training compute, energy, and carbon bill. A model that looks cheap during initial development may trigger dozens of retrains in production, causing its lifecycle footprint to explode.
- **Inference traffic scales continuously**: Per-query serving latency and power consumption compound over millions of operational predictions.
- **SCI for AI captures the complete lifecycle**: Standardized under **[ISO/IEC 21031:2024](https://www.iso.org/standard/86612.html)** and the **[Green Software Foundation (GSF) SCI Standard](https://grnsft.org/sci)** (see also the **[GSF SCI for AI Working Group](https://github.com/Green-Software-Foundation/sci-ai)**), **Software Carbon Intensity for AI (SCI for AI)** accounts for both operational electricity ($O$, serving + retraining) and embodied hardware manufacturing footprint ($M$).
- **Consumer SCI (QA) adapts for model mistakes**: By defining the functional unit as carbon per 1,000 *correct* predictions delivered, Quality-Adjusted Consumer SCI — shorthand **Consumer SCI (QA)** — directly penalizes model inaccuracy and mistakes under drift, ensuring that environmental assessment is aligned with delivered real-world utility.

---

## 2. Technical Architecture

- **Format**: Zero-build static web application (pure HTML5, CSS3, and ES6+ JavaScript).
- **Standards**: Aligned with Green AI principles, ISO/IEC 21031:2024, and the Green Software Foundation (GSF) *Software Carbon Intensity (SCI) for AI* specification (`Green-Software-Foundation/sci-ai`).
- **Files**:
  - `index.html`: UI structure organized into 4 workflow tabs (*01. Models*, *02. Configuration*, *03. Assessment*, *04. Which Model to Deploy*) and a right-aligned unnumbered reference tab (*Methodology*).
  - `styles.css`: Custom "field-notebook" theme with grid paper styling, side-by-side paper bill slips, carbon badges, and full-width panels.
  - `app.js`: State management, O(1) memory closed-form simulation engine, GSF SCI carbon accounting, bounded SVG chart renderers, and requestAnimationFrame debouncing.
- **Dependencies**: Google Fonts (`Space Grotesk` and `IBM Plex Mono`). Falls back to system monospace and sans-serif fonts offline.

---

## 3. Mathematical & Simulation Models

### 3.0 Notation & Mathematical Symbols

The following standardized symbols and units are used consistently across this documentation, the application user interface, and the simulation engine:

| Symbol | Concept / Metric | Dimension / Unit | Description |
|---|---|---|---|
| $Acc_{\text{dev}}$ / $\text{baseline}$ | Development Test Accuracy | $\%$ | Initial offline validation benchmark accuracy before deployment ($50.0\% - 100.0\%$). |
| $\text{robustness}$ | Drift Robustness | $\text{pts} / 100\text{k}$ | Accuracy points lost per 100,000 predictions at drift magnitude $m = 1.0$ (lower = more resilient). |
| $m$ | Data Drift Severity | Dimensionless ($0.0 - 1.0$) | Scenario severity multiplier ($0.0 = \text{static}, 0.2 = \text{mild}, 0.5 = \text{moderate}, 0.9 = \text{severe}$). |
| $n$ | Inference Count | Counts | Cumulative predictions served since the most recent (re)training event. |
| $\text{floor}$ | Accuracy Floor | $\%$ | Asymptotic performance floor under severe drift (default $50.0\%$, random guessing). |
| $R_{\text{daily}}$ | Daily Requests | $\text{requests} / \text{day}$ | Continuous daily incoming prediction volume ($100 - 100{,}000$). |
| $R_{\text{total}}$ | Total Lifecycle Predictions | Counts | Total queries served throughout deployment ($R_{\text{total}} = \text{days} \times R_{\text{daily}}$). |
| $\Delta_{\text{daily}}$ | Daily Accuracy Loss Rate | $\% / \text{day}$ | Continuous accuracy degradation per calendar day. |
| $\tau$ | Retraining Threshold | $\%$ | Cutoff accuracy threshold below which automatic retraining is triggered. |
| $X$ | Sub-threshold Fallback | $\text{days}$ | Periodic retraining interval fallback when live/baseline accuracy is below $\tau$ (default $30\text{ days}$). |
| $K_{\text{retrain}}$ | Retrains Count | Events | Number of retraining events executed across the deployment lifecycle. |
| $\overline{Acc}$ | Operational Accuracy | $\%$ | Time-averaged live accuracy sustained across deployment under continuous drift. |
| $R_{\text{correct}}$ | Correct Predictions Delivered | Counts | Functional volume of accurate predictions ($R_{\text{correct}} = \operatorname{round}((\overline{Acc} / 100) \times R_{\text{total}})$). |
| $t_{\text{train}}$ | Training Runtime | Seconds ($\text{s}$) | CPU execution time required to fit candidate architecture parameters (paid on initial train and each retrain). |
| $t_{\text{inf}}$ | Inference Latency | Milliseconds ($\text{ms}$) | Per-prediction inference CPU execution latency. |
| $T_{\text{dev}}, T_{\text{deploy}}, T_{\text{total}}$ | CPU Runtime | Seconds ($\text{s}$) | Active CPU execution runtime across development, deployment, and full lifecycle. |
| $P_{\text{machine}}$ | Machine Power Draw | Watts ($\text{W}$, $\text{J/s}$) | Host instance active electrical power consumption under $100\%$ CPU load. |
| $E_{\text{dev}}, E_{\text{deploy}}, E_{\text{total}}$ | Workload Electrical Energy | Watt-hours ($\text{Wh}$) | Direct electrical energy consumed by candidate model compute workloads. |
| $\text{PUE}$ | Power Usage Effectiveness | Multiplier | Datacenter facility cooling and infrastructure overhead factor (default $1.20$). |
| $I$ | Grid Carbon Intensity | $\text{gCO}_2\text{e} / \text{kWh}$ | Location-based marginal carbon emission factor of the regional electrical grid (no offsets). |
| $O_{\text{dev}}, O_{\text{deploy}}, O_{\text{total}}$ | Operational Emissions | $\text{gCO}_2\text{e}$ | Scope 2 greenhouse gas emissions resulting from electricity consumed by servers and facility. |
| $M_{\text{dev}}, M_{\text{deploy}}, M_{\text{total}}$ | Embodied Hardware Carbon | $\text{gCO}_2\text{e}$ | Scope 3 server manufacturing, supply chain, and recycling carbon amortized by execution time and utilization. |
| $U$ | Hardware Utilization | $\%$ ($10\% - 100\%$) | Server hardware utilization factor (default $100\%$; lower utilization inflates embodied carbon $M$). |
| $C_{\text{dev}}, C_{\text{deploy}}, C_{\text{total}}$ | Lifecycle Carbon | $\text{gCO}_2\text{e}$ | Combined operational and embodied greenhouse gas emissions ($C = O + M$). |
| $\text{Provider SCI}$ | Provider Carbon Intensity | $\text{gCO}_2\text{e} / \text{model}$ | Pre-deployment development carbon footprint per candidate model trained ($R = 1\text{ model}$). |
| $\text{Consumer SCI}$ | Consumer Carbon Intensity | $\text{gCO}_2\text{e} / 1\text{k preds}$ | Operational carbon footprint per 1,000 live production inferences served ($R = R_{\text{total}} / 1{,}000$). |
| $\text{Consumer SCI (QA)}$ | Quality-Adjusted Consumer SCI | $\text{gCO}_2\text{e} / 1\text{k correct}$ | Operational carbon footprint per 1,000 *correct* predictions delivered ($R = R_{\text{correct}} / 1{,}000$). |

---

### 3.1 Continuous Drift Model
Drift is continuous and occurs with every individual prediction data point served since the most recent (re)training:

```text
accuracy(n) = max(floor, baseline - (robustness * m / 100,000) * n)
```

Formally:
$$Acc(n) = \max\left(\text{floor}, \text{baseline} - \frac{\text{robustness} \times m}{100{,}000} \times n\right)$$

- **`baseline`** ($Acc_{\text{dev}}$): The initial test accuracy of the candidate model (%).
- **`robustness`**: Accuracy drop per 100,000 data points under drift magnitude 1 (`m = 1.0`). Lower values indicate greater stability (more robust).
- **`m`**: Continuous drift severity factor (slider between `0.0` and `1.0`, where `0.0` denotes static data).
- **`n`**: Number of individual predictions served since the model was last trained or retrained.
- **`floor`**: The minimum possible accuracy (default: `50.0%`, simulating random guessing for binary classification).

Drift is strictly non-beneficial (it never improves accuracy). The candidate model starts serving at baseline accuracy (`n = 0`).

---

### 3.2 Time & Traffic Model
Predictions arrive at a steady daily rate. Real-world durations and prediction counts are derived directly:

```text
total_predictions (R_total) = days * daily_requests
daily accuracy drop (Δ_daily) = (robustness * m * daily_requests) / 100,000
```

Formally:
$$R_{\text{total}} = \text{days} \times R_{\text{daily}}$$
$$\Delta_{\text{daily}} = \frac{\text{robustness} \times m \times R_{\text{daily}}}{100{,}000}$$

- **Deployment duration**: Measured in months (`1` to `36` months, covering up to 3 years; $1\text{ month} = 30\text{ days}$, so $\text{days} = \text{months} \times 30$).
- **Daily requests** ($R_{\text{daily}}$): Number of requests served per day (log scale slider from `100` to `100,000` requests/day).
- Training and retraining are assumed to execute concurrently in background infrastructure without interrupting live inference traffic.
- Chart time axes automatically format their units:
  - Up to 2 months: `days`
  - Beyond 2 months: `months`

---

### 3.3 Energy & Compute Model
Compute time and electrical energy consumption are decoupled across offline development and online operational deployment:

```text
Development CPU Time (T_dev)   = t_train + (t_inf / 1,000) * 10,000 eval_predictions
Deployment CPU Time (T_deploy) = (t_train * retrains) + (t_inf / 1,000) * total_predictions

Total CPU Time (T_total) = Development CPU Time + Deployment CPU Time
Total Energy (Wh)        = Total CPU Time (seconds) * machine_power (Watts) / 3,600
```

Formally:
$$T_{\text{dev}} = t_{\text{train}} + \left(\frac{t_{\text{inf}}}{1{,}000}\right) \times 10{,}000$$
$$T_{\text{deploy}} = (t_{\text{train}} \times K_{\text{retrain}}) + \left(\frac{t_{\text{inf}}}{1{,}000}\right) \times R_{\text{total}}$$
$$T_{\text{total}} = T_{\text{dev}} + T_{\text{deploy}}$$
$$E_{\text{dev}} = \frac{T_{\text{dev}} \times P_{\text{machine}}}{3{,}600} \quad [\text{Wh}], \qquad E_{\text{deploy}} = \frac{T_{\text{deploy}} \times P_{\text{machine}}}{3{,}600} \quad [\text{Wh}]$$
$$E_{\text{total}} = E_{\text{dev}} + E_{\text{deploy}} = \frac{T_{\text{total}} \times P_{\text{machine}}}{3{,}600} \quad [\text{Wh}]$$

- **Development Stage**:
  - **Initial Training Energy**: Paid once before deployment ($t_{\text{train}} \times P_{\text{machine}} / 3{,}600$).
  - **Evaluation Energy**: 10,000 test predictions evaluating model accuracy before shipment ($(t_{\text{inf}} / 1{,}000) \times 10{,}000 \times P_{\text{machine}} / 3{,}600$).
  - **Development CPU Time** ($T_{\text{dev}}$): Total CPU time invested during development.
- **Deployment Stage**:
  - **Retraining Energy**: Each retrain incurs the exact same CPU time ($t_{\text{train}}$) and energy as initial training.
  - **Inference Energy**: Accrues continuously with production requests served over time ($(t_{\text{inf}} / 1{,}000) \times R_{\text{total}} \times P_{\text{machine}} / 3{,}600$).
  - **Deployment CPU Time** ($T_{\text{deploy}}$): Total CPU time expended during live operations.
- **Machine Power Profiles** ($P_{\text{machine}}$):
  - **Small** (`m5.large`): 2 vCPU · 7.9 J/s (Watts)
  - **Medium** (`m5.xlarge`): 4 vCPU · 15.9 J/s (Watts)
  - **Large** (`m5.2xlarge`): 8 vCPU · 31.8 J/s (Watts)

---

### 3.4 Retraining Policies
The active retraining policy dictates when retraining is triggered:

1. **No retraining**: Model is never retrained; accuracy degrades continuously toward the floor ($\text{floor}$).
2. **Fixed schedule**: Retrains on a calendar schedule every $N$ months ($\text{cycle\_days} = N \times 30$).
3. **Accuracy**: Retrains whenever live accuracy drops below a shared global threshold $\tau$ ($Acc(n) < \tau$):
   $$\text{cycle\_days} = \max\left(1, \frac{\text{baseline} - \tau}{\Delta_{\text{daily}}}\right)$$
   $$K_{\text{retrain}} = \left\lfloor \frac{\text{days} - 10^{-6}}{\text{cycle\_days}} \right\rfloor$$
   - **Sub-threshold Fallback**: If a model's baseline test accuracy is lower than or equal to the cutoff threshold ($\text{baseline} \le \tau$) or live accuracy falls below $\tau$, an advanced fallback setting triggers periodic retraining every $X = 30$ days (configurable from 5 to 90 days, or toggled off). If disabled, sub-threshold models do not retrain.

A retrain event resets inference count ($n = 0$) and fully restores baseline accuracy for subsequent incoming predictions.

---

### 3.5 Expected Energy Formulation
When estimating future operational energy during model selection, initial development training is excluded since all models must pay it at least once:

```text
expected energy (Joules) = (expected_retrains * t_train + (t_inf / 1,000) * total_predictions) * machine_watts
```

Formally:
$$E_{\text{expected}} = \left(K_{\text{retrain}} \times t_{\text{train}} + \left(\frac{t_{\text{inf}}}{1{,}000}\right) \times R_{\text{total}}\right) \times P_{\text{machine}} \quad [\text{Joules}]$$

- **Fixed schedule**: $K_{\text{retrain}} = \lfloor (\text{days} - 1) / (N \times 30) \rfloor$
- **Accuracy threshold**: $K_{\text{retrain}} = \lfloor (\text{days} - 1) / \text{cycle\_days} \rfloor$, where $\text{cycle\_days}$ is the interval until threshold $\tau$ is crossed ($0$ if it is never reached).

---

### 3.6 GSF Software Carbon Intensity (SCI) for AI Model
In accordance with ISO/IEC 21031:2024 and the Green Software Foundation SCI for AI specification (`Green-Software-Foundation/sci-ai`):

```text
SCI = (O + M) / R = ((E * I * PUE) + M) / R
```

Formally:
$$\text{SCI} = \frac{O + M}{R} = \frac{(E_{\text{facility}} \times I) + M}{R}$$

#### Operational Emissions ($O$)
Operational emissions quantify the greenhouse gas emissions resulting from electricity consumed by ML compute workloads:
```text
O = E_facility * I = (Compute Energy (Wh) * PUE / 1,000) * I  [gCO2e]
```

Formally:
$$E_{\text{facility}} = \frac{E_{\text{workload}} \text{ (Wh)}}{1{,}000} \times \text{PUE} \quad [\text{kWh}]$$
$$O = E_{\text{facility}} \times I \quad [\text{gCO}_2\text{e}]$$

- **$E_{\text{workload}}$**: Total workload electricity consumption ($E_{\text{dev}}$, $E_{\text{deploy}}$, or $E_{\text{total}}$ in Wh).
- **$I$**: Location-based grid carbon intensity ($\text{gCO}_2\text{e} / \text{kWh}$). Market-based instruments (RECs, green tariffs, unbundled certificates, carbon offsets) are strictly excluded per ISO/IEC 21031:2024 normative rules.
  - **Low-carbon grid**: `50 gCO₂e/kWh` (hydro/nuclear dominant, e.g. France, Sweden)
  - **Cleaner grid**: `230 gCO₂e/kWh` (mixed renewables/gas, e.g. California, UK, EU average)
  - **Global average**: `450 gCO₂e/kWh` (World electricity grid mean)
  - **Carbon-heavy grid**: `650 gCO₂e/kWh` (coal/fossil dominant, e.g. Poland, India, parts of US Midwest)
- **$\text{PUE}$**: Datacenter Power Usage Effectiveness overhead multiplier (default: `1.20`, standard modern hyperscale facility overhead).

#### Embodied Hardware Emissions ($M$)
Embodied emissions quantify the Scope 3 manufacturing, supply chain, transport, and end-of-life recycling footprint of physical compute hardware, amortized over execution time:
```text
M = TE * (TR / EL) * (vCPU / total_server_vCPUs) / U  [gCO2e]
```

Formally:
$$M = TE \times \left(\frac{TR}{EL}\right) \times \left(\frac{\text{vCPU}}{\text{total\_server\_vCPU}}\right) \div U = \left(\frac{T}{3{,}600}\right) \times \left(\frac{\text{emb\_rate\_per\_hr}}{U}\right) \quad [\text{gCO}_2\text{e}]$$

- **$TE$**: Total embodied carbon of physical server hardware ($1{,}200\text{ kg CO}_2\text{e} = 1{,}200{,}000\text{ gCO}_2\text{e}$ per standard 2-socket 48 vCPU rack server per Boavizta / GSF reference data).
- **$EL$**: Expected hardware lifespan ($4\text{ years} = 35{,}040\text{ hours}$).
- **$TR$**: Workload active CPU runtime in hours ($TR = T / 3{,}600$).
- **$U$**: Hardware utilization factor (default: `100%`, slider from $10\%$ to $100\%$; lower utilization inflates embodied allocation per active hour).
- **Amortized baseline instance emission rates at 100% utilization** ($\text{emb\_rate\_per\_hr}$):
  - **Small** (`m5.large`, 2 vCPU): `1.427 gCO₂e / hour`
  - **Medium** (`m5.xlarge`, 4 vCPU): `2.854 gCO₂e / hour`
  - **Large** (`m5.2xlarge`, 8 vCPU): `5.708 gCO₂e / hour`

#### Persona Functional Boundaries ($R$)
The SCI metric requires expressing intensity per functional unit $R$:
1. **Provider Persona (Development Boundary)**:
   - Boundary: Initial training and offline validation benchmark ($O_{\text{dev}} + M_{\text{dev}}$).
   - Functional unit: Per 1 trained candidate model ($R = 1\text{ model}$).
   $$\text{Provider SCI} = O_{\text{dev}} + M_{\text{dev}} \quad [\text{gCO}_2\text{e} / \text{model}]$$
2. **Consumer Persona (Default Production Rate)**:
   - Boundary: Operational serving and continuous retraining during production ($O_{\text{deploy}} + M_{\text{deploy}}$).
   - Functional unit: Per 1,000 live production inferences ($R = R_{\text{total}} / 1{,}000$).
   $$\text{Consumer SCI} = \frac{O_{\text{deploy}} + M_{\text{deploy}}}{R_{\text{total}} / 1{,}000} \quad [\text{gCO}_2\text{e} / 1\text{k predictions}]$$
3. **Quality-Adjusted Consumer Persona (Consumer SCI (QA))**:
   - Boundary: Deployment emissions ($O_{\text{deploy}} + M_{\text{deploy}}$) per correct prediction delivered.
   - Functional unit: Per 1,000 *correct* predictions delivered to end users ($R = R_{\text{correct}} / 1{,}000$, where $R_{\text{correct}} = \operatorname{round}((\overline{Acc} / 100) \times R_{\text{total}})$), explicitly rewarding models that maintain higher accuracy under drift.
   $$\text{Consumer SCI (QA)} = \frac{O_{\text{deploy}} + M_{\text{deploy}}}{R_{\text{correct}} / 1{,}000} = \frac{\text{Consumer SCI}}{\overline{Acc} / 100} \quad [\text{gCO}_2\text{e} / 1\text{k correct predictions}]$$

#### Accounting for Retraining Impact in SCI
The continuous data drift retraining impact is fully incorporated into the SCI calculation:
- **Operational retrain electricity**:
  $$E_{\text{retrain}} = \frac{K_{\text{retrain}} \times t_{\text{train}} \times P_{\text{machine}}}{3{,}600 \times 1{,}000} \times \text{PUE} \quad [\text{kWh}]$$
- **Operational retrain carbon**:
  $$O_{\text{retrain}} = E_{\text{retrain}} \times I \quad [\text{gCO}_2\text{e}]$$
- **Embodied retrain carbon**:
  $$M_{\text{retrain}} = \left(\frac{K_{\text{retrain}} \times t_{\text{train}}}{3{,}600}\right) \times \left(\frac{\text{emb\_rate\_per\_hr}}{U}\right) \quad [\text{gCO}_2\text{e}]$$
- Both $O_{\text{retrain}}$ and $M_{\text{retrain}}$ are directly summed into $O_{\text{deploy}}$ and $M_{\text{deploy}}$. Consequently, every drift-triggered retraining event actively increases both Consumer SCI and Consumer SCI (QA) scores proportionally.

---

## 4. User Interface & Tab Structure

The application is structured into **five dedicated tabs** for optimal separation of concerns:

```
+-----------------------------------------------------------------------------------------+
| Header: Title, Eyebrow, Narrative Description, Indicative Disclaimer Callout            |
+-----------------------------------------------------------------------------------------+
| Tabs: [ 01 Models ] [ 02 Configuration ] [ 03 Assessment ] [ 04 Which Model to Deploy ]                [ Methodology ] |
+-----------------------------------------------------------------------------------------+
```

### Universal Tab Banners & Dynamic "Restore Defaults" Controls
The four core workflow tabs (*01 Models*, *02 Configuration*, *03 Assessment*, *04 Which Model to Deploy*) feature a dedicated top banner providing immediate orientation and reproducibility:
- **Plain-Language Default Explanations**: A clean card with a green indicator badge (`DEFAULT SETUP`) summarizes in plain, non-technical words the default candidate architectures, simulation duration, daily requests, carbon grid intensity, and active decision heuristic relevant to that view.
- **Dynamic "↺ Restore Defaults" Action**: A responsive button that automatically remains hidden while default values are active, and smoothly surfaces across the workflow tabs the moment any model attribute, scenario slider, retraining setting, or decision rule is modified. Clicking the button instantly restores all pipelines, operational sliders, and active rules back to the canonical baseline scenario and dismisses the button.
*(Note: The unnumbered Methodology tab serves strictly as an educational technical reference specification and intentionally omits the defaults banner.)*

### Tab 1 · 01 Models (Candidate Models)
A dedicated workspace focused entirely on candidate model configuration:
- **Candidate ML Models (Tab 03-Style Notebook Boxes)**: Model cards designed with the exact receipt/bill box aesthetic of Tab 03 (rounded 12px corners, ink border, warm cream paper background, header with categorical color dot, editable model name, and remove button):
  - **Inline Parameter Controls**: All configuration parameters are embedded and adjusted directly inside each model's box with live sliders and formatted readouts:
    - **Accuracy & Drift Specs**: Baseline development test-set accuracy (50–100%) and drift robustness (0.1–5.0 pts accuracy drop per 100k data points under drift magnitude 1).
    - **Compute & Latency Specs**: Initial training time (5–3,600 s on log scale, paid on initial train and each continuous retrain) and inference latency per prediction (0.01–50 ms on log scale).
  - **Model Archetype Stat Box**: Highlights the model's structural archetype (e.g. *High-capacity deep model*, *Balanced ensemble model*, *Lightweight fast baseline*) alongside its baseline accuracy. Extraneous status badges (such as "Ready for scenario") are intentionally omitted for clean readability.
  - **Zero Deployment Energy**: Total Energy (Wh / Joules) is strictly omitted from Tab 01 cards, as deployment energy depends entirely on scenario variables (traffic, drift, duration, retraining policy) configured in Tab 02 and evaluated in Tab 03.
- **Add / Remove / Select**: Quick controls to add new candidate models (+ Add model) or remove models.
- **Action Banner**: Displays configured candidate model count with a prominent **[ Next: Configuration → ]** button.

---

### Tab 2 · 02 Configuration (Operating Conditions & Lifecycle)
A focused configuration workspace organized in a **balanced 2-column grid** (fitting two cards side-by-side across the standard width):
- **Row 1 (Traffic & Drift)**:
  - **Part 1 · Lifecycle Period & Serving Requests**: Service duration slider (1–36 months) and daily requests slider (100–100,000 / day).
  - **Part 2 · Data Drift & Accuracy Floor**: Preset drift chips (*Stable*, *Mild*, *Moderate*, *Severe*), fine continuous drift magnitude slider (0.00–1.00), and minimum acceptable accuracy floor (0–90%).
- **Row 2 (Model Lifecycle Operations)**:
  - **Part 3 · Retraining Policy**: Interactive selection between *No retraining*, *Fixed schedule*, and *Accuracy* (with cutoff threshold slider $\tau$, and sub-threshold periodic fallback every $X = 30$ days).
  - **Part 4 · Machine Instance**: Hardware instance tier (Small, Medium, Large) specifying CPU core power draw and vCPU counts.
- **Row 3 (Carbon & Infrastructure Context)**:
  - **Part 5 · Grid Carbon Intensity ($I$)**: Grid carbon intensity presets (*Low-carbon 50 g*, *Cleaner grid 230 g*, *Global avg 450 g*, *Carbon-heavy 650 g*) and fine slider (10–900 gCO₂e/kWh).
  - **Part 6 · Facility PUE & Hardware Embodied ($M$)**: Datacenter facility PUE multiplier (1.0–2.0) and server hardware utilization slider ($U$, 10–100%).
- **Action Banner**: Full-width configuration readiness summary with a single forward progression button: **[ View Assessment & Score Cards → ]** (encouraging a guided unidirectional workflow).

---

### Tab 3 · 03 Assessment (Score Cards, Charts & Table)
A spacious, full-width analytics report organized with executive metrics first:

1. **Header & Navigation**:
   - Scenario summary headline displaying duration, daily requests rate, and active grid intensity. Global navigation is handled via the top tab bar.
2. **Executive Metric Comparison Table (Placed at the Very Top)**:
   - Comprehensive side-by-side metrics table positioned as the very first section in Tab 03.
   - Compares every model across:
     - **Model**: Candidate architecture name and categorical indicator dot.
     - **Development Accuracy**: Baseline test validation benchmark score ($Acc_{\text{dev}}$, %).
     - **Operational Accuracy**: Time-averaged live accuracy sustained under continuous drift ($\overline{Acc}$, %).
     - **Retrains**: Total count of drift- or schedule-triggered retraining events ($K_{\text{retrain}}$).
     - **Total Energy**: Combined development and deployment electrical energy ($E_{\text{total}}$, Wh or kWh).
     - **Lifecycle Carbon**: Total operational electrical emissions and embodied server carbon ($C_{\text{total}} = O_{\text{total}} + M_{\text{total}}$, gCO₂e or kgCO₂e).
     - **Consumer SCI (QA)**: Quality-adjusted software carbon intensity per 1,000 *correct* predictions delivered ($[O_{\text{deploy}} + M_{\text{deploy}}] / [R_{\text{correct}} / 1{,}000]$, g or mg CO₂e / 1k correct).
   - **Best values bolded**: For every metric column, the winning value across models is highlighted and bolded in green (`<b>` / `.best`).
   - Clean, direct model comparison with no extraneous rank column. Total CPU time is omitted in favor of direct baseline-versus-operational accuracy and carbon metrics.
   - Clicking any row selects that model and focuses it across the score cards and charts.
3. **Itemized Energy, Performance & SCI for AI Score Cards (Horizontal Scrollable Round Boxes)**:
   - Modern rounded cards (`border-radius: 12px`, ink border, offset shadow) presented in a smooth **horizontal scrollable view**.
   - Stable hover state without vertical translation jumping.
   - **Header**: Model name with categorical color dot and focused state outline.
   - **Three Distinct Parts**:
     - **ENERGY Part**:
       - *Collapsible detailed listing*: Development (Initial Training Energy, Evaluation Energy, Development CPU Time), Deployment (Retraining Energy, Inference Energy, Deployment CPU Time), and a **mini energy breakdown plot** in neutral greyscale (Train, Eval, Retrain, Infer).
       - *Highlighted aggregate (Always visible)*: **TOTAL ENERGY** in Wh alongside Total CPU Time.
     - **PERFORMANCE Part**:
       - *Collapsible detailed listing*: Accuracy distribution violin plot rendered in clean neutral greyscale, Max Accuracy, Min Accuracy, Operational Accuracy, and Correct Predictions delivered.
       - *Highlighted aggregate (Always visible)*: **OPERATIONAL ACCURACY** (%) alongside accuracy range.
     - **SCI FOR AI Part (All Options Visible in Card Box)**:
       - *Collapsible detailed listing*: Operational Carbon ($O$), Embodied Carbon ($M$), Lifecycle Carbon, Retraining Carbon, Carbon Share mini plot ($O$ vs $M$) in neutral greyscale, and all functional units ($R$): Consumer SCI (QA) (per 1k correct), Consumer SCI (per 1k preds), Provider SCI (per model).
       - *Highlighted aggregate (Always visible)*: **CONSUMER SCI (QA)** (g or mg CO₂e per 1,000 correct predictions).
       - *Mini summary (Always visible)*: Compact rows displaying standard Consumer SCI (per 1k preds) and Provider SCI (per model).
   - **Global & Individual Controls**:
     - An **[ Expand details ] / [ Collapse details ]** toggle button in the header expands or collapses all breakdowns simultaneously.
     - Individual `▸ Detailed listing` disclosure toggles on each card allow inspecting specific models without affecting others.
   - **Efficiency**:
     - Energy / 1k Correct and Consumer SCI (QA).
     - Relative **Efficiency Band (A–E)** grade badge.
     - Clean card footer focused on efficiency grade without extraneous comparative observation clutter.
     - **Interactive focus**: Clicking any card focuses that model across the charts below.
4. **Interactive SVG Charts (Separated Energy & Carbon Curves)**:
   - **Accuracy over time**: Smooth sawtooth curves showing continuous drift degradation, threshold lines, and hover inspection crosshairs.
   - **Cumulative Energy over time**: Clear time series in Wh starting at initial training, climbing with daily inference, and stepping up by retraining energy cost.
   - **Cumulative Carbon over time**: Dedicated time series in gCO₂e combining operational electrical emissions and hardware embodied footprint.
5. **Action Banner (at bottom)**:
   - Assessment completion notice with a prominent **[ Next: Which Model to Deploy → ]** button.

---

### Tab 4 · 04 Which Model to Deploy (Model Selection Rules under Drift)
A dedicated decision engine comparing production deployment choices against standard AutoML offline practice:

1. **Streamlined Rule Selection (The Core 3 Perspectives)**:
   - Focuses strictly on the three most critical selection strategies in machine learning engineering:
     - **Rule 1 · Highest development accuracy** *(AutoML Baseline)*: The conventional benchmark heuristic used by AutoML and static test suites. Ranks models by offline pre-deployment accuracy, blind to real-world continuous data drift and recurring retraining penalties.
     - **Rule 2 · Highest operational accuracy**: Prioritizes raw operational performance, selecting the candidate that sustains the highest time-averaged operational accuracy across production inferences under drift.
     - **Rule 3 · Lowest Consumer SCI (QA)**: The green utility standard under ISO/IEC 21031:2024. Selects the model with the lowest quality-adjusted carbon footprint per 1,000 *correct* predictions delivered ($R = 1,000\text{ correct}$), explicitly penalizing model errors and drift-induced accuracy loss.
2. **Standardized Three-Information Outcome Card**:
   - Each card for the selected model consistently displays the exact same three primary metrics:
     - **Header**: Model name, categorical color dot, and `[AutoML Baseline]` badge if applicable, along with initial baseline accuracy, retrain count, and Total CPU Time.
     - **Box 1 · Total Energy**: Formatted in Wh/kWh with development vs deployment sub-breakdown, accompanied by dual comparison badges against both the AutoML baseline and the fleet's lowest-energy model (e.g. `★ #1 Lowest energy in fleet` or `↑ +74.8 Wh (+208%) vs Logistic regression`).
     - **Box 2 · Operational Accuracy**: Operational accuracy score with min–max drift range, accompanied by dual comparison badges against both the AutoML baseline and the fleet's top operational accuracy model (e.g. `★ #1 Highest accuracy in fleet` or `↓ −3.10 pts vs Random forest (91.10%)`).
     - **Box 3 · Consumer SCI (QA)**: Quality-adjusted carbon intensity per 1,000 correct predictions delivered, accompanied by dual comparison badges against both the AutoML baseline and the fleet's greenest model (e.g. `★ #1 Lowest carbon / QA in fleet` or `↑ +562% vs Logistic regression (0.09 mg/1k)`).
     - **Selection Outcome Summary Takeaway**: A dedicated 3-part synthesis at the base of the card:
       - **What the selected model optimizes for**: Explains whether the model targets pre-deployment benchmark score, sustained live predictive accuracy under drift, or carbon-efficient correct inferences.
       - **Relation to best-performing models in key metrics**: Explicitly compares the chosen model against the fleet's top performers in **Operational Accuracy** and **Consumer SCI (QA)**, quantifying accuracy deltas, carbon percentage savings, or highlighting "win-win" alignments where a single model wins on both axes.
       - **Strategic Insight**: Provides actionable guidance on whether AutoML's baseline choice incurs a hidden retraining tax or whether an alternative architecture offers superior lifecycle efficiency.
     - **Focus Action Button**: **[ Focus <Model> in Assessment & Charts → ]** selects that candidate model and transitions directly to Tab 03.
3. **Action Banner (at bottom)**:
   - Technical specifications guidance notice with a prominent **[ Next: Methodology → ]** button.

---

### Methodology Tab (Right-Aligned Technical Specification)
Positioned on the far-right of the navigation bar without a number prefix and indicated by an open-book specification icon, the **Methodology** tab provides a comprehensive 9-part reference guide:
1. **§1 AutoML Blind Spot**: Theoretical contrast between static test accuracy ranking ($\operatorname{argmax} [ Acc_{\text{dev}} ]$) and operational Lifecycle Assessment ($\operatorname{argmin} [ \text{Consumer SCI (QA)} ]$).
2. **§2 Lifecycle Architecture**: Visual flowchart displaying the separation between Stage 1 Offline Development (1× fitting + 10k eval $\to$ Provider SCI) and Stage 2 Online Deployment under continuous drift ($R_{\text{daily}}$ daily requests + drift $m$ + $K_{\text{retrain}}$ retrains $\to$ Consumer SCI & Consumer SCI (QA)).
3. **§3 Drift & Retraining Math**: Closed-form continuous degradation equation $Acc(n)$, daily drop rate $\Delta_{\text{daily}}$, cycle length $(\text{baseline} - \tau) / \Delta_{\text{daily}}$, and exact $O(1)$ analytical integration of operational accuracy $\overline{Acc}$ via piecewise trapezoids.
4. **§4 Compute & Energy Formulation**: Clear mathematical boundary split between development and deployment CPU runtime, converted to electrical Watt-hours (Wh).
5. **§5 SCI for AI (ISO/IEC 21031:2024)**: Standardized formula $\text{SCI} = (O + M) / R = ((E \times I \times \text{PUE}) + M) / R$, detailing operational electricity ($E$), location-based grid intensity ($I$), Scope 3 embodied hardware carbon ($M$), modern datacenter PUE (1.20), and strict prohibition of market carbon offsets.
6. **§6 Functional Boundaries & Adapting for Model Mistakes**: Mathematical derivation proving $\text{Consumer SCI (QA)} = \frac{\text{Consumer SCI}}{\overline{Acc} / 100}$, explaining how inaccurate inferences shrink the effective functional denominator and directly penalize degraded models.
7. **§7 Live Model Verification Trace**: Interactive dynamic parameter inspector with model selector pills, allowing users to trace real-time arithmetic calculations for their candidate models under active slider settings.
8. **§8 Hardware & Grid Calibration**: Complete calibration tables covering Boavizta server life-cycle assessments (Small, Medium, Large instances) and IEA regional grid carbon intensities (50 to 650 gCO₂e/kWh).
9. **§9 Standards, Primary References & Literature**: Structured bibliography linking directly to authoritative international standards (ISO/IEC 21031:2024, GSF SCI for AI), LCA datasets (Boavizta, IEA, Ember), and foundational peer-reviewed literature (Gama et al. 2014, Strubell et al. 2019).

---

## 5. Performance & Memory Optimizations

To guarantee zero memory spikes during interaction:
1. **O(1) Analytical Simulation**: Closed-form mathematical formulas compute retrain counts, operational energy, and embodied carbon without allocating temporary arrays.
2. **O(1) Constant-Time Lookups**: Hover crosshairs evaluate accuracy, energy, and carbon via modulo arithmetic (`tDay % cycle`) with zero array filtering.
3. **Bounded SVG Sampling**: SVG paths are capped at an adaptive resolution of max 250 points, perfectly matching monitor pixel density.
4. **Capped Marker Nodes**: Circle markers are limited to at most 30 DOM nodes to prevent SVG DOM explosion.
5. **requestAnimationFrame Throttling**: Slider inputs are debounced to 60 FPS, eliminating redundant renders during drag gestures.

---

## 6. Default Configuration & Preset Values

### Default Candidate ML Models

| Model | Baseline Accuracy ($Acc_{\text{dev}}$) | Training Time ($t_{\text{train}}$) | Inference Latency ($t_{\text{inf}}$) | Robustness ($\text{rob}$) |
|---|---|---|---|---|
| **Deep neural net** (Slot 1) | 93.0% | 1,200 s (20 min) | 0.50 ms / pred | 3.2 pts / 100k |
| **Random forest** (Slot 2) | 92.0% | 180 s (3 min) | 0.15 ms / pred | 0.2 pts / 100k |
| **Logistic regression** (Slot 3) | 84.0% | 15 s | 0.02 ms / pred | 0.5 pts / 100k |

### Default Operational Parameters
- **Drift Magnitude ($m$)**: `0.50` (Moderate drift preset)
- **Deployment Duration**: `12` months (360 days)
- **Daily Requests ($R_{\text{daily}}$)**: `5,000` requests / day ($R_{\text{total}} = 1{,}800{,}000$ total requests)
- **Accuracy Floor**: `50.0%`
- **Retraining Policy**: Accuracy cutoff threshold $\tau = 82.0\%$ (lower than all candidate baseline accuracies; yields 2 retrains for Deep neural net, 0 for Random forest, 2 for Logistic regression). Includes a sub-threshold fallback option to retrain periodically every $X = 30$ days.
- **Machine**: Medium (`m5.xlarge`, 4 vCPU, 15.9 W, embodied rate $2.854\text{ gCO}_2\text{e/h}$)
- **Grid Carbon Intensity ($I$)**: `230 gCO₂e/kWh` (Cleaner grid preset)
- **Hardware Utilization ($U$)**: `100%`
- **Facility PUE**: `1.20`
- **Active Selection Rule**: Rule 1 (Highest development accuracy — AutoML Baseline)

### Canonical Simulation Outcomes (Under Default Conditions)

Under these canonical default settings, the simulation demonstrates the AutoML selection paradox:

| Model | Dev Acc ($Acc_{\text{dev}}$) | Operational Acc ($\overline{Acc}$) | Retrains ($K_{\text{retrain}}$) | Total Energy ($E_{\text{total}}$) | Lifecycle Carbon ($C_{\text{total}}$) | Consumer SCI (QA) | Outcome / Fleet Status |
|---|---|---|---|---|---|---|---|
| **Deep neural net** | **93.0%** (Winner) | 90.2% | 2 | 670 Wh | 185 gCO₂e | 0.088 g / 1k correct | Rule 1 Pick (AutoML Baseline) · Brittle to drift |
| **Random forest** | 92.0% | **91.1%** (Winner) | **0** (Winner) | 75 Wh | 21 gCO₂e | 0.011 g / 1k correct | Rule 2 Pick · Saves 88.8% energy vs baseline |
| **Logistic regression** | 84.0% | 83.2% | 2 (fallback) | **0.4 Wh** (Winner) | **0.1 gCO₂e** (Winner) | **0.0001 g / 1k** (Winner) | Rule 3 Pick · Ultra-green, saves >99.9% carbon |

---

## 7. Standards, References & External Sources

This application's mathematical accounting boundaries, data drift dynamics, and carbon coefficients are derived from official international standards, open life-cycle databases, and peer-reviewed computer science literature:

### 7.1 International Standards & Industry Specifications

1. **ISO/IEC 21031:2024 Standard**
   - *Title*: Information technology — Environmental sustainability — Software carbon intensity (SCI) specification
   - *Organization*: International Organization for Standardization (ISO) / International Electrotechnical Commission (IEC) JTC 1/SC 39
   - *Link*: [ISO/IEC 21031:2024](https://www.iso.org/standard/86612.html)
   - *Role in Project*: Formal international standard defining the Software Carbon Intensity metric $\text{SCI} = (O + M) / R$. Establishes the normative requirement that location-based physical grid emission factors must be utilized, and strictly prohibits deducting market-based carbon offsets or unbundled renewable energy certificates (RECs) from the software intensity score.

2. **Green Software Foundation (GSF) — Software Carbon Intensity (SCI) Specification**
   - *Title*: Software Carbon Intensity (SCI) Specification v1.0
   - *Organization*: Green Software Foundation (Linux Foundation)
   - *Link*: [GSF SCI Specification](https://grnsft.org/sci) · [Green Software Foundation](https://greensoftware.foundation/)
   - *Role in Project*: Standardizes operational ($O$) and embodied ($M$) software carbon boundaries and hardware allocation models ($M = TE \times (TR / EL) \times (\text{vCPU} / \text{total\_vCPU}) \div U$).

3. **Green Software Foundation (GSF) — SCI for AI Working Group**
   - *Title*: Software Carbon Intensity for Artificial Intelligence (SCI for AI)
   - *Repository*: [Green-Software-Foundation/sci-ai](https://github.com/Green-Software-Foundation/sci-ai)
   - *Role in Project*: Extends the foundational SCI standard to machine learning and AI workloads. Introduces lifecycle persona separation:
     - **Provider Persona**: Scopes initial training and offline benchmarking per model ($R = 1\text{ model}$).
     - **Consumer Persona**: Scopes live inference serving and drift retraining per query volume ($R = 1{,}000\text{ predictions}$).
     - **Quality-Adjusted Functional Unit (Consumer SCI (QA))**: Defines functional utility as $R = \text{correct predictions} / 1{,}000$, penalizing models whose inaccuracy degrades real-world value.

### 7.2 Open Life-Cycle Assessment (LCA) & Grid Databases

4. **Boavizta Digital Technologies Environmental Footprint Database**
   - *Organization*: Boavizta Working Group (Open Data / Open Source)
   - *Link*: [Boavizta Data Visualizer](https://dataviz.boavizta.org/) · [Boavizta Working Group](https://boavizta.org/)
   - *Role in Project*: Provides empirical life-cycle assessment (LCA) data for enterprise cloud servers (Scope 3 manufacturing emissions). Used to calibrate the baseline embodied footprint: $TE = 1{,}200\text{ kg CO}_2\text{e}$ for a 48 vCPU enterprise rack server over an expected lifespan $EL = 4\text{ years}$ ($35{,}040\text{ hours}$), translating to $0.7135\text{ gCO}_2\text{e}$ per vCPU-hour at 100% utilization.

5. **International Energy Agency (IEA) Emission Factors**
   - *Title*: IEA Emissions Factors 2023 Database
   - *Organization*: International Energy Agency (IEA)
   - *Link*: [IEA Emissions Factors Database](https://www.iea.org/data-and-statistics/data-product/emissions-factors-2023)
   - *Role in Project*: Calibrates regional location-based grid emission intensities ($I$):
     - Low-carbon grids (e.g., Sweden, France): $\sim 50\text{ gCO}_2\text{e/kWh}$
     - European mixed renewable/gas average: $\sim 230\text{ gCO}_2\text{e/kWh}$
     - Global electricity generation mean: $\sim 450\text{ gCO}_2\text{e/kWh}$
     - Fossil/coal dominant grids (e.g., India, Poland): $\sim 650\text{ gCO}_2\text{e/kWh}$

6. **Ember Climate — Global Electricity Review**
   - *Organization*: Ember Climate
   - *Link*: [Ember Electricity Data](https://ember-energy.org/countries-and-regions/)
   - *Role in Project*: Provides annual country-by-country marginal and average carbon intensity benchmarks for global data center regions.

### 7.3 Foundational Academic Literature

7. **Gama et al. (2014) — Concept Drift in Machine Learning**
   - *Citation*: Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys (CSUR), 46(4), 1–37.
   - *DOI*: [10.1145/2523813](https://doi.org/10.1145/2523813)
   - *Role in Project*: Establishes theoretical principles for modeling continuous distribution drift over streaming data, performance decay trajectories, and detection/retraining trigger mechanisms.

8. **Strubell et al. (2019) — Energy and Carbon in Modern ML**
   - *Citation*: Strubell, E., Ganesh, A., & McCallum, A. (2019). *Energy and Policy Considerations for Deep Learning in NLP*. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 3645–3650.
   - *DOI*: [10.18653/v1/P19-1355](https://doi.org/10.18653/v1/P19-1355)
   - *Role in Project*: Seminal paper highlighting that continuous iterative training, hyperparameter searches, and periodic model updates consume orders of magnitude more lifecycle energy than single inference passes.

9. **Patterson et al. (2021) — Carbon Footprint of Machine Learning**
   - *Citation*: Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L. M., Rothchild, D., So, D. R., Texier, M., & Dean, J. (2021). *Carbon Emissions and Large Neural Network Training*. arXiv preprint arXiv:2104.10350.
   - *Link*: [arXiv:2104.10350](https://arxiv.org/abs/2104.10350)
   - *Role in Project*: Validates the importance of the "4Ms" (Model, Machine, Mechanization/PUE, and Map/Grid Carbon Intensity) in calculating holistic machine learning carbon footprints.

10. **Schwartz et al. (2020) — Green AI**
    - *Citation*: Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). *Green AI*. Communications of the ACM, 63(12), 54–63.
    - *DOI*: [10.1145/3381831](https://doi.org/10.1145/3381831)
    - *Role in Project*: Foundational theoretical inspiration for this project: establishes the Green AI paradigm and advocates for efficiency as a primary evaluation criterion in AI alongside accuracy, motivating the trade-off analysis between benchmark performance and real-world operational cost.

