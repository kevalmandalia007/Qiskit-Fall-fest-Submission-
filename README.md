# Latent Quantum Neural ODEs for Chaotic Environments

**How much quantum memory does it take to imitate a chaotic environment?**

Qiskit Fall Fest 2026 · Track 2 (Quantum Application Engineering) with Track 3 elements · MIT License

## Problem Statement

### Background

A quantum system is never perfectly isolated. A qubit that interacts with its surroundings leaks information into them, and its observed dynamics stop being governed by a closed Schrödinger equation.

- If the environment is **integrable**, the leaked information flows back as revivals. These are memory effects, also called non-Markovian dynamics.
- If the environment is **chaotic**, the information is scrambled across many degrees of freedom and effectively never returns.

Modelling such reduced dynamics usually means either simulating the full environment, whose cost grows exponentially with its size, or assuming it is memoryless, which misses the revivals. A compact, learnable model of the environment's effect would avoid both. It is unknown, however, how much hidden memory such a model needs, whether that depends on chaos, and whether quantum memory is a better modelling choice than classical memory.

### Formal setup

**The true system.** An observed qubit $`S`$ (the bell) is coupled to an unobserved environment $`B`$ of $`N = 5`$ qubits:

```math
H = g\,Z_0 Z_1 + J\sum_{i=1}^{N-1} Z_i Z_{i+1} + \sum_{i=0}^{N}\left(h_x X_i + h_z Z_i\right),
```

The parameter $`h_z`$ selects the environment:

```math
h_z = 0 \;\;\text{(integrable)}, \qquad h_z = 0.809 \;\;\text{(chaotic)}.
```

The system starts in the product state

```math
\rho(0) = \rho_S(0) \otimes \left(|{+y}\rangle\langle{+y}|\right)^{\otimes N},
```

where $`\rho_S(0)`$ is one of the six Bloch-sphere directions $`\pm x, \pm y, \pm z`$.

**Data.** Only the bell's Bloch vector is observed:

```math
r_a(t) = \mathrm{Tr}\!\left[\sigma_a^{(S)}\,\rho(t)\right], \qquad a \in \{x, y, z\},
```

on a time grid, giving the trajectories

```math
\mathcal{D} = \big\{\, \vec r^{\,(d)}(t) \;:\; d = 1,\dots,6,\;\; t \in [0, 20] \,\big\}.
```

**The model class.** A latent quantum neural ODE consists of the bell plus $`k`$ hidden qubits, initialised in $`|0\rangle^{\otimes k}`$ and evolving under a trainable generator:

```math
\frac{d\rho_\theta}{dt} = -i\,[H_\theta, \rho_\theta] \;+\; \sum_{q=1}^{k}\gamma_q\,\mathcal{D}[\sigma_q^-](\rho_\theta),
\qquad
H_\theta = \sum_j \theta_j P_j ,
```

where $`P_j`$ runs over all 1-local and nearest-neighbour 2-local Paulis. The prediction is

```math
r^\theta_a(t) = \mathrm{Tr}\!\left[\sigma_a^{(S)}\rho_\theta(t)\right].
```

Setting $`\gamma_q = 0`$ gives the unitary model; $`\gamma_q \geq 0`$ trainable gives the leaky model.

### Task

Given only the bell's trajectories on the training window $`t \in [0, 10]`$, learn $`(\theta, \gamma)`$ so that the model **forecasts** the bell on the unseen window $`t \in (10, 20]`$. The training loss is

```math
\min_{\theta,\gamma}\;\; \mathcal{L}(\theta,\gamma) = \frac{1}{N_{\mathrm{train}}}\sum_{d,\,t \le 10,\,a}\Big(r_a^{\theta}(d,t) - r_a(d,t)\Big)^2 .
```

### Research questions

1. **Memory requirement.** What is the smallest number of hidden qubits $`k`$ that reproduces and forecasts the bell's dynamics to tolerance $`\varepsilon`$?
2. **Effect of chaos.** At fixed $`k`$, how does forecast quality differ between the integrable and chaotic environments, and why?
3. **Quantum vs classical memory.** Under identical training conditions and matched parameter budgets, does the latent quantum model outperform classical models (a linear state-space model and an augmented neural ODE)? Which physical ingredient, coherence or dissipation, accounts for the difference?
4. **Hardware feasibility.** Can the trained model be executed on a current IBM quantum processor, and recovered with error mitigation to near the shot-noise limit?

### Evaluation metrics

| Metric | Definition | Purpose |
|---|---|---|
| Train MSE | loss on $`t \le 10`$ | fitting ability |
| Test MSE | mean squared error on $`10 < t \le 20`$ | forecasting ability |
| Valid time $`T_{\mathrm{valid}}`$ | first time the smoothed RMS error exceeds ε (0.10, 0.15, 0.20) | how long predictions remain correct |
| Forecast horizon | $`T_{\mathrm{valid}} - 10`$ | correct prediction beyond the training data |
| Hardware RMS error | RMS deviation from the ideal model, before and after mitigation | fidelity of execution on a noisy device |

The valid time is computed from the RMS error

```math
e(t) = \sqrt{\tfrac{1}{18}\textstyle\sum_{d,a}\big(r_a^\theta(d,t) - r_a(d,t)\big)^2}
```

and its running mean $`\bar e(t)`$ over 1 time unit:

```math
T_{\mathrm{valid}} = \min\{\, t : \bar e(t) > \varepsilon \,\}.
```

### Constraints for a fair comparison

Every model, quantum or classical, must satisfy the following:

- same data and the same train/test split;
- same loss and the same optimiser (L-BFGS-B, 6 random restarts, fixed seed);
- parameter counts matched within ±5 (≈ 15, 27, 39);
- the initial condition $`\vec r(0)`$ reproduced exactly, so no model gets the first point for free.

The chaotic and integrable character of the environments must be verified independently, via the level-spacing ratio ⟨r⟩ against the Poisson (0.386) and GOE (0.531) values, rather than assumed.

### Scope

**In scope:** small environments ($`N = 5`$) whose exact dynamics can be computed as ground truth. Models are trained by classical simulation with exact gradients, and the trained models are executed on emulated (and optionally real) IBM hardware.

**Out of scope:** claims of computational quantum advantage. The question is which **model structure** (coherent quantum memory, dissipation, classical latent state) best captures open-system dynamics with and without chaos, not whether a quantum computer computes it faster.

---

## Summary

We study an open quantum system: one observed qubit (the *bell*) coupled to an unobserved 5-qubit spin chain (the *room*). The room is either integrable or chaotic. We learn the bell's reduced dynamics with a **latent quantum neural ODE**, which is the bell plus $`k`$ hidden helper qubits evolving under a trainable Hamiltonian. We then benchmark it against classical models of matched size.

| Result | Integrable room | Chaotic room |
|---|---|---|
| Level-spacing ratio ⟨r⟩ (Poisson ≈ 0.386, GOE ≈ 0.531) | 0.309 | 0.527 |
| Forecast horizon, unitary model, k = 3 | +5.8 | +0.7 |
| Test MSE, best classical model | 0.104 | 0.015 |
| Test MSE, leaky quantum model (29 params) | **0.023** | **0.011** |
| Valid time, leaky quantum vs best classical | 10.7 vs 5.8 | 27.6 vs 16.4 |
| Hardware-emulation RMS error, raw → mitigated | 0.082 → 0.047 | 0.067 → 0.041 |

**Main findings**

1. With equal quantum memory, the chaotic environment is forecast about **8× less far** than the integrable one. A closed finite model re-phases into *false echoes*.
2. Classical models fail on coherent echoes; unitary quantum models fail on chaos.
3. Adding trainable dissipation to the hidden qubits (damped pseudomodes) yields the best model in the chaotic room and a near-best model in the integrable room.

> No computational quantum advantage is claimed. All claims concern model structure (inductive bias) on small, classically simulable systems, tested under identical training rules.

---

## 1. Physical system

The bell (qubit 0) couples to a mixed-field Ising chain (qubits 1–5):

```math
H = g\,Z_0 Z_1 \;+\; J\sum_{i=1}^{4} Z_i Z_{i+1} \;+\; \sum_{i=0}^{5}\left(h_x X_i + h_z Z_i\right),
\qquad g = 0.5,\; J = 1,\; h_x = 0.9045 .
```

| Room | $`h_z`$ | Character |
|---|---|---|
| Integrable ("simple") | 0 | transverse-field Ising, information returns as echoes |
| Chaotic | 0.8090 | mixed-field Ising, information is scrambled |

Choosing $`g \neq J`$ breaks the reflection symmetry of the chain.

**Initial states.** The bell starts along one of six directions, $`\pm x, \pm y, \pm z`$. Every room qubit starts in $`|{+y}\rangle`$, which satisfies $`\langle X\rangle = \langle Y\rangle = \langle Z\rangle = 0`$. The room therefore carries no information of its own and starts near infinite temperature.

**Observable.** The bell's Bloch vector:

```math
r_a(t) = \mathrm{Tr}\!\left[\sigma_a^{(0)}\,\rho(t)\right], \qquad a \in \{x, y, z\}.
```

Its length $`|\vec r(t)|`$ measures how much information the bell still holds.

| Room | $`\lvert\vec r\rvert`$ for $`t > 5`$ (avg. over 6 directions) |
|---|---|
| Integrable | 0.56 – 0.93 (repeated revivals) |
| Chaotic | 0.19 – 0.36 (decays and stays low; finite-size floor) |

### 1.1 Simulation

**Exact evolution.** We diagonalise $`H = V\,\mathrm{diag}(E)\,V^\dagger`$ and evolve as follows:

```math
|\psi(t)\rangle = V\, e^{-iEt}\, V^\dagger |\psi(0)\rangle .
```

**Circuits.** We use a second-order Suzuki–Trotter step built from `rx`, `rzz` and `rz` gates, with bonds applied in even/odd layers:

```math
e^{-iH\,dt} \approx e^{-iH_X\,dt/2}\; e^{-i(H_{ZZ}+H_Z)\,dt}\; e^{-iH_X\,dt/2} .
```

The circuits are evaluated with `EstimatorV2`, so the same code runs on IBM hardware.

| Room | $`dt`$ | Max error, $`t \le 10`$ |
|---|---|---|
| Integrable | 0.25 | 0.212 |
| Integrable | 0.10 | 0.037 |
| Chaotic | 0.25 | 0.047 |
| Chaotic | 0.10 | 0.010 |

The error scales as $`O(dt^2)`$. The integrable room is more sensitive because revivals depend on precise phases.

### 1.2 Chaos diagnostic

We use the mean ratio of consecutive level spacings over the middle half of the spectrum, computed on a 9-qubit version of each room:

```math
r_n = \frac{\min(s_n, s_{n+1})}{\max(s_n, s_{n+1})}, \qquad s_n = E_{n+1} - E_n .
```

| Room | ⟨r⟩ measured | Reference |
|---|---|---|
| Integrable | 0.309 | Poisson 0.386 (lower with extra symmetries) |
| Chaotic | 0.527 | GOE 0.531 |

---

## 2. Model: latent quantum neural ODE

The model has the bell plus $`k`$ hidden helper qubits in a chain. The helpers start in $`|0\rangle`$ and are never measured.

```math
\frac{d}{dt}|\Psi(t)\rangle = -i\,H_\theta\,|\Psi(t)\rangle,
\qquad
H_\theta = \sum_j \theta_j P_j,
\qquad
r_a^\theta(t) = \langle \Psi(t)|\,\sigma_a^{(0)}\,|\Psi(t)\rangle .
```

**Ansatz.** The terms $`P_j`$ are all single-qubit Paulis on every qubit plus all nine two-qubit Paulis on each neighbouring pair, giving

```math
N_\theta(k) = 3(k+1) + 9k = 12k + 3 .
```

| $`k`$ | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| Parameters | 3 | 15 | 27 | 39 | 63 |
| Hilbert-space dimension $`2^{k+1}`$ | 2 | 4 | 8 | 16 | 64 |

**Interpretation**

- The model is a neural ODE whose vector field is a learned Schrödinger generator.
- The helpers act as latent dimensions, as in augmented and latent neural ODEs, or as pseudomodes in open-system theory.

**Two structural facts**

- **$`k = 0`$ cannot fit.** A lone qubit under unitary evolution keeps $`|\vec r(t)| = 1`$ for all $`t`$.
- **$`k = 5`$ represents the real room exactly.** $`H`$ is itself a chain of 1- and 2-local Paulis, and the $`|{+y}\rangle`$ initial states are absorbed by a local basis change that preserves the ansatz.

### 2.1 Training

**Loss.** Mean squared error over 6 directions, all training times and 3 components:

```math
\mathcal{L}(\theta) = \frac{1}{N}\sum_{d,\,t,\,a}\Big(r_a^\theta(d,t) - r_a^{\mathrm{true}}(d,t)\Big)^2 .
```

**Exact gradient (Daleckii–Krein).** With $`H_\theta = V\,\mathrm{diag}(E)\,V^\dagger`$:

```math
\frac{\partial\, e^{-iH_\theta t}}{\partial \theta_j}
= V\Big[\big(V^\dagger P_j V\big)\circ \Phi(t)\Big]V^\dagger,
\qquad
\Phi_{mn}(t) =
\begin{cases}
\dfrac{e^{-iE_m t} - e^{-iE_n t}}{E_m - E_n}, & E_m \neq E_n,\\[2mm]
-it\,e^{-iE_m t}, & E_m = E_n .
\end{cases}
```

One diagonalisation gives all gradients. On hardware, the same gradient would be obtained with the parameter-shift rule.

**Protocol**

- Train on $`0 \le t \le 10`$ (41 points × 6 directions).
- Test on $`10 < t \le 20`$, which is never seen in training, so the test is a forecast.
- Optimiser: L-BFGS-B, 6 random initialisations (seed 0), keep the best.

### 2.2 Verification

| Check | Result |
|---|---|
| Analytic gradient vs central finite differences | relative error 3.8 × 10⁻⁹ |
| Model compiled to `rz, sx, x, cx` vs matrix model (k = 2, t = 2) | 0.130 at 10 steps, 0.007 at 40 steps |
| Leaky model with γ → 0 vs unitary model | 3.6 × 10⁻¹⁵ |

---

## 3. Results I: memory vs chaos

### 3.1 Scaling sweep (unitary model)

| Room | k | Params | Train MSE | Test MSE | Restart spread (best – median) |
|---|---|---|---|---|---|
| Integrable | 1 | 15 | 0.0986 | 0.1138 | 0.099 – 0.145 |
| Integrable | 2 | 27 | 0.0130 | 0.0329 | 0.013 – 0.032 |
| Integrable | 3 | 39 | 0.0025 | 0.0204 | 0.0025 – 0.0077 |
| Chaotic | 1 | 15 | 0.0519 | 0.1986 | 0.052 – 0.095 |
| Chaotic | 2 | 27 | 0.0145 | 0.1061 | 0.015 – 0.021 |
| Chaotic | 3 | 39 | 0.0030 | 0.0463 | 0.0030 – 0.0060 |

### 3.2 Forecast horizon

First define the RMS error at each time:

```math
e(t) = \sqrt{\frac{1}{18}\sum_{d,a}\Big(r_a^\theta(d,t) - r_a^{\mathrm{true}}(d,t)\Big)^2} .
```

Let $`\bar e(t)`$ be $`e(t)`$ smoothed with a running mean over 1 time unit. The valid time and forecast horizon are then:

```math
T_{\mathrm{valid}} = \min\{\,t : \bar e(t) > \varepsilon\,\},
\qquad
\text{horizon} = T_{\mathrm{valid}} - 10 .
```

| Room | k | $`T_{\mathrm{valid}}`$ at ε = 0.10 | ε = 0.15 | ε = 0.20 |
|---|---|---|---|---|
| Integrable | 1 | 0.25 | 0.45 | 0.65 |
| Integrable | 2 | 1.05 | 1.90 | 11.95 |
| Integrable | 3 | 11.00 | **15.80** | 17.25 |
| Chaotic | 1 | 0.80 | 1.25 | 1.90 |
| Chaotic | 2 | 1.25 | 2.25 | 10.45 |
| Chaotic | 3 | 10.35 | **10.70** | 11.00 |

**Observations**

- **Memory is the bottleneck.** For $`k \le 2`$ the models already fail inside the training window.
- **Same memory, different forecast.** At $`k = 3`$ both rooms are fitted equally well (train MSE 0.0025 vs 0.0030). The forecast horizon, however, is **+5.8 vs +0.7** (≈ 8×), and the ordering holds for every tolerance.

**Mechanism.** A closed model of dimension $`D = 2^{k+1}`$ produces a finite sum of Bohr frequencies:

```math
r_a^\theta(t) = \sum_{m,n=1}^{D} c^{(a)}_{mn}\, e^{-i(E_m - E_n)t},
```

which is quasi-periodic and must eventually re-phase into a revival.

- The integrable room's signal is built from a few regular frequencies, so the model can continue it.
- The chaotic room has level repulsion and irregular spacings, so its signal does not re-phase on these timescales. The model therefore produces a **false echo** after $`t = 10`$.

---

## 4. Results II: quantum vs classical

### 4.1 Contenders

All models obey the same rules:

- the same data and split;
- the same loss;
- the same optimiser (L-BFGS-B, 6 restarts, seed 0);
- approximately matched parameter counts;
- an initial condition fixed exactly at $`\vec r(0)`$.

**(a) Classical linear state-space model** (system identification), with $`\mathbf z = (\vec r, \mathbf h)\in\mathbb R^d`$ and output $`\mathbf z_{1:3}`$:

```math
\dot{\mathbf z} = A\,\mathbf z + \mathbf b,
\qquad
\mathbf h(0) = W\,[\vec r_0;\,1] .
```

This gives $`d^2 + d + 4(d-3)`$ parameters: 12, 24, 38 for $`d = 3, 4, 5`$.

**(b) Classical augmented neural ODE**, with $`\mathbf z(0) = (\vec r_0, 0, 0)`$, solved with RK4:

```math
\dot{\mathbf z} = W_2 \tanh\!\left(W_1 \mathbf z + \mathbf b_1\right) + \mathbf b_2 .
```

Hidden widths 1, 2, 3 give 16, 27, 38 parameters.

**(c) Quantum, unitary.** The model of Section 2, with 15, 27, 39 parameters.

**(d) Quantum, leaky helpers.** Each helper is amplitude-damped toward $`|0\rangle`$ at a trainable rate $`\gamma_q = \mathrm{softplus}(\phi_q) \ge 0`$:

```math
\frac{d\rho}{dt} = -i[H_\theta, \rho]
+ \sum_{q \in \mathrm{helpers}} \gamma_q \left( \sigma_q^- \rho\, \sigma_q^+ - \tfrac{1}{2}\{\sigma_q^+ \sigma_q^-, \rho\} \right).
```

The bell never dissipates directly, so all forgetting passes through the learned quantum memory. This variant has 16 or 29 parameters (k = 1, 2).

It is propagated as $`\mathrm{vec}(\rho(t+\Delta)) = e^{\mathcal{L}\Delta}\,\mathrm{vec}(\rho(t))`$ with the row-major superoperator

```math
\mathcal{L} = -i\left(H_\theta \otimes I - I \otimes H_\theta^{T}\right)
+ \sum_q \gamma_q\left( L_q \otimes L_q^{*} - \tfrac12 L_q^\dagger L_q \otimes I - \tfrac12 I \otimes (L_q^\dagger L_q)^{T} \right).
```

### 4.2 Results

Each cell shows test MSE on $`10 < t \le 20`$ / $`T_{\mathrm{valid}}`$ at ε = 0.15.

| Model | Params | Integrable room | Chaotic room |
|---|---|---|---|
| Quantum, unitary, k = 1 | 15 | 0.114 / 0.45 | 0.199 / 1.25 |
| Quantum, unitary, k = 2 | 27 | 0.033 / 1.90 | 0.106 / 2.25 |
| Quantum, unitary, k = 3 | 39 | **0.020 / 15.80** | 0.046 / 10.70 |
| Quantum, leaky, k = 1 | 16 | 0.064 / 0.95 | 0.017 / 16.00 |
| Quantum, leaky, k = 2 | 29 | 0.023 / 10.70 | **0.011 / 27.60** |
| Classical linear, d = 3 | 12 | 0.115 / 2.30 | 0.017 / 16.00 |
| Classical linear, d = 4 | 24 | 0.104 / 5.55 | 0.015 / 16.35 |
| Classical linear, d = 5 | 38 | 73.5 / 2.65 | 0.016 / 16.15 |
| Classical neural ODE, width 1 | 16 | 0.404 / 0.00 | 0.167 / 0.00 |
| Classical neural ODE, width 2 | 27 | 0.296 / 0.90 | 0.062 / 2.25 |
| Classical neural ODE, width 3 | 38 | 0.755 / 5.80 | 0.018 / 16.05 |

**Findings**

1. **Classical models cannot forecast coherent revivals.** Every classical model has integrable-room test MSE ≥ 0.10. Added capacity overfits: the width-3 neural ODE reaches train MSE 0.019 but test MSE 0.76. The $`d = 5`$ linear model learns an unstable mode, with $`\max \mathrm{Re}\,\lambda(A) = +0.39`$, so its forecast diverges.
2. **Unitary quantum models cannot forget.** Damped classical models beat them in the chaotic room at every size.
3. **Coherent memory plus controlled dissipation has the right inductive bias.**
   - The leaky model is the best model in the chaotic room at every size tested.
   - At 29 parameters it stays valid to $`t = 27.6`$, which is 1.7× the best classical model, with the lowest test error overall (0.011).
   - It is near-best in the integrable room (0.023 vs 0.020 for unitary k = 3).
   - This reproduces the damped-pseudomode structure of open-system theory, learned from data.

---

## 5. Results III: hardware

**Model.** We run the trained unitary $`k = 2`$ models (3 qubits) for both rooms at $`t = 0, 1, \dots, 20`$ over all 6 directions.

**Compilation.** Each prediction compiles $`e^{-iH_\theta t}`$ exactly as one 3-qubit unitary. This gives **≈ 29 CZ gates for any $`t`$**, against ≈ 2800 CX for 40 Trotter steps at $`t = 2`$, so circuit depth is independent of evolution time.

**Qubit selection.** We choose the connected chain minimising $`\epsilon_{\mathrm{CZ}}(a,m) + \epsilon_{\mathrm{CZ}}(m,c) + 3\,\epsilon_{\mathrm{RO}}(a)`$ from backend calibration data. This selects qubits 51–52–37 with 0.5 % readout error on the measured qubit; qubit 0 has 16.7 %.

**Noise model.** Qiskit Aer with the FakeTorino (IBM Heron, 133 qubits) noise model, 8000 shots, 1136 circuits per room.

**Readout-error mitigation**, using single-qubit calibration of $`p_{1|0}, p_{0|1}`$:

```math
\langle Z\rangle = \frac{\langle Z\rangle_{\mathrm{meas}} - \left(p_{0|1} - p_{1|0}\right)}{1 - p_{0|1} - p_{1|0}} .
```

**Zero-noise extrapolation** by unitary folding of every two-qubit gate, followed by Richardson extrapolation:

```math
G \;\rightarrow\; G\,\big(G^\dagger G\big)^{(s-1)/2},\quad s \in \{1, 3, 5\},
\qquad
E_0 \approx \frac{15E_1 - 10E_3 + 3E_5}{8}.
```

RMS error is measured against the ideal model; the shot-noise floor is ≈ 0.011.

| Room | Raw | REM | REM + ZNE |
|---|---|---|---|
| Integrable | 0.082 | 0.079 | **0.047** |
| Chaotic | 0.067 | 0.065 | **0.041** |

- Noise contracts $`\vec r`$ toward the origin, and ZNE removes ≈ 40 % of the error.
- In the chaotic room, unmitigated noise suppresses the false echo. Noise acts as uncontrolled dissipation, the ingredient the leaky model learns in a controlled way.

**Real device.** A notebook cell (off by default) submits the same circuits via `EstimatorV2`. It uses ZNE (factors 1, 3, 5; exponential/linear extrapolation), gate and measurement twirling (TREX), and XpXm dynamical decoupling. A real-hardware run is pending.

---

## 6. Reproducing

```bash
pip install -r requirements.txt jupyter
jupyter notebook qnode_chaos.ipynb      # Run All: ~15–20 min on CPU
```

In Google Colab, upload the notebook and choose *Runtime → Run all*; the first cell installs the Qiskit packages. To use real hardware, set `RUN_ON_HARDWARE = True` in Section 11 and provide an IBM Quantum Platform API key and instance.

**Stack:** qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0, JAX 0.11.2, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8.

**Seeds:** NumPy seed 0 for restarts; transpiler and simulator seed 7.

```
qnode-chaos/
├── qnode_chaos.ipynb   # 12 sections, explanations, pre-run outputs, 29 references
├── figures/            # plots exported from the notebook
├── requirements.txt
├── LICENSE
└── README.md
```

| Notebook part | Sections | Content |
|---|---|---|
| I. Real room | 1–6 | Hamiltonian, exact and Trotter simulation, level statistics, dataset |
| II. Fake room | 7–9 | Model, gradient check, circuit check, training, scaling, horizon |
| III. Comparison | 10 | Linear SSM, augmented NODE, leaky quantum model, stability |
| IV. Hardware | 11 | Qubit selection, noisy emulation, REM + ZNE, real-device cell |
| Summary | 12 | Results, limitations, future work |

---

## 7. Limitations and future work

**Limitations**

- One room size (5 qubits), one training window, one restart seed.
- The largest models are unitary $`k = 3`$ and leaky $`k = 2`$; exact representability requires $`k = 5`$.
- $`T_{\mathrm{valid}}`$ depends on ε; three values are reported.
- The finite chaotic room still fluctuates at long times.
- Hardware results are a noisy emulation.
- Parameter counts are matched within ±5.

**Next steps**

- Run on real IBM hardware.
- Implement helper dissipation on hardware via a partial swap with a sink qubit and mid-circuit reset (a collision model).
- Larger rooms and $`k \ge 4`$, with multiple seeds and error bars.
- Relate the forecast horizon to the scrambling rate measured with OTOCs.

---

## References

1. M. C. Bañuls, J. I. Cirac, M. B. Hastings, PRL **106**, 050405 (2011).
2. M. Suzuki, Commun. Math. Phys. **51**, 183 (1976).
3. V. Oganesyan, D. A. Huse, PRB **75**, 155111 (2007).
4. Y. Y. Atas, E. Bogomolny, O. Giraud, G. Roux, PRL **110**, 084101 (2013).
5. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS (2018).
6. E. Dupont, A. Doucet, Y. W. Teh, *Augmented Neural ODEs*, NeurIPS (2019).
7. Y. Rubanova, R. T. Q. Chen, D. Duvenaud, *Latent ODEs for Irregularly-Sampled Time Series*, NeurIPS (2019).
8. B. M. Garraway, PRA **55**, 2290 (1997).
9. D. Tamascelli, A. Smirne, S. F. Huelga, M. B. Plenio, PRL **120**, 030402 (2018).
10. H.-P. Breuer, E.-M. Laine, J. Piilo, PRL **103**, 210401 (2009).
11. I. Najfeld, T. F. Havel, Adv. Appl. Math. **16**, 321 (1995).
12. K. Mitarai et al., PRA **98**, 032309 (2018); M. Schuld et al., PRA **99**, 032331 (2019).
13. R. H. Byrd, P. Lu, J. Nocedal, C. Zhu, SIAM J. Sci. Comput. **16**, 1190 (1995).
14. A. Javadi-Abhari et al., *Quantum computing with Qiskit*, arXiv:2405.08810 (2024).
15. P. Virtanen et al., Nat. Methods **17**, 261 (2020).
16. N. Wiebe, C. Granade, C. Ferrie, D. G. Cory, PRL **112**, 190501 (2014).
17. K. Fujii, K. Nakajima, Phys. Rev. Applied **8**, 024030 (2017).
18. F. Vatan, C. Williams, PRA **69**, 032315 (2004).
19. J. Pathak, B. Hunt, M. Girvan, Z. Lu, E. Ott, PRL **120**, 024102 (2018).
20. L. Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall (1999).
21. G. Lindblad, Commun. Math. Phys. **48**, 119 (1976).
22. V. Gorini, A. Kossakowski, E. C. G. Sudarshan, J. Math. Phys. **17**, 821 (1976).
23. J. Bradbury et al., *JAX: composable transformations of Python+NumPy programs* (2018).
24. S. Bravyi, S. Sheldon, A. Kandala, D. C. McKay, J. M. Gambetta, PRA **103**, 042605 (2021).
25. K. Temme, S. Bravyi, J. M. Gambetta, PRL **119**, 180509 (2017).
26. T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, W. J. Zeng, IEEE QCE, 306 (2020).
27. J. J. Wallman, J. Emerson, PRA **94**, 052325 (2016).
28. E. van den Berg, Z. K. Minev, K. Temme, PRA **105**, 032620 (2022).
29. L. Viola, E. Knill, S. Lloyd, PRL **82**, 2417 (1999).

---

MIT License © 2026 Keval Mandalia
