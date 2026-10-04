<div align="center">

# Latent Quantum Neural ODEs for Chaotic Environments

### *How much quantum memory does it take to imitate a chaotic environment?*

[![Qiskit](https://img.shields.io/badge/Qiskit-2.5-6929C4?logo=qiskit&logoColor=white)](https://www.ibm.com/quantum/qiskit)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.11-A8B9CC)](https://github.com/jax-ml/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/qnode-chaos/blob/main/qnode_chaos.ipynb)

**Qiskit Fall Fest 2026** · Track 2: Quantum Application Engineering (+ Track 3 elements)

</div>

---

## TL;DR

A qubit inside a larger quantum system leaks information into its surroundings, and sometimes gets it back. We train **small quantum models with hidden "memory" qubits** to imitate this, and benchmark them against classical models of the same size.

| | Finding |
|---|---|
| 🔁 | **Unitary quantum memory** forecasts an *integrable* environment **8× further** than a *chaotic* one: its few frequencies re-phase into **false echoes** |
| 📉 | **Classical models** (linear state-space, neural ODE) **cannot forecast coherent echoes** at any size tested |
| 🏆 | **Quantum memory + a trainable leak** (damped pseudomodes) is **best in the chaotic room** (valid **1.7× longer** than the best classical model) and near-best in the integrable one |
| ⚛️ | The trained model compiles to a **fixed-depth circuit (~29 CZ at any time t)** and is recovered on an emulated **IBM Torino** device with error mitigation |

> **Scope:** no computational quantum advantage is claimed. The results concern **model structure (inductive bias)**, tested against classical baselines under identical training rules.

---

## The idea in one picture

```
        REAL ROOM  (the data)                          FAKE ROOM  (the model)

   🔔 bell ─── 5-qubit Ising chain                 🔔 bell ─── helper ─── helper ...
       │       fixed Hamiltonian H                     │       trainable H_θ  (+ leak rates γ)
       │       integrable  or  chaotic                 │       k hidden qubits = quantum memory
       ▼                                               ▼
   bell's Bloch vector r(t)    ───── fit on t ≤ 10 ─────►   r_θ(t)  ──► forecast t > 10
```

- **Simple (integrable) room:** information leaves the bell and comes back as **echoes**.
- **Chaotic room:** information is **scrambled** across the room and never returns.
- **Question:** how many hidden qubits does a model need to reproduce *and forecast* each case?

---

## Physics setup

The bell (qubit 0) couples to a mixed-field Ising chain (qubits 1–5):

$$H = g\,Z_0Z_1 + J\sum_{i=1}^{4} Z_iZ_{i+1} + \sum_{i=0}^{5}\big(h_xX_i + h_zZ_i\big), \qquad g=0.5,\ J=1,\ h_x=0.9045$$

The chaos knob is $h_z$, and we verify chaos with level-spacing statistics $\langle r\rangle$:

| Room | $h_z$ | $\langle r\rangle$ measured | Theory |
|---|---|---|---|
| Simple | 0 | **0.309** | integrable ≲ 0.386 |
| Chaotic | 0.809 | **0.527** | GOE = 0.531 |

## The model: a latent quantum neural ODE

$$\frac{d}{dt}|\Psi\rangle = -iH_\theta|\Psi\rangle,\qquad H_\theta=\sum_j\theta_jP_j,\qquad r_a(t)=\langle\Psi(t)|\,\sigma_a^{(\text{bell})}\,|\Psi(t)\rangle$$

- The building blocks $P_j$ are **all** 1-qubit Paulis plus all nearest-neighbour 2-qubit Paulis, giving $12k+3$ parameters.
- The helpers are **never measured**: they are latent quantum memory, analogous to augmented and latent neural ODEs.
- **Leaky variant:** a Lindblad amplitude-damping term on each helper with trainable rate $\gamma_q$, which adds only $k$ parameters.
- **Exact gradients** come from the Daleckii–Krein formula and are verified against finite differences ($3.8\times10^{-9}$ agreement).

---

## Results

### 1 · Scaling: chaos needs more memory

| Room | Hidden qubits $k$ | Train MSE | Test MSE | Forecast beyond training |
|---|---|---|---|---|
| Simple | 3 | 0.0025 | 0.020 | **+5.8** time units |
| Chaotic | 3 | 0.0030 | 0.046 | **+0.7** time units |

Both rooms are fitted equally well, but the chaotic forecast breaks right after training ends.

<p align="center"><img src="figures/07_k3_fits_to_t30.png" width="85%"></p>

### 2 · Quantum vs classical, same budget (~27 parameters)

| Model | Simple room: test MSE / valid time | Chaotic room: test MSE / valid time |
|---|---|---|
| Quantum, unitary | 0.033 / 1.9 | 0.106 / 2.3 |
| **Quantum, leaky helpers** | **0.023 / 10.7** | **0.011 / 27.6** |
| Classical linear state-space | 0.104 / 5.6 | 0.015 / 16.4 |
| Classical augmented neural ODE | 0.296 / 0.9 | 0.062 / 2.3 |

Valid time is the first time the smoothed RMS error exceeds 0.15. All models used the same data, loss, optimiser (L-BFGS-B, 6 restarts) and initial condition.

<p align="center"><img src="figures/08_classical_comparison.png" width="85%"></p>

**Why:** coherent quantum memory carries the echoes, and the leak removes the false ones. Classical models can forget but cannot hold the phase relations behind echoes. The largest linear model even learned an **unstable mode** and exploded (test MSE 73).

### 3 · Hardware: emulated IBM Torino (133-qubit Heron)

| Room | Raw | Readout mitigation | + Zero-noise extrapolation |
|---|---|---|---|
| Simple | 0.082 | 0.079 | **0.047** |
| Chaotic | 0.067 | 0.065 | **0.041** |

Values are RMS error against the ideal model, with 8000 shots (shot-noise floor ≈ 0.011).

- **Exact 3-qubit compilation** keeps depth constant at about 29 CZ for any $t$ (Trotter steps would need thousands).
- **Calibration-aware qubit selection** uses qubits 51-52-37 (0.5% readout error) instead of qubit 0 (16.7%).
- **ZNE** uses unitary folding at scales 1, 3, 5 with Richardson extrapolation.

<p align="center"><img src="figures/10_hardware_emulation.png" width="85%"></p>

---

## Notebook map

| Part | Sections | What happens |
|---|---|---|
| **I. Real room** | 1–6 | Hamiltonian, exact and Trotter simulation, chaos test, dataset |
| **II. Fake room** | 7–9 | Latent quantum NODE, exact gradient, circuit check, training, scaling sweep, forecast horizon |
| **III. Quantum vs classical** | 10 | Linear SSM, augmented NODE, leaky-helper quantum model, stability check |
| **IV. Hardware** | 11 | Qubit selection, noisy emulation, REM + ZNE, real-device cell |
| **Summary** | 12 | Results, limitations, future work, 29 references |

## Repository structure

```
qnode-chaos/
├── qnode_chaos.ipynb     # full project, pre-run outputs, explanations + references
├── figures/              # key plots exported from the notebook
├── requirements.txt      # pinned dependencies
├── LICENSE               # MIT
└── README.md
```

**Stack:** `qiskit 2.5.2` · `qiskit-aer 0.17.2` · `qiskit-ibm-runtime 0.50.0` · `jax 0.11.2` · `numpy` · `scipy` · `matplotlib`

**Seeds:** restarts use seed 0; the transpiler and simulator use seed 7.

---

## Limitations & future work

- **Scope tested:** one room size (5 qubits), one training window, one restart seed. The largest models are unitary $k=3$ and leaky $k=2$.
- **Hardware:** results are a **noisy emulation**; the real-device run is pending.
- **Next steps:**
  - Leaky helpers on hardware via mid-circuit reset (collision models).
  - Larger $k$ and larger rooms.
  - Multiple seeds with error bars.
  - Link the forecast horizon to the scrambling rate (OTOCs).

## Key references

1. R. T. Q. Chen et al., *Neural Ordinary Differential Equations*, NeurIPS (2018).
2. E. Dupont, A. Doucet, Y. W. Teh, *Augmented Neural ODEs*, NeurIPS (2019).
3. D. Tamascelli et al., *Nonperturbative treatment of non-Markovian dynamics of open quantum systems*, PRL **120**, 030402 (2018).
4. M. C. Bañuls, J. I. Cirac, M. B. Hastings, *Strong and weak thermalization of infinite nonintegrable quantum systems*, PRL **106**, 050405 (2011).
5. Y. Y. Atas et al., *Distribution of the ratio of consecutive level spacings in random matrix ensembles*, PRL **110**, 084101 (2013).
6. J. Pathak et al., *Model-free prediction of large spatiotemporally chaotic systems from data*, PRL **120**, 024102 (2018).
7. K. Temme, S. Bravyi, J. M. Gambetta, *Error mitigation for short-depth quantum circuits*, PRL **119**, 180509 (2017).
8. T. Giurgica-Tiron et al., *Digital zero noise extrapolation for quantum error mitigation*, IEEE QCE (2020).
9. A. Javadi-Abhari et al., *Quantum computing with Qiskit*, arXiv:2405.08810 (2024).

The full list of 29 references is at the end of the notebook.

## License

[MIT](LICENSE) © 2026 Keval Mandalia
