# Fast-or-never: the innovation CUSUM after a latent level shift

Formal statement and proof of the mechanism found empirically in exp01
("the innovation CUSUM is fast or never on level breaks") and used
throughout FINDINGS.md. Numerically verified in
`experiments/exp06_theory_check.py` (outputs `paper_assets/exp06_*`);
helper functions in `lsc/theory.py`.

## Setup

State-space model with known parameters, in steady state:

    S_t = φ S_{t−1} + w_t,   w_t ~ N(0, q),   |φ| < 1
    Y_t = S_t + v_t,         v_t ~ N(0, r)

Steady-state Kalman filter: prediction variance P solves the Riccati
fixed point P = φ²Pr/(P+r) + q, gain K = P/(P+r), innovation variance
F = P + r. Standardized innovations e_t = (Y_t − φŜ_{t−1})/√F are iid
N(0,1) under the null.

A **level break** adds δ to the state path from t₀ on: S̃_t = S_t +
δ·1{t ≥ t₀} (the DGP used in all experiments), hence Ỹ_t = Y_t +
δ·1{t ≥ t₀}.

The **one-sided Page CUSUM** with drift allowance k and threshold h is
g_t = max(0, g_{t−1} + e_t − k), alarm when g_t ≥ h.

## Proposition 1 (innovation mean path)

The filter is a linear time-invariant map of Y in steady state, so the
broken path's standardized innovations are ẽ_t = e_t + μ_t with e_t the
null innovations and μ_t deterministic: for j = t − t₀ ≥ 0,

    μ_{t₀+j} = (δ − φ a_{j−1}) / √F,
    a_j = ρ a_{j−1} + K δ,    a_{−1} = 0,    ρ = φ(1 − K) ∈ (0, 1),

where a_j is the filter's mean response E[Ŝ_{t₀+j}] − E[Ŝ⁰_{t₀+j}].
Explicitly, μ decays geometrically at rate ρ from μ_{t₀} = δ/√F to

    μ_∞ = δ (1 − φ) / ((1 − φ(1 − K)) √F).

*Proof.* Linearity gives the decomposition with μ_t the innovation
response to the deterministic input δ·1{t≥t₀}. The response of the
filter mean: â_j ≡ mean state-estimate response satisfies â_j =
φâ_{j−1} + K(δ − φâ_{j−1}) (predict, then correct by K times the mean
innovation), i.e. â_j = ρâ_{j−1} + Kδ; the mean innovation response is
the input minus the prediction response, (δ − φâ_{j−1})/√F. Solving the
linear recursion: â_j = (Kδ/(1−ρ))(1 − ρ^{j+1}), and the limit of
(δ − φâ)/√F is δ(1 − φK/(1−ρ))/√F = δ(1−φ)/((1−ρ)√F). ∎

Interpretation: the filter adapts, so of the full shift δ only the
fraction (1−φ)/(1−φ(1−K)) survives in the innovations per step —
"innovations carry ≈ (1−φ)δ per step" as diagnosed in exp01 (the exact
factor includes the gain correction). The transient carries total
excess mass Σ_j (μ_{t₀+j} − μ_∞) = φ a_∞ /((1−ρ)√F), which is what a
"fast" detection consumes.

## Proposition 2 (never-detect bound)

Suppose μ_t ≤ μ̃ < k for all t ≥ t₁ ≥ t₀ (post-transient), and let
g_{t₁} = g < h. Then for any horizon L,

    P( max_{t₁ < t ≤ t₁+L} g_t ≥ h  |  g_{t₁} = g )
        ≤ (L + 1) · exp( −2 (k − μ̃)(h − g) ).

In particular, if the alarm did not fire during the transient, it fires
later with probability exponentially small in the threshold.

*Proof.* For n > t₁, g_n = max( g + Σ_{i=t₁+1}^n X_i ,
max_{t₁<m≤n} Σ_{i=m}^n X_i ) with increments X_i = e_i + μ_i − k =
z_i − (k − μ_i), z_i iid N(0,1). An alarm by t₁+L requires some
anchored sum Σ_{i=m}^n X_i ≥ h − g for some m ∈ (t₁, t₁+L] (or the
g-anchored sum ≥ h − g). Each X_i is stochastically dominated by
z_i − (k − μ̃), for which θ* = 2(k − μ̃) solves E[e^{θX}] = 1; the
exponential martingale e^{θ*Σ} with the maximal inequality gives
P(sup_n Σ_{i=m}^n X_i ≥ h − g) ≤ e^{−θ*(h−g)} for each of the ≤ L+1
anchor points; a union bound finishes. ∎

## Proposition 3 (raw CUSUM: slow but sure — Wald approximation)

The raw-Y CUSUM standardizes Y by its training moments; after the
break the standardized mean shift Δ = δ/σ_Y (σ_Y² = q/(1−φ²) + r)
persists forever. If Δ > k the post-break increments have positive
drift Δ − k, so the alarm is certain as the horizon grows, with
first-passage (Wald) delay

    E[D] ≈ h / (Δ − k),

and if Δ < k the same bound as Proposition 2 applies to raw CUSUM
(both detectors are "never" for small enough shifts). This is an
approximation (it ignores boundary overshoot and reflection at 0),
not a bound.

## Corollary (the fast-or-never / slow-but-sure dichotomy)

At matched FAR (thresholds h_lsc, h_raw calibrated on the same nulls):

- If μ_∞ < k < Δ: the innovation CUSUM detects only via its transient
  (length O(1/(1−ρ)) obs) — fast when it fires, never otherwise —
  while the raw CUSUM detects with probability → 1 at Wald delay
  h_raw/(Δ−k). This is the regime of every level scenario in the
  grids, and is why raw CUSUM wins detect rate while the innovation
  CUSUM wins delay-conditional-on-detection.
- If also Δ < k (tiny shifts): both are never-detectors; detect rates
  sit at FAR.
- μ_∞ > k requires δ(1−φ)/((1−φ(1−K))√F) > k — with φ = 0.95 and
  k = 0.5 this needs δ of the order of 10σ_ref: within any reasonable
  break magnitude the innovation CUSUM is *structurally* in the
  fast-or-never regime. Lowering k helps only until the null ARL
  (hence h) inflates — the calibration harness prices that in
  automatically.

## Numerical verification (exp06, 1000 reps)

- **Mean path (A):** MC average of standardized innovations after a
  3σ_ref shift matches μ_t everywhere; max pointwise deviation 0.079
  with per-point MC SE 0.032 over 250 points (`exp06_innovation_path.png`).
- **Reduction and bound (B):** detection-probability curves of the full
  filter MC and the reduced simulation (iid N(0,1) + μ_t) agree within
  MC error at δ = 1σ and 3σ; the Proposition 2 bound is never violated
  (`exp06_detect_vs_h.png`).
- **Against the actual grid_v1 numbers (C,** `exp06_theory_table.csv`**):**
  at the arenas' calibrated thresholds, μ_∞ vs k = 0.5 cleanly sorts
  the observed behavior: δ ≤ 1σ gives bound ≤ 0.7% — observed
  innovation-CUSUM detect rates 0.04–0.13 ≈ FAR (pure transient +
  false alarms); δ = 3σ is a knife-edge (μ_∞ = 0.43–0.48, gap
  0.02–0.07) — observed partial detect 0.55–0.67, exactly the
  fast-or-never signature. Raw CUSUM: Wald delays 68/84/110 vs
  observed medians 58/75/91 across SNRs at 3σ (approximation ~15–20%
  conservative), and at 1σ its drift 0.577 barely exceeds k, Wald
  delay 1334 ≫ the 250-obs horizon — matching its observed partial
  detect 0.30 without any fitting.

## Scope and caveats

The theory is for the steady-state filter with known parameters and a
one-sided CUSUM; the experiments use fitted (training-prefix)
parameters, diffuse initialization, and a two-sided CUSUM. exp06 shows
the known-parameter theory nevertheless predicts the fitted-parameter
experiments' behavior to first order. Parameter-estimation noise adds
a second-order inflation of the null CUSUM (it is priced into h by
calibration). The persistence-break analogues (conditional level
freeze; quieting suppression) are analyzed mechanistically in
FINDINGS.md but not formalized here.
