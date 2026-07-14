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

## ARMA(1,1) equivalence of the whitening rungs (M1)

The two whitened rungs of the ladder — the ARIMA benchmark and the
Kalman filter — are the **same filter** on this DGP, not two competing
estimators. The observable Y has an exact ARMA(1,1) reduced form.
Applying the AR operator to Y,

    (1 − φL) Y_t = (1 − φL) S_t + (1 − φL) v_t = w_t + v_t − φ v_{t−1} =: u_t,

and u_t is an MA(1): its autocovariances are

    γ_u(0) = q + r(1 + φ²),   γ_u(1) = −φ r,   γ_u(h) = 0 (h ≥ 2).

Matching u_t to (1 − θL) ε_t with ε_t white (variance σ_ε²) gives
(1+θ²)σ_ε² = γ_u(0) and −θσ_ε² = γ_u(1); the invertible root is

    θ = (m − √(m²−4)) / 2,   m = (q + r(1+φ²)) / (φ r),   σ_ε² = φ r / θ.

Equivalently, the marginal autocovariances of Y are γ_Y(0) = q/(1−φ²)
+ r, γ_Y(1) = φ q/(1−φ²), and γ_Y(h) = φ^{h−1} γ_Y(1) for h ≥ 1 — the
AR(1)-decaying tail that fixes the same (θ, σ_ε²).

**Two identities tie the reduced form to Proposition 1 and the Riccati
fixed point** (both verified to machine precision, `test_theory
.test_arma11_riccati_identities`; helper `lsc.theory
.arma11_representation`):

    σ_ε² = F          (the ARMA innovation variance IS the Riccati F)
    θ    = ρ = φ(1−K) (the MA parameter IS the innovation-mean decay rate).

The second identity is immediate from the innovation recursion: writing
ν_t = Y_t − φŜ_{t−1} for the (unstandardized) Kalman innovation and
Ŝ_{t−1} = φŜ_{t−2} + Kν_{t−1}, substitution gives ν_t = Y_t − φY_{t−1}
+ φ(1−K)ν_{t−1} — a recursion in Y and its own lag with the state
eliminated, identical to the ARMA(1,1) innovation ε_t obeying
ε_t = Y_t − φY_{t−1} + θ ε_{t−1} once θ = φ(1−K). Since also σ_ε² = F, the standardized series coincide: **at
steady state with correct parameters the Kalman one-step innovations and
the ARMA(1,1) innovations are the same linear innovations of the same
Gaussian process.** The state-space layer buys nothing over ARMA
whitening for second-moment monitoring on this DGP.

**Numerical confirmation (exp07, ≥200 null paths, T=500).** With TRUE
parameters the two series agree to machine precision (median Pearson
ρ = 1.000000, max|Δ| ≈ 10⁻⁹ across SNR ∈ {0.1, 0.5, 2.0} — the
statsmodels ARMA filter vs. the hand-written steady-state Kalman
recursion, entirely independent code paths). With ESTIMATED parameters
(each rung fit on the training prefix, the ladder's real operating
condition) ρ̄ = 0.991 (≥ 0.95): the small wedge is pure estimation
error, and forcing the ARIMA order to the true (1,0,1) tightens it to
ρ̄ = 0.9995. **Decision A1 (pre-registered) fires: the rungs are
equivalent; §5 collapses to a two-rung ladder (raw vs. whitened).**

*Order-selection caveat (the M1 finding).* AIC over the benchmark grid
almost never selects the theoretically-exact (1,0,1): at φ = 0.95 it
prefers (1,0,0) at SNR 0.1 (37% of paths) and the differencing order
(0,1,1) at SNR 0.5 / 2.0 (64% / 71%). This is a near-unit-root artifact,
not a bug — at φ = 0.95 an IMA(1,1) and an AR(1) both approximate the
ARMA(1,1) closely enough to preserve ρ̄ ≥ 0.95, and the AIC-selected
rung is a legitimate whitener. The scope note is that the equivalence is
strongest near a unit root; at small φ (the M3 sweep) the reduced form
is genuinely ARMA(1,1) and mis-differencing would degrade it, so the
ladder's ARIMA rung is reported as-built (AIC-selected), with the
forced-(1,0,1) result given alongside.

## Corollary (Proposition 1 in reduced form — filter-agnostic fast-or-never)

Because the ARMA(1,1) innovations *are* the steady-state Kalman
innovations (previous section), Proposition 1 is not a statement about
the Kalman filter specifically — it is a statement about the *whitening
filter* of the observable, however implemented. Restated in reduced-form
notation: let ε_t be the innovations of the ARMA(1,1) representation
(1 − φL)Y_t = (1 − θL)ε_t, with MA parameter θ and innovation variance
σ_ε². After a level shift δ in the state at t₀ (equivalently, a step of
size δ(1 − φ) in the mean of the differenced series, since the shift
enters Y as δ·1{t ≥ t₀} and (1 − φL) maps it to δ(1 − φ) for t > t₀), the
standardized innovation mean follows

    μ_{t₀+j} = (δ − φ a_{j−1}) / √σ_ε²,   a_j = θ a_{j−1} + K δ,   a_{−1}=0,

i.e. *the identical geometric transient, with the decay rate equal to the
MA parameter θ* (using θ = ρ = φ(1 − K) and σ_ε² = F), converging to the
same

    μ_∞ = δ(1 − φ) / ((1 − θ)√σ_ε²).

The fast-or-never bound of Proposition 2 then applies verbatim to ε_t.
Two consequences. (1) The negative first-moment result transfers to *any*
correctly-specified ARMA(1,1) whitening — an ARIMA(1,0,1) residual CUSUM
inherits the same fast-or-never behavior as the innovation CUSUM, because
they whiten to the same series; there is nothing Kalman-specific about it.
(2) The decay rate is *observable* from the reduced form — it is the MA
root θ, which an analyst can read off a fitted ARMA(1,1) without ever
writing down a state space. Since θ = φ(1 − K) → 1 as φ → 1 (K bounded),
the transient lengthens and μ_∞ → 0 near the unit root: the fast-or-never
trap is a large-θ (persistent) phenomenon, exactly what the φ sweep (M3)
confirms.

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
