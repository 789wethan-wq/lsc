# When Does Filtering Help You See a Break? Latent-State Diagnostics for Structural Change at Calibrated False-Alarm Rates

## Abstract

We ask when filtering a latent state helps detect structural change,
using a protocol that calibrates every detector to the same
false-alarm rate (5% per 500 observations) on matched null data. The
answer is a trichotomy — *no, yes, no* — across three break types.
(i) Level shifts in a persistent state: no — a raw-data CUSUM
dominates at every SNR, and the innovation CUSUM is provably "fast or
never," a boundary condition of persistence (μ∞ sorts detection,
Spearman 0.94). (ii) Observation-noise variance changes: yes, but the
advantage is prewhitening, not the state estimate — the observable is
exactly ARMA(1,1), so ARIMA and Kalman whitening are provably the same
filter. (iii) State-innovation (shock) variance changes — the
Great-Moderation/crisis-volatility channel: no, a partial null. Raw
matches or beats whitening on the coarse ×3 break (0.72/0.96/0.96 vs
0.26/0.79/1.00), but the subtle ×1.5 break leaves every rung near the
false-alarm floor — whitening *fails to recover* it. On real data
(industrial production, GDP, Treasury yields, unemployment) every
alarm attributes to a second-moment feature; the headline NBER
association reaches an uncorrected permutation p = 0.008, but this
does not survive a family-wise or FDR correction across the full grid
of tests the real-data section actually runs (§9), and real-time
vintages confirm COVID while downgrading 2008. Two extensions probe
the protocol's edges: offline PELT matches raw CUSUM on level breaks
but not variance breaks, and a bounded-memory statistic fixes raw
CUSUM's blindness to a second level break, though not a second
variance break. Read together, the results are deflationary for the
latent layer's detection power: a raw or ARIMA-whitened benchmark
matches or beats the state-aware detector on every break type studied,
and what filtering buys instead is breadth and attribution, not power
(§10).

**JEL classification:** C12, C22, C52.
**Keywords:** structural change; sequential change detection; CUSUM;
state-space models; Kalman filter; prewhitening; false-alarm rate;
latent-state diagnostics; volatility regimes.

---

## 1. Introduction

Structural change is usually sought in observables. But many economic
quantities of interest — trend output, a natural rate, an underlying
volatility regime — are latent states observed only with noise, and the
natural first step is to filter: estimate the state, then watch the
estimate for breaks. Intuition says the filtering should help, by
stripping observation noise before the change is sought. This paper asks
when that intuition holds, under a protocol that makes the question
answerable. Our contributions are as follows.

- Contribution 1 (protocol): a calibrated-FAR parity harness. Every
  method's threshold comes from the same routine on the same matched-null
  draws; empirical FAR is re-verified on fresh nulls; causality is
  enforced structurally (parameters fit on a training prefix only;
  forward-only filtering; alarm scores are NaN on training data) and
  tested bit-identically (perturbing future observations leaves all
  scores at earlier times unchanged, exactly). The protocol extends to
  the honesty of the research process itself: three pre-registered
  hypotheses were falsified, and every post-hoc change is logged
  (CHANGELOG) and the failures reported as findings rather than
  suppressed.
- Contribution 2 (negative + positive results): the intuition is wrong
  for first moments at high persistence — and we can prove why
  (fast-or-never theorem), then bound the claim by sweeping φ (μ∞ sorts
  detection, and the innovation CUSUM escapes the regime at low φ). For
  second moments the answer is a two-rung, two-channel decomposition. The
  ladder collapses to *two* rungs because the ARIMA and Kalman rungs are
  the same filter — the observable is exactly ARMA(1,1), an equivalence
  we state as theory and confirm to machine precision — so "prewhitening"
  and "state estimation" are not separable here; the state layer's
  channel-specific advantage on this DGP reduces to prewhitening, which
  ARIMA residuals also supply. And the prewhitening advantage is
  *channel-specific*: it holds for observation-noise breaks (decisive at
  high SNR, where the latent signal masks the noise change and raw sits at
  chance) but *inverts* for state-innovation breaks, where the raw
  variance CUSUM matches or beats the whitened rungs — the channel that
  matters for the volatility-regime events of the application.
- Contribution 3 (method): a tail-robust exceedance-indicator CUSUM that
  preserves the second-moment advantage under heavy-tailed noise, reached
  by rejecting two documented alternatives — a falsified clipped variant
  and an in-composite variant diluted below its standalone form (§8.3).
- Contribution 4 (application discipline): attribution, permutation
  tests, sensitivity, pinned data snapshots, and real-time vintages for
  the real-data claims — including a self-correction on the headline GFC
  timing.

**Related work.** The paper sits at the intersection of several
literatures.

**Quickest detection and SPC.** Sequential change detection descends
from Page's (1954) CUSUM and the quickest-detection tradition (Lorden
1971; Moustakides 1986); we use the CUSUM as the common statistic across
information sets rather than proposing a new stopping rule. That
literature also supplies the evaluation currency we adopt: the average
run length (ARL), with methods compared at a matched in-control ARL₀ —
the statistical-process-control convention (Page 1954; Montgomery 2013).
The broader statistical-surveillance research program this evaluation
convention sits inside is surveyed in Frisén (2003), a complement to
the Basseville & Nikiforov (1993) treatment cited below for the
innovation-CUSUM axis specifically.

**Innovation-based state-space monitoring.** The idea of monitoring a
*state-space* model through its innovations is not ours: it is the core
of the innovation-based change-detection literature — the generalized
likelihood ratio test on Kalman innovations (Willsky & Jones 1976) and
the systematic treatment of CUSUM/GLR schemes on innovation sequences in
Basseville & Nikiforov (1993). Our contribution on this axis is *not*
the innovation CUSUM or ARL-matching per se, both of which are standard
in SPC and quickest detection; it is that this matched-error protocol is
essentially *absent* from the applied latent-state econometrics
literature, and that transplanting it there — calibrating every
information set to a common false-alarm rate — is precisely what
produces the negative results (raw data wins on levels; the state
estimate adds nothing over ARIMA whitening on variances).

**Econometric CUSUM-of-residuals.** In econometrics, sequential
monitoring of structural change is the CUSUM-of-recursive-residuals line
and its modern moving-window ("MOSUM") monitoring form (Brown, Durbin &
Evans 1975; Chu, Stinchcombe & White 1996), which watch *observable*
regression residuals; our object is a *latent* state, and the residuals
we monitor are a filter's innovations. The bounded-memory fix we give
the multi-break re-arm failure in §7 is in this same MOSUM family: a
moving two-window mean-shift statistic, rather than a CUSUM accumulated
against a fixed training-prefix baseline.

**Regime-switching.** Regime-switching models (Hamilton 1989; Kim &
Nelson 1999) offer a state-aware alternative whose regime probabilities
we include as a benchmark and find saturate under calibration on
nonstationary data.

**Great Moderation and volatility empirics.** The empirical target of
the second-moment results is the Great Moderation volatility decline
(McConnell & Pérez-Quirós 2000; Stock & Watson 2002); as we show, that
and the crisis-volatility events are *state-innovation* (shock-variance)
breaks, the channel on which raw and prewhitened detectors are
interchangeable.

**Parametric volatility models.** A separate literature models
conditional variance parametrically — GARCH (Bollerslev 1986) and
stochastic-volatility state-space models (Kim, Shephard & Chib 1998);
its object is *estimation* of a volatility process, not distribution-free
monitoring at a calibrated false-alarm rate, so a full stochastic-volatility
comparison remains future work. A GARCH(1,1) benchmark is reportable
now, however: fit on the training prefix only (via the `arch` package),
causally forward-filtered for conditional variance over the full series
with the fixed fitted parameters, and run through the same three-arm
max-CUSUM used for the raw and ARIMA rungs on its standardized residuals
(`experiments/garch_detector.py`, `experiments/exp15_garch_benchmark.py`).
Calibrated at n_reps = 500 matching the published grid (empirical FAR
0.050 in every cell checked), GARCH sits at the false-alarm floor on
both variance channels at every SNR tested — r-channel ×1.5: 0.098
(SNR 0.5), 0.096 (SNR 2.0); q-channel ×1.5: 0.066 (SNR 0.5), 0.098
(SNR 2.0) — against raw's 0.56/0.10/0.21/0.23 and ARIMA's
0.94/0.87/0.10/0.16 on the same four cells (§5, Table 3). This is not
merely "GARCH is dominated": on this DGP it contributes nothing over
chance. The likely mechanism is a generative mismatch — GARCH(1,1) is
built for conditional heteroskedasticity (volatility clustering driven
by squared past shocks), a different assumption than this paper's DGP
(a permanent step change in noise variance layered on a highly
persistent φ = 0.95 latent state) — but we have not isolated the
mechanism beyond this scope note, and do not rule out an
implementation-specific cause. A break-aware GARCH variant (allowing
its own parameters to shift) and the full stochastic-volatility
comparison remain open.

**Offline changepoint detection.** Offline changepoint methods (PELT,
Killick et al. 2012) solve a retrospective segmentation problem; our
monitoring is strictly causal and calibrated to a false-alarm rate, so
the two are not comparable on delay (§8.5). We nonetheless calibrate
PELT to the same false-alarm rate on null paths and ask whether it
*localizes* a break at all, given the full sample: it is competitive
with the raw-Y CUSUM on an obvious 3σ level shift (localization rate
0.83–0.92 across SNRs vs. the causal detector's 0.97–0.99) but far
weaker on pure variance breaks (0.00–0.02 vs. the dedicated raw
variance-CUSUM's 0.10–1.00 across the same grid), because PELT's default
cost model is a mean-shift detector — a concrete reason a dedicated
variance statistic, not an off-the-shelf offline method, is the right
tool for the channel this paper is about. The canonical retrospective
alternative for multiple mean breaks specifically, rather than PELT's
general segmentation cost, is the dynamic-programming estimator of Bai
& Perron (2003); we do not benchmark against it directly since our
question is causal monitoring at a calibrated false-alarm rate, not
retrospective break-date estimation, but it is the natural reference
point for §7's multi-break discussion.

What is new here is not any single detector but the calibrated-parity
harness that makes latent-state and raw-data detectors directly
comparable, and a reduced-form result (the exact ARMA(1,1) equivalence
of the ARIMA and Kalman rungs) that dissolves the "does filtering help?"
question for second moments into a prewhitening question with a known
answer.

## 2. Framework and evaluation protocol

**Model layer.** S_t = φS_{t−1} + w_t (var q), Y_t = S_t + v_t (var r);
the estimator sees only Y and fits (φ, q, r) by maximum likelihood on a
training prefix of 125 observations (25% of the T = 500 baseline sample;
the same 25% fraction at other T), then runs a forward-only filter with
frozen parameters. Standardized one-step innovations e_t and the filtered
state are the raw material for diagnostics.

**Diagnostics layer.** Eleven features of the filtered path — level
change, slope, acceleration, instability, rolling persistence, a Page
CUSUM of innovations (allowance k = 0.5), variance CUSUMs of e²−1 (k =
0.25 and 0.05), a quietness CUSUM of 1−e² (k = 0.05), rolling innovation
autocorrelation, and a CUSUM of the filtered state against its training
baseline. Each feature is standardized at every time point by the median
and IQR of its null distribution at that same time point (pooled-over-time
standardization is a design flaw that blunted the composite; §8.4). The
composite score is the max of standardized features; the composite is
itself calibrated on nulls, which prices in the multiple-feature testing
automatically.

**Calibration parity.** Threshold = (1−FAR) quantile of the
per-replication maximum score over matched-null draws; identical routine,
budgets, and seed layout for LSC detectors and benchmarks (raw-Y CUSUM,
ARIMA+CUSUM, plain-HMM regime flips). Calibration, evaluation, FAR-check,
and feature-scale seeds are disjoint by construction.

**Metrics.** Detection rate, pre-break false alarms, censored delay; for
multi-break: event-level precision/recall/F1 with one-to-one greedy
matching in a 100-observation window. To connect the protocol to the
statistical-process-control and quickest-detection literatures we also
report the two average-run-length quantities those fields are built
around (`paper_assets/arl_table.csv`). The in-control **ARL₀** is the
mean observations between false alarms implied by the calibrated
per-observation false-alarm hazard: with monitored window length L = 375
(T = 500, 25% training) and empirical window-FAR α, the hazard is
p = 1 − (1−α)^{1/L} and ARL₀ = 1/p ≈ L/α; our 5% window target
corresponds to ARL₀ ≈ 7300 observations, and the empirical detectors sit
at 5900–7600 (raw CUSUM calibrates hot, increasingly so at higher SNR:
4.0% → 6.2% → 8.2% vs. the 5% target). The out-of-control
**ARL₁** is the post-break detection delay (mean delay conditional on
detection, reported beside the detection rate since misses are censored):
e.g. at 3σ level, SNR 0.5, ARL₁ ≈ 77–86 observations. Calibrating a
common ARL₀ is exactly the ARL-matching convention of SPC (Basseville &
Nikiforov 1993); we express it as a window-FAR because our horizon is
finite.

| Arena | Method | Empirical FAR | ARL₀ (obs) |
|---|---|---|---|
| SNR 0.1 | lsc_composite | 5.2% | 7023 |
| SNR 0.1 | lsc_kalman_cusum | 3.4% | 10841 |
| SNR 0.1 | lsc_state_cusum | 5.0% | 7311 |
| SNR 0.1 | raw_cusum | 4.0% | 9187 |
| SNR 0.5 | lsc_composite | 4.8% | 7624 |
| SNR 0.5 | lsc_kalman_cusum | 4.8% | 7624 |
| SNR 0.5 | lsc_state_cusum | 5.4% | 6756 |
| SNR 0.5 | raw_cusum | 6.2% | 5859 |
| SNR 2.0 | lsc_composite | 6.2% | 5859 |
| SNR 2.0 | lsc_kalman_cusum | 4.8% | 7624 |
| SNR 2.0 | lsc_state_cusum | 7.8% | 4618 |
| SNR 2.0 | raw_cusum | 8.2% | 4383 |

*Table 1. ARL₀ by arena and method (grid_v1 core, T = 500; target
ARL₀ ≈ 7300 at the 5% window-FAR). MC SEs on empirical FAR ≤ 0.013
(n_reps = 500, √(p(1−p)/n_reps)).*

| Arena | Scenario | Method | Detect rate | ARL₁ (mean delay) |
|---|---|---|---|---|
| SNR 0.1 | level 3σ | lsc_composite | 0.766 | 112.6 |
| SNR 0.1 | level 3σ | lsc_kalman_cusum | 0.654 | 63.9 |
| SNR 0.1 | level 3σ | raw_cusum | 0.966 | 72.3 |
| SNR 0.1 | variance ×3 | lsc_composite | 0.990 | 29.3 |
| SNR 0.1 | variance ×3 | lsc_kalman_cusum | 0.870 | 80.4 |
| SNR 0.1 | variance ×3 | raw_cusum | 0.728 | 104.0 |
| SNR 0.5 | level 3σ | lsc_composite | 0.530 | 86.4 |
| SNR 0.5 | level 3σ | lsc_kalman_cusum | 0.554 | 77.2 |
| SNR 0.5 | level 3σ | raw_cusum | 0.990 | 81.8 |
| SNR 0.5 | variance ×3 | lsc_composite | 0.992 | 25.2 |
| SNR 0.5 | variance ×3 | lsc_kalman_cusum | 0.224 | 107.2 |
| SNR 0.5 | variance ×3 | raw_cusum | 0.076 | 137.6 |
| SNR 2.0 | level 3σ | lsc_composite | 0.670 | 63.2 |
| SNR 2.0 | level 3σ | lsc_kalman_cusum | 0.674 | 49.0 |
| SNR 2.0 | level 3σ | raw_cusum | 0.988 | 96.6 |
| SNR 2.0 | variance ×3 | lsc_composite | 0.976 | 17.2 |
| SNR 2.0 | variance ×3 | lsc_kalman_cusum | 0.268 | 114.6 |
| SNR 2.0 | variance ×3 | raw_cusum | 0.058 | 172.9 |

*Table 2. ARL₁ (detection rate and mean delay conditional on detection)
at the canonical level-3σ and variance-×3 breaks, T = 500. MC SEs on
detect rate ≤ 0.023 (n_reps = 500).*

## 3. Simulation design

Arenas: AR(1) latent state (φ = 0.95) at spec-SNR (stationary state
variance / observation variance) ∈ {0.1, 0.5, 2.0}, with a persistence
sweep φ ∈ {0.5, 0.8, 0.95, 0.99} (SNR held fixed by q = SNR·(1−φ²)) used
to map the fast-or-never boundary (§4); and a local-level (random-walk
state) arena, analyzed in §8.4, where — as we now demonstrate rather than
assert — level detection is degenerate for every method and variance
detection becomes possible only after whitening. Breaks at mid-sample
come in **two variance channels** whose distinction is central to §5.
An *observation-noise* (r) break scales the white-noise component's
standard deviation (×1.5, ×3, ×⅔ quieting); a *state-innovation* (q)
break scales the state shock's standard deviation by the same factors —
the identical SD convention, so "×1.5" means SD×1.5 in both channels.
The two are structurally different: an r-break changes only the marginal
variance of Y, whereas a q-break changes both the marginal variance and
the autocorrelation structure (it shifts the reduced-form ARMA(1,1) MA
parameter). Also: level shifts (0.5, 1, 3 σ_ref, where σ_ref = √(q/(1−φ²)) is the
latent state's own stationary standard deviation), logistic ramps, and
pure persistence changes (φ → 0.995 or 0.80 with the stationary variance
held fixed). T = 500 baseline with a T ∈ {200, 2000} sweep;
misspecification arenas with t₅ observation noise and a nonlinear tanh
state drift.

## 4. First moments: raw data wins, and we can say exactly why

**Empirics.** At matched FAR, raw-Y CUSUM has the best level-shift
detection rate at every SNR (0.97–0.99 at 3σ) and every T; the latent
state CUSUM approaches but never overtakes it (0.19 → 0.94 as SNR rises).
Two pre-registered "latent advantage" hypotheses were falsified. The
latent-innovation CUSUM is the speed champion conditional on firing:
median delay 24–53 observations at 3σ versus raw's 58–91, at the cost of
detection rate (0.55–0.67) — "fast or never."

**Theory.** The empirics above are not an artifact of tuning; they follow
from two facts about the detectors' post-break drift. Statements and
proofs are in Appendix B (long-form companion:
`experiments/THEORY.md`).

> **Proposition 1 (fast-or-never).** After a state level shift δ, the
> standardized innovation mean decays geometrically at rate ρ = φ(1−K) to
> μ∞ = δ(1−φ)/((1−φ(1−K))√F), where K is the steady-state Kalman gain and
> F the innovation variance. If μ∞ < k, the post-transient innovation
> CUSUM has negative drift, and the probability of an alarm in the next L
> observations is bounded by (L+1)·exp(−2(k−μ∞)h) for threshold h.

The detector therefore fires during the adaptation transient or, with
exponentially small probability, never. With φ = 0.95 and k = 0.5, μ∞ > k
would require shifts of order 10σ: the innovation CUSUM is *structurally*
in the fast-or-never regime, which is why it is the speed champion but
never the detection champion.

**Fast-or-never is a property of the whitening filter, not of Kalman.**
By the ARMA(1,1) equivalence (§5, Appendix B) the innovations here are
the innovations of the reduced-form ARMA(1,1) of Y, so Proposition 1 can
be stated with no state space at all: after the shift, the standardized
ARMA(1,1) innovation mean decays geometrically *at rate equal to the MA
parameter* θ = φ(1 − K) to the same μ∞ = δ(1−φ)/((1−θ)√σ_ε²), and the
bound applies verbatim. Two consequences: an ARIMA(1,0,1) residual CUSUM
inherits the *same* fast-or-never behavior as the Kalman-innovation CUSUM
(they whiten to the same series — nothing is Kalman-specific), and the
decay rate θ is *observable* — an analyst reads it off a fitted ARMA(1,1)
without positing a latent state. Since θ → 1 as φ → 1, the trap is a
persistence phenomenon, which is what the φ sweep tests next.

> **Proposition 2 (raw-CUSUM delay).** The raw-Y CUSUM sees the full shift
> as a sustained standardized drift Δ = δ/σ_Y, giving a Wald
> first-passage mean delay of approximately h/(Δ−k) once Δ > k (Wald
> 1947; Siegmund 1985), and negligible power when Δ ≤ k.

Verification (`experiments/exp06_theory_check.py`, 1000 reps): the Monte
Carlo innovation path matches μ_t within MC error; the Proposition 1 bound
is never violated; at the actual calibrated thresholds, μ∞ sorts every
observed detection rate — δ ≤ 1σ gives bound ≤ 0.7% (observed ≈ FAR);
δ = 3σ is a knife-edge (μ∞ = 0.43–0.48 vs k = 0.5) matching the observed
0.55–0.67. Proposition 2's Wald delays 68/84/110 across SNRs bracket the
observed raw medians 58/75/91 (≈15–20% conservative) and explain the
partial 1σ detection (0.30) without any fitting: Δ = 0.577 barely exceeds
k, so the Wald delay (1334) dwarfs the 250-observation horizon.

**The persistence boundary (φ sweep).** Proposition 1's regime is not
unconditional — μ∞ = δ(1−φ)/((1−φ(1−K))√F) is *increasing in* (1−φ), so
the fast-or-never trap should loosen as the state becomes less
persistent. Sweeping φ ∈ {0.5, 0.8, 0.95, 0.99} at fixed SNR (Figure 1)
confirms it and turns the theory into a
falsifiable ordering: μ∞ sorts the innovation-CUSUM detection rate across
all cells (Spearman 0.94, n=24: 4 φ × 3 SNR × 2 shifts — a stratified
permutation null that shuffles φ-pairing within each SNR×shift stratum,
so only φ/μ∞ specifically, not SNR or shift size, can drive the
ordering, rejects at p<0.00005 over 20,000 draws;
`experiments/exp12_spearman_null_test.py`), fast-regime cells (μ∞ ≥ k = 0.5) detect
0.83–1.00 while never-regime cells (μ∞ < 0.5) detect 0.07–0.67, and at
3σ the detector escapes the trap at low persistence (0.98/0.97 at
φ = 0.5/0.8) but is caught at high persistence (0.65/0.30 at
φ = 0.95/0.99) — while the raw CUSUM detects at 0.96–1.00 for *every* φ,
so the φ-dependence is specific to the filtered detector. The negative
first-moment result is therefore a boundary condition of *persistent*
latent states — which is the empirically relevant case (trend output, a
natural rate, a volatility level are near-unit-root), and the case the
rest of the paper studies at φ = 0.95. The one off-trend cell — φ = 0.99,
low SNR, where detection (0.63) exceeds what μ∞ (0.21) predicts — is
honest evidence of the transient's own contribution: at near-unit-root φ
the adaptation transient is so long that its accumulated mass fires the
CUSUM even when the asymptotic drift is negligible (the "fast" branch,
strengthened by a long transient). μ∞ governs the post-transient tail;
total detection adds the transient mass.

![**Figure 1.** The φ sweep (`grid_v6_phisweep`). Left: predicted
asymptotic innovation drift μ∞ against the observed innovation-CUSUM
detection rate for every φ × SNR × shift cell (point labels = φ); μ∞
sorts detection (Spearman 0.94), and the dotted line marks the
fast-or-never boundary k = 0.5. Right: at 3σ the detector escapes the
trap at low persistence (0.97–1.00 at φ = 0.5–0.8) and is caught at high
persistence (0.30–0.67 at φ = 0.95–0.99).](paper_assets/grid_v6_muinf_scatter.png)

**Assumptions and estimation error.** Propositions 1–2 assume the
steady-state filter with *known* parameters. Two facts keep this from
being a limitation in practice. First, the filter reaches its steady
state within the 125-observation training prefix of §2 (25% of T = 500),
well before monitoring begins (the innovation autocorrelation is flat by
then). Second, the
error from estimating (φ, q, r) rather than knowing them is second-order:
in the M1 equivalence check (§5, `experiments/exp07`) the ARIMA and
Kalman standardized-innovation series computed with *estimated*
parameters correlate at ρ̄ = 0.99 with each other and, computed with
*true* parameters, at ρ = 1.000 to a max discrepancy of ≈10⁻⁹ — so the
entire estimated-vs-true gap is the small residual that leaves the
median correlation at 0.99. The theory describes the estimated filter to
within that gap. The scope of this claim should be stated precisely:
ρ̄ = 0.99 establishes agreement of the innovation *series* under
estimation; the detection-rate consequences of estimation error are
shown empirically — every grid in the paper is run with estimated
parameters — not proved analytically. On at least one margin this gap
is not second-order: for the Table 2 flagship cell (SNR 0.5, φ = 0.95,
level 3σ), holding sidedness and calibration fixed and replacing
estimated (φ, q, r) with the arena's true values raises the two-sided
innovation-CUSUM detection rate from 0.554 to 0.970, and pairing known
parameters with a one-sided CUSUM (a common alternative convention,
not what Table 2 reports) reaches 0.990 — so a reader reimplementing
this cell under either convention should expect a substantially higher
detection rate than 0.554, which reflects the estimated-parameter,
two-sided construction actually used, not an error
(`experiments/exp10_cusum_ablation.py`).

**A dense magnitude continuum, not just the reported points.** The
level-shift results above are demonstrated at 0.5/1/3 σ_ref; a
referee-requested check reruns the actual raw and innovation CUSUM
detectors (`make_raw_cusum_detector`, `make_innovation_cusum_detector`),
with estimated (training-prefix MLE) parameters as in Table 2, across a
dense 21-point magnitude grid (0.0–4.0 σ_ref in steps of 0.2) at the
three benchmark SNRs, φ = 0.95, T = 500
(`experiments/exp11_break_magnitude_sweep.py`,
`paper_assets/exp11_level_sweep.csv`). Raw's dominance holds at every
one of the 63 grid points (0 violations where raw trails innovation by
more than 3pp) — not just the two or three magnitudes reported in
Table 2. The knife-edge framing needs one qualification the discrete
points did not surface: the *empirical* ratio at which the innovation
CUSUM's detection rate crosses 50% (ratio ≈ 2.0–2.4 across the three
SNRs) sits well below the ratio at which the *theoretical* μ∞ crosses
k = 0.5 (ratio ≈ 3.1–3.5) — consistent with the transient-mass effect
already noted for the φ = 0.99 low-SNR cell above, now shown to be a
general, continuous phenomenon rather than a single favorable cell: μ∞
still sorts detection (the φ-sweep Spearman result), but is not itself
the empirical half-detection point.

## 5. Second moments: a whitening ladder

Is the latent layer's second-moment advantage about the *state estimate*,
or merely about *prewhitening* the autocorrelated observations before
applying a variance statistic? We probe this with a three-rung ladder,
calibrated by the same routine on the same matched-null seed blocks:

- **raw** — a variance CUSUM (up-arm Page CUSUMs of z²−1 at allowances
  k = 0.25 and 0.05, a down-arm CUSUM of 1−z², max over arms, no
  per-time standardization) on z from the raw observations, standardized
  by frozen training-prefix moments;
- **ARIMA** — the identical statistic on the standardized one-step
  residuals of an AIC-selected, training-prefix-frozen ARIMA model
  (whitened, but not state-aware);
- **latent** — not the same statistic re-run on Kalman innovations, but
  the paper's eleven-feature diagnostic composite (§2) evaluated on the
  Kalman-filtered path, whose score is a max over features each
  standardized per time point against Monte Carlo null replications.

The raw and ARIMA rungs isolate one axis — prewhitening — holding the
detection statistic fixed. The latent rung changes two things at once,
the information set (Kalman-filtered, state-aware) and the statistic
itself (composite, per-time-standardized, versus the lower rungs'
unstandardized 3-arm CUSUM), so its comparison to raw/ARIMA below should
be read as *state-aware composite* vs. *whitened single statistic*, not
as a third setting of one controlled instrument.

**The ladder has three rungs but only two are distinct.** For the AR(1)
+ noise DGP the observable Y has an *exact* ARMA(1,1) reduced form:
differencing by (1 − φL) leaves an MA(1), whose invertible root gives an
MA parameter θ and innovation variance σ_ε² satisfying two identities we
verify to machine precision (Appendix B; `lsc.theory
.arma11_representation`) — σ_ε² = F, the Kalman innovation variance, and
θ = ρ = φ(1 − K), the Proposition-1 decay rate. So the steady-state
Kalman innovations *are* the ARMA(1,1) innovations: the ARIMA and latent
rungs are the same filter, not two competitors. We confirm this
numerically (`experiments/exp07_arma_equivalence.py`, ≥200 null paths):
with true parameters the two standardized-innovation series agree to a
median correlation of 1.000 (max discrepancy ≈ 10⁻⁹), and with each rung
fit on the training prefix — the actual operating condition — they still
correlate at ρ̄ = 0.99, the small wedge being pure estimation error
(forcing the ARIMA order to the true (1,0,1) tightens it to 0.9995).
(AIC, incidentally, rarely selects (1,0,1): near the unit root at φ = 0.95
it prefers (1,0,0) or a differencing (0,1,1); this is a benign near-unit-
root artifact — those orders approximate the ARMA(1,1) closely enough to
preserve ρ̄ ≥ 0.95 — reported in full in Appendix B.) The practical
consequence: the ladder is really **raw vs. whitened**, and "does the
*state estimate* help beyond ARIMA whitening?" has the answer *no, by
construction*. What remains is the genuinely empirical question — when
does whitening help at all? — which turns out to depend on *which
variance channel* breaks.

**Table 3. The ladder, both channels** (detection rate at T = 500, 5%
calibrated FAR; MC SEs ≤ 0.02 in `paper_assets/ladder_table.csv`,
`break_channel` column).

| channel | break | rung | SNR 0.1 | SNR 0.5 | SNR 2.0 |
|---|---|---|---|---|---|
| **r** (obs-noise) | ×1.5 | raw | **1.00** | 0.56 | **0.10** |
|                   |      | ARIMA | 0.90 | 0.94 | 0.87 |
|                   |      | latent | 0.82 | 0.87 | 0.91 |
|                   | ×3   | raw | 1.00 | 1.00 | 0.85 |
|                   |      | ARIMA | 0.98 | 1.00 | 1.00 |
|                   |      | latent | 0.99 | 0.99 | 0.98 |
| **q** (state-innov) | ×1.5 | raw | **0.09** | 0.21 | **0.23** |
|                     |      | ARIMA | 0.03 | 0.10 | 0.16 |
|                     |      | latent | 0.06 | 0.11 | 0.23 |
|                     | ×3   | raw | 0.72 | 0.96 | 0.96 |
|                     |      | ARIMA | 0.26 | 0.79 | 1.00 |
|                     |      | latent | 0.44 | 0.76 | 0.98 |

**Reading the ladder: the ordering inverts across channels.** The subtle
×1.5 break is the discriminating case, and it tells opposite stories in
the two channels.

*Observation-noise (r) breaks — prewhitening wins.* The *raw* rung is
strongly SNR-dependent, falling monotonically from 0.996 (SNR 0.1)
through 0.560 to 0.102 (SNR 2.0, within 5 pp of the 6.0% empirical FAR,
i.e. chance). The mechanism is transparent: as SNR rises the latent
state's variance dominates the marginal variance of Y, so a ×1.5 change
in the (now-small) observation-noise component is a shrinking fraction of
total variance, masked by state-driven autocorrelation. Prewhitening
removes exactly that autocorrelation: the *ARIMA* rung is flat across SNR
(0.90 / 0.94 / 0.87) and the *latent* rung tracks it closely (0.82 /
0.87 / 0.91) — an empirical proximity, not a consequence of the
equivalence above, which concerns the shared innovation series rather
than the composite statistic built on top of it. On this channel the
advantage over raw is *prewhitening under autocorrelation*, decisive
where the latent signal masks the noise change.

*State-innovation (q) breaks — the ordering inverts, read separately by
break size.* The *coarse* ×3 break carries the inversion claim cleanly:
every rung detects above chance, and the raw rung matches or beats the
whitened rungs at every SNR (raw 0.72 / 0.96 / 0.96 vs ARIMA 0.26 /
0.79 / 1.00, the whitened rung catching up only at the high-SNR
ceiling). The *subtle* ×1.5 break must be read against the same
≤5-pp-from-FAR "chance" standard applied to the r channel above
(empirical FARs on this grid are 4.2–6.6%): raw at SNR 0.1 (0.09) and
ARIMA at SNR 0.1 and 0.5 (0.03, 0.10) are within 5 pp of their empirical
FARs — chance — and only the higher-SNR cells clear the standard (raw
0.21 / 0.23 at SNR 0.5 / 2.0; ARIMA 0.16 at SNR 2.0). The honest
subtle-break statement is therefore not "raw detects it" but "whitening
fails to recover it": no rung detects a ×1.5 shock-variance break well,
and whitening only loses ground. The raw rung's SNR-dependence still
*reverses sign* relative to the r channel — it rises with SNR (0.09 →
0.21 → 0.23) — and the mechanism is the mirror image: a
q-break inflates the *state's own* variance, which dominates the marginal
variance of Y at high SNR, so a raw z² statistic sees it directly —
whereas prewhitening *strips out* the state-carried signal along with the
autocorrelation. Prewhitening reveals a break in the white component and
removes a break in the state. Quieting (×⅔, i.e. reduced q) is
undetectable by every rung (≤ 0.07, at FAR): a low-q state contributes
too little to Y to register its own reduction.

*Where does raw's q-break advantage come from? A φ sweep.* Holding
the shock variance q and observation variance r fixed and sweeping the
persistence φ makes the state's stationary variance q/(1−φ²) — and hence
the induced SNR — rise as the **1/(1−φ²) amplification factor**
(Figure 2; `configs/grid_v8_phiqbreak.yaml`). The result is instructive
and only partly what one might guess.
On the *subtle* ×1.5 break, raw's advantage Δ = detect(raw) −
detect(ARIMA) is driven by exactly this amplification: it vanishes when
the state is white (Δ = 0.00 at φ = 0.1, where whitening is a no-op),
rises with the amplification (Spearman 0.83), and — the clean check —
equals the SNR-sweep value from the fixed-φ grid at the *same induced
SNR* (Δ = 0.11 vs 0.11 at SNR 0.5; 0.07 vs 0.07 at SNR ≈ 2), so the
φ-sweep and the SNR-sweep are one experiment: raw's edge on a subtle
shock-variance break is the state's variance share, nothing more. (These
Δ values are differences between near-floor detection rates — e.g. 0.21
vs 0.10 at SNR 0.5 — so they measure the visibility of a marginal
advantage, not a large power gap.) But the
law is not monotone to the unit root — Δ peaks at φ = 0.95 and recedes at
φ = 0.99, because there the raw detector's own baseline degrades (its
calibrated threshold jumps from 275 to 1829, the nonstationarity penalty
of §8.4). And on the *coarse* ×3 break the amplification story does not
hold at all: raw beats ARIMA at every φ, including φ = 0.1 (Δ = 0.34),
with Spearman(amp, Δ) negative — a gross shock-variance break is visible
to raw z² regardless of amplification, while whitening removes it. (This
is the whitening itself, not order mis-selection: at low φ the ARIMA rung
selects a stationary AR(1), not a differencing model.) The honest reading:
prewhitening strips the state-carried variance signal for q-breaks of any
size — that is B2 — and the 1/(1−φ²) amplification governs only *how
visible* the residual raw advantage is on the marginal, subtle break.

| φ | Amplification 1/(1−φ²) | Δ, subtle ×1.5 (raw − ARIMA) | Δ, coarse ×3 (raw − ARIMA) |
|---|---|---|---|
| 0.10 | 1.01 | 0.000 | 0.344 |
| 0.50 | 1.33 | 0.016 | 0.526 |
| 0.70 | 1.96 | 0.038 | 0.528 |
| 0.85 | 3.60 | 0.104 | 0.204 |
| 0.95 | 10.26 | 0.112 | 0.168 |
| 0.99 | 50.25 | 0.074 | 0.302 |

*Table 4. Raw's detection-rate advantage over the ARIMA rung (Δ),
swept over φ at fixed q, r (`grid_v8_phiqbreak`). On the subtle break Δ
tracks the amplification factor and peaks at φ = 0.95 before receding
at the unit-root edge; on the coarse break Δ stays large at every φ,
including φ = 0.1 where amplification is negligible.*

![**Figure 2.** The φ × q cross-grid (`grid_v8_phiqbreak`). Left: raw's
advantage Δ = detect(raw) − detect(ARIMA) against φ; the subtle ×1.5
advantage vanishes at φ = 0.1 and peaks at φ = 0.95, while the coarse ×3
advantage persists at every φ. Right: the same Δ against the 1/(1−φ²)
amplification factor, with the fixed-φ SNR-sweep values (grid_v5)
overlaid — on the subtle break the φ-swept and SNR-swept experiments
coincide at matched induced SNR; note the Δ values are differences
between near-floor detection rates.](paper_assets/grid_v8_phiq_amplification.png)

This resolves the pre-registered decision rule (`experiments/CHANGELOG.md`)
as **Outcome B2**: the "prewhitening beats raw" result is *specific to
the observation-noise channel*. It matters because the events the
application targets — the Great Moderation, crisis volatility — are
state-innovation (shock-variance) breaks, not observation-noise breaks;
on that channel a raw variance CUSUM is at least as good as any amount of
whitening — decisively on the coarse break, while on the subtle break
every rung is near the floor and whitening only loses ground. This is not a weakness to hide but the explanation for a
real-data fact (§9): the raw variance CUSUM's crisis timing is
*indistinguishable* from the state-aware composite's, exactly as the q-
channel predicts. (The earlier r-channel-only framing recorded a
provisional "Outcome C"; the q channel is what disambiguates it.)
Attribution: the z²/e² variance-pressure arms drive the alarms at every
rung.

**Runway (T sweep, ×1.5, SNR 0.5).** The subtle-break detection scales
with sample length for every rung: at T = 200 all three are weak (raw
0.26, ARIMA 0.10, latent composite 0.11) — the whitened rungs need runway
to accumulate the CUSUM — and by T = 2000 all reach ceiling (raw 0.98,
ARIMA ≈1.00, latent 0.99). The T = 200 calibration is *not* hot for these
variance detectors (empirical FAR 5.0–5.8% vs the 5% target), unlike the
level CUSUMs (§8.1). At the coarse ×3 break the composite is at ceiling
even at T = 200 (1.00, median delay 26 of the ~100 post-break
observations, best level-oriented benchmark 0.17).

Scale *quieting* (×⅔) inverts the up-break pattern and is treated in §6
and §8.3: there the raw rung catches the low-SNR case (0.32 at SNR 0.1,
where noise dominates) while the whitened rungs and composite sit at
chance, and only the standalone exceedance detector recovers it at
moderate SNR.

## 6. Dynamics: near the information floor

Pure persistence changes (marginals preserved) are close to undetectable:
at SNR 0.5, no method exceeds FAR (a pre-registered hypothesis wrong in
both directions — raw CUSUM even reached 0.16 on persistence-up via a
conditional "level-freeze" artifact we dissect). Quieting changes
(φ down) actively *suppress* every excursion statistic below its null
level. Purpose-built quietness features (CUSUM of 1−e², rolling
innovation autocorrelation) rescue detection only where the information
exists: 0.33 at SNR 2.0 (the only above-FAR persistence detection
anywhere in the grid) and 0.17 at T = 2000. That 0.33 is a single
favorable cell, not a broad capability: its value is that it maps where
the information floor sits, showing detection is possible exactly where
the state is most visible and nowhere else. Scale-quieting (×⅔) is
detectable — but, outside the noise-dominated regime, only by the
exceedance detector (§8.3): at SNR 0.5 it reaches 0.41/0.33 (Gaussian/t₅)
while every other method sits at chance. The one exception is the same
low-SNR window that lifts the raw rung on the up-breaks (§5): at SNR 0.1
the raw variance CUSUM catches the quieting too (0.32), because when
observation noise dominates a scale change is directly visible in raw z²
regardless of its sign.

## 7. Multiple breaks: everyone is one-shot

Under a re-arm protocol applied identically to all methods (re-arm when
the score drains below half threshold plus a 20-obs refractory), raw
CUSUM never detects a second event (recall 0.00 — its fixed-baseline
statistic saturates and cannot drain), but the LSC CUSUMs rarely re-arm in
time either at 150-observation spacing (level→level second-event recall
≤ 0.05): first-alarm delay plus drain time exceeds the gap. The exception
is cross-channel pairs: level→variance is caught by the composite alone
(second-event recall 0.60, F1 0.63) because its variance features were
never saturated by the level event. Re-arming costs almost nothing under
the null (≤ 1.2% of null paths give a second alarm) — saturation, not
chatter, binds.

**A bounded-memory fix, and its limits.** The diagnosis
above — a fixed-baseline statistic compares every observation against
the *original* training-prefix reference and so never drains after a
permanent shift — points to a specific repair: replace the fixed
reference with a moving one. We build a MOSUM-style two-window
statistic (the same family as the CUSUM-of-recursive-residuals
monitoring literature; Chu, Stinchcombe & White 1996) that compares a
trailing window's mean to the window immediately before it, rather
than to the training baseline, for both the raw-Y and
innovation-CUSUM channels (`windowed_break_pressure`). On the level→level scenario this closes the
gap dramatically: the windowed raw-Y statistic's second-event recall
rises from 0.004 to 0.682 — essentially matching its own first-event
recall (0.692) — at *higher* precision than the unwindowed statistic
(0.99 vs 0.80), a real fix rather than a threshold trick, at a modest
first-event recall cost (0.692 vs 0.738). The windowed innovation-CUSUM
improves less (second-event recall 0.008 → 0.234) because the filter's
own adaptivity (μ∞, Proposition 1) already partially "forgets" a level
shift, leaving less room for a moving reference to add. The fix is
channel-specific, not general: on level→variance and variance→variance
scenarios, both windowed statistics stay at second-event recall ≈ 0.00,
because they are mean-shift statistics and a pure variance change
carries no mean signal for a moving-window *mean* comparison to see —
closing that gap needs a windowed *variance* statistic (a
moving-window analogue of the r/q-channel CUSUMs of §5), left to future
work. Multi-break detection therefore needs the statistic's channel
matched to the break, exactly as the single-break results of §5 already
required — the two sections are the same lesson at different time
scales, and the one real economic setting where it bites is a
recession cluster (§9): a level shock followed by a shock-variance
regime change is exactly the level→variance case the composite already
handles, but two level shocks in close succession — a double-dip — are
exactly where every fixed-baseline statistic here still fails.

## 8. Robustness

We stress-test the §5–§7 results along five axes: sample size,
distributional misspecification, a repaired heavy-tailed statistic,
composite-vs-standalone dilution, and real-data sensitivity to the
false-alarm target and window length.

**8.1 Sample size.** §5 numbers; short-T caveat: heavy-tailed null maxima
make quantile thresholds noisy, and at T = 200 the two baseline CUSUMs
calibrate hot (8.8–9.4% vs the 5% target). Quote empirical FARs alongside
any short-sample power claim.

**8.2 Misspecification.** t₅ observation noise: rankings preserved; raw
CUSUM unhurt on levels (0.99); the composite's subtle-variance case
collapses (×1.5: 0.87 → 0.16) — repaired in §8.3. The raw and ARIMA
variance rungs are far less tail-fragile than the composite: on the ×1.5
break at SNR 0.5 the raw rung falls only 0.56 → 0.43 and the ARIMA rung
0.94 → 0.74 under t₅ (against the composite's 0.87 → 0.16), because the
plain max-over-arms z²/e² statistic without per-time composite
standardization retains the tail excursions that carry the variance
signal — the same logic that motivates the standalone exceedance detector
(§8.3). At ×3 all variance rungs stay at ceiling under t₅ (raw and ARIMA
1.00, composite 0.97). Nonlinear tanh drift
(mildly bimodal state): level detection collapses for *every* method
(raw 0.15 at 3σ — the state's own regime-hopping inflates all null
thresholds), while variance detection is untouched (0.97 at ×1.5). The
two detection families read disjoint information channels.

**8.3 Heavy tails and the exceedance repair (a three-act story worth
telling honestly).** (i) Huberizing e² (clip at 2.5·MAD) — falsified,
worse everywhere, even Gaussian (×1.5: 0.87 → 0.06): the variance signal
lives in the tail the clip removes. (ii) An exceedance-indicator CUSUM
(count of |e| above its training 90th percentile; bounded summand under
any distribution) — the raw statistic separates the null and break
classes near-perfectly. Dropped into the composite in place of the
e²-pressure features (the "robust2" variant), and standardized
per-time-point as in §8.4, it is *diluted, not dead*: it reaches 0.58 at
×1.5 (SNR 0.5, Gaussian) and 0.21 at ×1.5 under t₅ — where it even edges
the e²-based composite's heavy-tail collapse (0.16, §8.2) — and ≈0.97 at
×3 under t₅. But it trails the standalone detector of (iii) at the subtle
break, because a max-over-≈10-features composite spreads the calibrated
5% false-alarm budget across all features: the exceedance feature's per-t
standardized score climbs past z ≈ 19 on a ×1.5 break, so bounded
increments *do* reach discriminating ratios (an earlier pooled-scale
build, before the §8.4 fix, wrongly suggested otherwise), yet the shared
threshold still costs it power relative to calibrating the statistic
alone. The remedy is *exposure*, not a different feature. (iii) The same
statistic as a
*standalone* calibrated detector (up-arm k = 0.05, down-arm k = 0.02; k
chosen on non-evaluation seeds, procedure logged): variance ×1.5 at 0.87
Gaussian / 0.75 t₅ (repairing 0.16), ×3 at ~1.0 with ~37-obs delay under
both distributions, and the first successful quieting detection (×⅔:
0.41/0.33). Both headline rates fell 3–5pp short of the pre-registered
bars — reported as such.

**8.4 The local-level (random-walk state) arena: degenerate for levels,
whitening-mandatory for variances.** The canonical latent-state model in
economics is the local level (random-walk state); we ran the level and
variance ladder cells there rather than dismiss it
(Figure 3; `configs/grid_v7_llevel.yaml`),
and it splits cleanly. *Level* detection is degenerate for **every**
method: at 3σ all five detectors sit at the 5% FAR (raw CUSUM 0.07–0.10,
Kalman-innovation 0.04–0.15, raw/ARIMA variance and composite ≤ 0.07). A
level break in a random-walk state is absorbed by a well-specified filter
as one large ordinary innovation — no sustained signal — and the raw-Y
CUSUM has no fixed baseline: its calibrated threshold is 1500–2400
(versus O(10–100) in the AR(1) arena) and it still calibrates hot. There
is no common null against which to rank level detectors, which is why we
study the identifiable AR(1) arena in the body. But *variance* detection
is **not** degenerate — it merely requires whitening. On a ×1.5
observation-noise break the raw z² CUSUM is at chance (0.06 at every SNR;
threshold ≈ 10⁴, meaningless on a nonstationary series) while the
ARIMA-differencing rung detects (0.71 / 0.84 / 0.58) and the Kalman
composite detects (0.97 / 0.89 / 0.68); at ×3 the whitened rungs reach
0.97–1.00 while raw stays at chance. As in §5, the ARIMA and composite
figures here are not two settings of one controlled statistic — the
composite is the broader, per-time-standardized eleven-feature detector,
not the ARIMA rung's own CUSUM re-run on Kalman innovations — so the gap
between them is not evidence that the state estimate adds power beyond
ARIMA specifically. What the two whitened rungs agree on is the
qualitative point, and ARIMA alone already establishes it (0.71/0.84/0.58
vs. raw's 0.06 at every SNR): this is the exact complement of §5's
r-channel AR(1) result — where a raw variance CUSUM could *win* when
observation noise dominates — and it sharpens the paper's thesis:
prewhitening is not merely helpful but *mandatory* once the observable is
nonstationary, because the raw statistic has no stationary baseline to
calibrate against.

![**Figure 3.** The local-level (random-walk state) arena at SNR 0.5
(`grid_v7_llevel`). Left: a 3σ level break is degenerate for every
method — all five detectors sit at the 5% FAR line. Right: a ×1.5
observation-noise variance break is caught by both whitened detectors —
ARIMA-differencing (0.84) and the state-aware composite (0.89, a
distinct, broader statistic, not the same CUSUM re-run on Kalman
innovations) — while the raw variance CUSUM stays at chance: whitening is
mandatory once Y is
nonstationary.](paper_assets/grid_v7_llevel_degeneracy.png)

**8.5 An offline benchmark: PELT, calibrated to the same false-alarm
rate.** Related work dismisses offline changepoint methods
as solving a different problem (retrospective segmentation, not causal
monitoring), which is true but does not by itself say how well an
off-the-shelf method would do if pressed into the same role. We
calibrate PELT (Killick, Fearnhead & Eckley 2012; `ruptures`, l2 cost,
applied to Y standardized by the training-prefix mean/std — the raw
rung's own standardization) to a 5% false-alarm rate on null AR(1)
paths by bisecting its penalty parameter, exactly mirroring the
threshold-calibration protocol used for every causal detector, then ask
whether it *localizes* a break within 25 observations, given the full
sample — an offline-localization question, deliberately not a delay
comparison, since PELT sees future data the causal detectors cannot.

| Arena (SNR) | Scenario | PELT localize rate |
|---|---|---|
| 0.1 | level 0.5σ | 0.02 |
| 0.1 | level 1σ | 0.07 |
| 0.1 | level 3σ | 0.83 |
| 0.1 | variance ×1.5 | 0.00 |
| 0.1 | variance ×3 | 0.20 |
| 0.5 | level 0.5σ | 0.03 |
| 0.5 | level 1σ | 0.11 |
| 0.5 | level 3σ | 0.91 |
| 0.5 | variance ×1.5 | 0.01 |
| 0.5 | variance ×3 | 0.01 |
| 2.0 | level 0.5σ | 0.02 |
| 2.0 | level 1σ | 0.11 |
| 2.0 | level 3σ | 0.92 |
| 2.0 | variance ×1.5 | 0.02 |
| 2.0 | variance ×3 | 0.02 |

*Table 5. PELT localization rate at a FAR-matched (5%) operating
point, n = 300 per cell (`exp08_pelt`). MC SEs ≤ 0.024 (n = 300).*

On the canonical 3σ level break PELT is competitive with the causal
raw-Y CUSUM (0.83–0.92 here vs. 0.97–0.99 in Table 1's arena, seeing
the whole path rather than detecting online) — an off-the-shelf offline
method genuinely can find an obvious level shift. But on variance
breaks it is close to useless (0.00–0.02, against the dedicated raw
variance-CUSUM's ladder-table numbers of 0.10–1.00, §5), because PELT's
default l2 cost model is a mean-shift statistic: it responds to a
variance change only through the second-order effect that within-
segment sum-of-squares grows, a far weaker signal than a statistic
built for variance directly. The comparison sharpens rather than
undercuts the paper's thesis: an off-the-shelf offline segmentation
method is not a substitute for a channel-matched statistic, even when
it is given the unfair advantage of seeing the whole sample.

**Protocol lessons (each cost us a wrong result before it was fixed).**
plain-HMM regime probabilities saturate and cannot be FAR-calibrated on
nonstationary data; probability-scale scores need log-odds; EM needs
persistent-initialization restarts; composite features must be
standardized per-time-point, not pooled; order-statistic thresholds have
Beta(n+1−k, k) noise regardless of distribution, so heavy-tailed
detectors need larger calibration budgets.

## 9. Real data (illustrative)

Four FRED series — industrial production, GDP, and 10-year Treasury
yields (pinned snapshots, 2026-07-11) plus the unemployment rate
(pinned 2026-07-16) — rolling causal monitoring (train 120 months /
monitor 60), per-segment parametric bootstrap calibration at 5% FAR per
window, alarms attributed to the feature that crossed. Throughout this
section the detectors are distribution-free monitors calibrated to a
false-alarm rate; no parametric conditional-variance model is estimated
and no such claim is made. Association with NBER-registered events is
tested by permutation: the observed count of registered events "hit"
(an alarm within 12 months after) is compared to the distribution of
hit counts from 20,000 resamples of the same number of alarm months
drawn uniformly from all monitored months; the reported p is the
fraction of resamples at least as extreme as the observed count, so a
small p means alarms cluster after registered events more than chance
alone would produce (`real_data_eval.py`).

| Series | Method | Alarms | Hits / events | Stray | Perm. p |
|---|---|---|---|---|---|
| INDPRO | lsc_composite | 4 | 3/9 | 1 | 0.008 |
| INDPRO | lsc_kalman_cusum | 2 | 2/9 | 0 | 0.021 |
| INDPRO | lsc_tail_cusum | 5 | 0/9 | 5 | 1.000 |
| INDPRO | raw_cusum | 1 | 1/9 | 0 | 0.148 |
| INDPRO | raw_var_cusum | 5 | 1/9 | 4 | 0.554 |
| GDP | lsc_composite | 3 | 2/9 | 1 | 0.067 |
| GDP | lsc_kalman_cusum | 1 | 1/9 | 0 | 0.164 |
| GDP | lsc_tail_cusum | 3 | 2/9 | 1 | 0.063 |
| GDP | raw_cusum | 0 | — | — | — |
| GDP | raw_var_cusum | 2 | 1/9 | 1 | 0.309 |
| GS10 | lsc_composite | 7 | 1/3 | 6 | 0.325 |
| GS10 | lsc_kalman_cusum | 2 | 1/3 | 1 | 0.109 |
| GS10 | lsc_tail_cusum | 4 | 0/3 | 4 | 1.000 |
| GS10 | raw_cusum | 2 | 1/3 | 1 | 0.104 |
| GS10 | raw_var_cusum | 9 | 1/3 | 8 | 0.402 |
| UNRATE | lsc_composite | 3 | 2/9 | 1 | 0.057 |
| UNRATE | lsc_kalman_cusum | 4 | 4/9 | 0 | 0.0002 |
| UNRATE | lsc_tail_cusum | 6 | 2/9 | 4 | 0.205 |
| UNRATE | raw_cusum | 4 | 4/9 | 0 | 0.0004 |
| UNRATE | raw_var_cusum | 7 | 2/9 | 5 | 0.257 |

*Table 6. Real-data alarm summary at 5% FAR, 120-month training
(`rd_eval.csv`). GDP's raw_cusum fired zero alarms across
all 12 windows, so no permutation test applies. UNRATE (new, P2) shows
the largest raw_cusum/lsc_kalman_cusum association by p-value in the
table (4/4 hits, zero strays, out of 9 peaks in range) — but see the
model-fit discussion below, which complicates a straightforward
reading of that association. GS10's permutation p-values rest
on only 3 registered events, so only four hit-counts (0–3) are achievable
and the resulting p's are far coarser than INDPRO's or UNRATE's (n=9);
read the raw hit/alarm counts there as more informative than the exact
p (the 20,000-draw permutation test is, in effect, approximating a
discrete Fisher's-exact-type test with only four attainable outcomes
on this series). Alarm and hit counts are exact given the fixed historical series,
not Monte Carlo estimates; only the permutation p-values carry
resampling uncertainty, from the 20,000-draw test — SE = √(p(1−p)/20000)
≤ 0.0035 across every p-value reported here.*

**Multiple-comparisons correction.** Table 6 reports 19 valid
permutation tests (5 methods × 4 series, less GDP's zero-alarm
raw_cusum cell, which admits no test). A Bonferroni threshold across
all 19 (α/19 ≈ 0.0026) or a Benjamini–Hochberg FDR procedure at
q = 0.05 (which requires the third-ranked p-value to clear ≈0.0079)
both leave only two entries standing: UNRATE's raw_cusum (p = 0.0004)
and lsc_kalman_cusum (p = 0.0002). INDPRO/lsc_composite's headline
p = 0.008 — the association featured in the abstract — does not
survive either correction; it would need to clear 0.0026 (Bonferroni)
or 0.0079 (its own BH rank) and falls short of both. This is not a
favorable correction to report: the two associations that do survive
are exactly the ones the UNRATE model-fit discussion below flags as
resting on windows where the AR(1) specification is misspecified
(three of UNRATE's four hits sit in φ-clipped windows). Read together,
no single-series NBER association in this table clears both the
multiple-testing bar and the model-fit bar at once — INDPRO clears the
model-fit bar but not the multiple-testing bar; UNRATE's raw_cusum and
lsc_kalman_cusum clear the multiple-testing bar but not the model-fit
bar. We report the INDPRO/composite association at its nominal
p = 0.008 throughout this section, as originally computed, but it
should be read as a suggestive single-series association from an
illustrative application, not a family-wise-significant finding across
the comparison actually run. Both corrections treat the 19 tests as
independent; they are not (several methods share alarm-generating
CUSUM machinery on the same underlying series), so this is a valid but
conservative approximation rather than an exact one — Bonferroni's
validity does not require independence, but a joint, max-statistic
permutation null across methods per series would give a tighter bound
in principle. An attempted implementation
(`experiments/exp13_joint_fwer.py`) against the real INDPRO alarm data
did not yield a usable result: the combined statistic collapsed almost
entirely onto `lsc_composite`'s own marginal test rather than
genuinely pooling evidence across methods, since the joint null did
not model real cross-method correlation from the underlying score
paths. A second attempt, using a circular-shift joint null instead
(shifting all five methods' alarm months by the *same* random amount
each draw — which genuinely preserves whatever real cross-method
timing correlation exists, since a common shift is a rigid rotation
that doesn't touch relative offsets between methods, unlike
independent per-method redraws), gives a materially different and
usable result for INDPRO specifically
(`experiments/exp13c_circular_shift.py`): the total hit count summed
across all five methods gives an *exact* p-value for each series
(the shift's sample space is small and discrete — at most 780
possible values — so every possible shift is enumerated, not Monte
Carlo sampled), against a Bonferroni threshold of α/4 = 0.0125 across
the four series. INDPRO (total hits 7 vs. a null with mean 2.52,
max 10, over 780 possible shifts): p = 0.028, does not survive.
GS10 (total 4 vs. null mean 1.30, max 6, 720 shifts): p = 0.076,
does not survive. GDP (total 6 vs. null mean 1.50, max 6, 240
shifts, quarter-constrained since GDPC1 is quarterly): p = 0.0125
exactly — a mathematical tie with the threshold itself (3/240 =
0.05/4 to machine precision), not a survival or a failure by any
real margin. Half of GDP's six hits come from a single window:
`lsc_composite`, `raw_var_cusum`, and `lsc_kalman_cusum` all fire in
the same quarter (2020-Q2), all hitting the same event (2020-02
COVID) — one synchronized co-firing, not three independent
agreements. UNRATE (total 14 vs. null mean 3.57, max 16, 780
shifts): p = 0.0115, nominally clearing the threshold — but 9 of
its 14 hits (64%) sit in the same φ-clipped (degenerate AR(1) fit)
windows already flagged below as undermining `raw_cusum`/
`lsc_kalman_cusum`'s marginal association, and the GFC window shows
all five methods firing together in that single misspecified
window, not five independent detections; only the remaining 5 hits
(36%), all from the one well-estimated 2020 window, reflect a
model the paper's own diagnostics would trust. Read together: no
series clears both the multiple-testing bar and the model-fit bar
at once, the same conclusion this section already draws from the
per-method corrections above — but this is now a *checked* claim
across all four series (`experiments/exp13d_export_other_series.py`,
`experiments/exp13d_all_series_circular_shift.py`), not the
provisional, INDPRO-only estimate an earlier draft of this section
had to leave incomplete pending real data for the other three
series — see Appendix A for the two bugs this extension found and
fixed along the way.

**Industrial production (INDPRO, 1948–2026).** Composite alarms: 2008-09
and 2020-04 (both variance_pressure), 1990-12 (variance_quiet), 1969-08
(variance_quiet; within the false-alarm budget and reported as such).
Hits 3/9 NBER peaks within 12 months, 1 stray vs 0.7 expected;
permutation p = 0.008 (innovation CUSUM p = 0.021; raw CUSUM 1 hit,
p = 0.15). The **raw variance CUSUM** — the bottom rung of the ladder,
added here to test real-data uniqueness — does catch the GFC (2008-09,
up-arm, the same month as the composite) but embeds it among four stray
quieting alarms (1967, 1968, 1988, 2019) and misses COVID, so its
NBER association is markedly weaker (5 alarms, 1 hit, p = 0.55). The
crisis *is* detectable by a raw variance statistic; what the latent layer
buys on real data is a *clean* alarm profile — an uncorrected, low-stray
association (p = 0.008 vs. 0.55) — not the crisis catch itself, and even
that comparison is within-series selectivity rather than a
family-wise-significant result (see the multiple-comparisons correction
above). A permutation-test
implementation note: the p-values in this section are recomputed with a
per-series, per-method seed (2026-07-16) after we found the original
shared, sequentially-consumed random generator made one series' p-value
depend on which *other* series happened to be present in the run —
harmless for any single number in isolation, but a reproducibility
hazard the fix removes; every value moved by ≤ 0.002 from the
originally reported estimates, consistent with Monte Carlo noise at
20,000 draws, not a change in any underlying alarm.

**What real-time data changes (ALFRED vintages).** Re-running each
decision on the data *as it existed that month* both tempers and sharpens
the timing claims. COVID: robust — the composite's real-time alarm is at
the 2020-04 vintage (data month 2020-03), one month earlier than on
revised data and ~2 months before the NBER announcement. GFC: downgraded —
the crossing is at the same data month (2008-09) but only in the 2008-12
vintage, so real-time knowledge is coincident with the NBER announcement
(2008-12-01), though still 4 months ahead of the raw level CUSUM's
real-time alarm (2009-04). Because the raw variance CUSUM alarmed on
revised INDPRO, we ran it through the identical vintage protocol for an
apples-to-apples comparison: its real-time timing is *indistinguishable
from the composite's* — same vintage and same data month for both GFC
(2008-12 vintage, 2008-09) and COVID (2020-04 vintage, 2020-03), both
ahead of the raw level CUSUM. On real-data crisis *timing*, then, the
prewhitening rung and the state-aware composite are interchangeable; the
composite's advantage is confined to the clean association profile of §9
and to the simulation-calibrated ×1.5 subtlety threshold of §5, not to
real-time crisis detection.

**GDP (GDPC1, quarterly).** GFC and COVID caught (variance_pressure,
2008Q4 and 2020Q2). The raw variance CUSUM again catches both crises
(2009Q2 and 2020Q2, up-arm) with fewer alarms than on INDPRO (2 alarms,
1 hit, p = 0.31) — the same "catches the crisis, misses the clean
association" pattern. The registered Great Moderation event (1984Q1) is an
honest miss: causal rolling monitoring detects the quieting only in
1992Q2 (composite) / 1997Q1 (tail shortfall), because 60-quarter training
windows contain 1970s volatility until the early 1990s. Retrospective
full-sample break dates are not reproducible by honest monitoring.

**10-year Treasury yield changes (GS10).** Volcker regime caught 4
months after the registered 1979-10 event (variance_pressure; raw and
innovation CUSUM the same month); the post-Volcker disinflation
quietings are flagged (1989–1996, shortfall/variance_quiet); the 2008
ZLB event is missed within 12 months. A third registered event, the
2022-03 hiking-cycle onset, is missed by *every* method within 24
months (`rd_gs10_summary.csv`) — unlike Volcker, a persistent
directional yield rise without a matching shock-variance regime shift
falls outside every detector's reach here, a genuine miss reported as
such rather than a horizon-window artifact. The raw variance CUSUM also
flags Volcker (1980-02, up-arm) but is the noisiest detector on this
series (9 alarms, 8 stray, 1/3 events hit, p = 0.40) — on a series that
is *all* volatility regime, a raw z² statistic fires on every regime
shift, which is exactly why its event-association washes out. (The
Volcker-catch window itself is well-fit — Ljung-Box p=0.42/0.19,
φ=0.24, unclipped — so unlike UNRATE below, this alarm is not an
artifact of degenerate model fit; the intervening windows bracketing
the shock, 1973–1988, fit far worse, but neither produces the reported
alarm.)

**Unemployment rate (UNRATE, monthly).** Table 6's raw_cusum and
lsc_kalman_cusum results on UNRATE look like the strongest association
in the real-data application — 4/4 hits, zero strays, p=0.0002–0.0004
— but a model-fit check (`experiments/exp09_real_data_fit_check.py`)
complicates that reading. In three of the four windows producing these
hits (the 1974, 2001, and 2008 recessions), the AR(1) MLE fits a
*negative* φ (−0.82, −0.19, −0.41) before the pipeline's clipping to
[0.01, 0.99] — not merely weak persistence, but a sign that the model
class itself is misspecified for these windows, since a stationary
AR(1) with φ∈[0,1) cannot represent the dynamics the likelihood is
pointing toward. The clip substitutes a boundary value for whatever
the unconstrained estimate wanted, and the resulting filter behavior
in these windows (Kalman gains of 0.05, 0.54, and 0.22 — not uniformly
small) has no clean interpretation as "the state estimate found little
persistence." Only the fourth hit, 2020, sits in a window with a
well-estimated, unclipped φ (0.948). The honest summary: UNRATE's four
hits are real, and raw_cusum and lsc_kalman_cusum's close agreement on
their timing is a genuine empirical fact, but three of the four rest
on windows where the paper's own latent-state model does not describe
the series, which weakens UNRATE's standing as evidence for the
diagnostic framework specifically, as opposed to evidence that
unemployment has level-type breaks raw CUSUM is well-suited to catch
regardless of any state-space model. This qualitative reading is now
also a direct test rather than an inference:
`experiments/exp17_unrate_phi_gated.py` excludes the three φ-clipped
windows (segments where the pipeline's clip bound, |φ| at 0.01 or
0.99, actually bound) from both the hit count and the resampling
universe — a gated test cannot compare a restricted numerator against
an unrestricted denominator without biasing the result. Gating drops
both detectors from 4/9 to 1/9 hits (only the well-estimated 2020
window survives) against a resampling universe of 540 of the original
780 monitored months, and the association collapses entirely:
p = 0.1474 for both raw_cusum and lsc_kalman_cusum, against the
ungated 0.0002–0.0004. Properly isolated from the model-misspecified
windows that produce most of its apparent significance, UNRATE's
association is not distinguishable from chance.

**Sensitivity.** All numbers below are on INDPRO, the headline series.

| Variant | Method | Alarms | Hits / events | Perm. p |
|---|---|---|---|---|
| FAR 1% | lsc_composite | 3 | 3/9 | 0.003 |
| FAR 1% | lsc_kalman_cusum | 1 | 1/9 | 0.146 |
| FAR 1% | lsc_tail_cusum | 1 | 0/9 | 1.000 |
| FAR 1% | raw_cusum | 1 | 0/9 | 1.000 |
| FAR 1% | raw_var_cusum | 4 | 2/9 | 0.100 |
| FAR 5% (baseline) | lsc_composite | 4 | 3/9 | 0.008 |
| FAR 5% (baseline) | lsc_kalman_cusum | 2 | 2/9 | 0.021 |
| FAR 5% (baseline) | lsc_tail_cusum | 5 | 0/9 | 1.000 |
| FAR 5% (baseline) | raw_cusum | 1 | 1/9 | 0.148 |
| FAR 5% (baseline) | raw_var_cusum | 5 | 1/9 | 0.554 |
| FAR 10% | lsc_composite | 5 | 2/9 | 0.148 |
| FAR 10% | lsc_kalman_cusum | 3 | 2/9 | 0.055 |
| FAR 10% | lsc_tail_cusum | 5 | 2/9 | 0.148 |
| FAR 10% | raw_cusum | 2 | 2/9 | 0.018 |
| FAR 10% | raw_var_cusum | 6 | 1/9 | 0.619 |
| FAR 20% | lsc_composite | 9 | 1/9 | 0.773 |
| FAR 20% | lsc_kalman_cusum | 3 | 2/9 | 0.052 |
| FAR 20% | lsc_tail_cusum | 7 | 1/9 | 0.682 |
| FAR 20% | raw_cusum | 2 | 2/9 | 0.020 |
| FAR 20% | raw_var_cusum | 8 | 1/9 | 0.727 |
| Window 180 mo | lsc_composite | 14 | 2/8 | 0.556 |
| Window 180 mo | lsc_kalman_cusum | 3 | 2/8 | 0.046 |
| Window 180 mo | lsc_tail_cusum | 5 | 3/8 | 0.015 |
| Window 180 mo | raw_cusum | 2 | 2/8 | 0.017 |
| Window 180 mo | raw_var_cusum | 14 | 1/8 | 0.878 |

*Table 7. INDPRO sensitivity to the false-alarm target (1/5/10/20%,
120-month training) and to a longer training window (180 months, 5%
FAR). Alarm and hit counts are exact given the fixed historical series,
not Monte Carlo estimates; only the permutation p-values carry
resampling uncertainty, from the 20,000-draw test — SE = √(p(1−p)/20000)
≤ 0.0035 across every p-value reported here.*

Two independent stress tests both isolate the composite's association
as fragile *relative to itself*, not to the other detectors. The FAR
sweep is monotone and clean for the composite: tightening from 5% to
1% *sharpens* the association (p = 0.008 → 0.003, alarm count falls
from 4 to 3 as the marginal, least-reliable alarm drops out), while
relaxing to 10% and 20% floods it with noise (p = 0.148, then 0.773) —
exactly the dilution a well-behaved calibrated statistic should show.
raw_cusum and lsc_kalman_cusum stay comparatively stable across the same
sweep (p in the 0.02–0.15 range throughout), because they fire too few
alarms for the FAR target to matter much either way. Training window
180 months breaks both variance-based detectors' bootstrap calibration
on nonstationary real data — the composite and the raw variance CUSUM
each fire 14 alarms / 21 windows (p = 0.556 and 0.878, both
uninformative) — while the level (raw CUSUM), innovation, and tail
detectors stay sane (2–5 alarms, p as low as 0.015–0.046). It is the
*second-moment* statistic, not the composite machinery, that is
sensitive to a training window long enough to straddle a volatility
regime; training windows must be short enough to be locally stationary
(120 months worked).

## 10. Discussion

**What, then, is the latent layer actually for?** The results invite a
deflationary reading, and we take it seriously rather than deflect it. For
raw *detection power*, the state-space layer is largely redundant: raw
CUSUM owns level shifts (§4); ARIMA whitening owns observation-noise
variance and, by the exact ARMA(1,1) equivalence, *is* the Kalman
innovation filter — the state estimate contributes nothing beyond it
(§5); and on the shock-variance channel that motivates the application, a
raw variance CUSUM matches or beats whitening outright (§5, whose φ sweep
traces the residual advantage to the state's 1/(1−φ²) variance
amplification on subtle breaks). A practitioner armed with a raw CUSUM,
a raw variance CUSUM, and an off-the-shelf ARIMA residual CUSUM would
reproduce most of the detection frontier without ever writing down a
latent state. So what survives as genuinely the latent layer's? Four
things, and only these. (1) *Dynamics, at the floor.* Persistence and
quieting changes have no raw analogue; the state features (rolling
innovation autocorrelation, the 1−e² quietness CUSUM) are the *only*
above-FAR detection of a pure persistence change anywhere in the grid —
but that detection is a single favorable cell (0.33 at SNR 2.0), a
capability at the information floor rather than a robust one. (2) *Speed.* Conditional on firing, the innovation CUSUM is the
fastest level detector (median delay 24–53 vs raw's 58–91) — the very
"fast" half of fast-or-never. (3) *A single attributable instrument.* The
composite reads ~10 channels under one calibrated FAR budget and reports
*which* crossed, which is what turns a real-data alarm into an
interpretable event (every crisis alarm attributes to a named
second-moment feature, §9). (4) *A clean association profile.* On real
data the composite's crisis *timing* is no better than a raw variance
CUSUM's, but its *selectivity* is — a lower, less stray-alarm-laden
association with NBER dates on the same series (uncorrected p = 0.008
vs 0.55) that a raw z² statistic firing on every wiggle cannot match.
This is a within-series comparison of selectivity, not a claim that
either p-value is family-wise significant across the full real-data
grid — it is not (§9). The honest one-sentence answer: the latent
layer is not a better detector of any single break type; it is a
*breadth-and-interpretation* instrument whose irreducible content is
dynamics (weakly) and attribution (robustly), not the second-moment
power one might have expected filtering to buy.

The mechanics behind that verdict are summarized in four points.

- The latent-state diagnostics layer is not a better level-shift
  detector; it is a *different instrument*, reading second moments and
  dynamics. But the whitening ladder (§5) sharpens the claim past the
  point where "the state estimate" survives as an ingredient on the
  observation-noise channel: the Kalman innovations *are* the ARMA(1,1)
  innovations (an exact reduced-form equivalence, verified to machine
  precision), and empirically the ARIMA rung tracks the composite closely
  there (0.90/0.94/0.87 vs 0.82/0.87/0.91) — so on this channel
  prewhitening is doing the work, even though the composite is a
  distinct, per-time-standardized eleven-feature statistic rather than
  the ARIMA rung's own CUSUM re-run on Kalman innovations. What the state framing still
  buys is dynamics features (persistence, quieting) and the fast-or-never
  speed edge on levels. The division of labor is now measured (grids),
  reduced (raw vs. whitened, the ARIMA and Kalman filters proven identical
  at the innovation-series level, not the composite's detection numbers),
  and derived (fast-or-never with its φ boundary).
- Practical recipe, now channel-aware. Run raw CUSUM for levels. For
  scale changes the right move depends on *which variance moves*, and one
  usually does not know: (i) if the break is in the observation
  (measurement) noise, *whiten first* (ARIMA residuals suffice; the
  Kalman state adds nothing) — prewhitening is decisive when the series is
  autocorrelated and the latent signal is strong, and a raw variance
  CUSUM works only when observation noise dominates; (ii) if the break is
  in the state's own shock variance — the Great-Moderation / crisis-
  volatility case — a raw variance CUSUM is at least as good as whitening
  at every SNR, and whitening can *hurt*. Because the practitioner rarely
  knows the channel, one might default to running *both* a raw and a
  whitened variance CUSUM. Checked directly against a 50/50-mixed,
  channel-unknown-to-the-detector population (r- and q-channel breaks
  at ×1.5, SNR ∈ {0.1, 0.5, 2.0}, both detectors jointly recalibrated
  to hold one combined 5% false-alarm budget — not run independently
  at their own 5% each, which would silently inflate the compounded
  FAR — `experiments/exp14_mixed_channel.py`), running both is not a
  free win: it loses to the single better detector at every SNR
  tested, with the gap widening as SNR rises — 0.553 (raw) vs. 0.493
  (combined) at SNR 0.1 (a 0.06 loss), 0.560 (ARIMA) vs. 0.490 at SNR
  0.5 (0.07), and 0.623 (ARIMA) vs. 0.457 at SNR 2.0 (0.166) — because
  jointly calibrating two statistics to one FAR budget raises the bar
  each individually must clear, and that tax outweighs the benefit of
  channel-agnosticism in this test. The honest recommendation is
  weaker than "run both, it's free": if forced to pick one detector
  under real channel uncertainty, ARIMA was the better single choice
  in two of the three SNRs tested here; running both is only clearly
  justified if the analyst has a specific reason to think the channel
  mix is skewed toward the regime where raw wins (low SNR). Use the
  exceedance-indicator variant under heavy tails and the composite for
  breadth. Calibrate everything on matched nulls at a common FAR (a common
  ARL₀) and report empirical FARs.
- The calibrated-parity protocol itself is a contribution: it exposed
  every failure mode above, and it is what makes the negative results
  informative rather than anecdotal.
- Limitations / future work: a bounded-memory (MOSUM-style) statistic
  fixes the multiple-breaks re-arm failure for level-type second events
  (§7) but not variance-type ones — a windowed *variance* statistic is
  the natural next step; adaptive composite weighting (breadth tax);
  switching-SSM (Kim filter) model layer; formalizing the
  persistence-break mechanisms; a vol-regime reference set for scoring
  the exceedance detector on real data; a plain GARCH(1,1) benchmark
  is now reported (Related Work) and sits at the false-alarm floor on
  both variance channels, contributing nothing over chance on this
  DGP; a break-aware GARCH variant (allowing its own parameters to
  shift, in the spirit of Bai & Perron 2003) and a full
  stochastic-volatility state-space comparison remain open — the
  plain-GARCH result rules out the most obvious "just use GARCH"
  objection without resolving whether a purpose-built regime-shift
  volatility model would fare differently.

---

## References

Bai, J., and P. Perron (2003). "Computation and Analysis of Multiple
Structural Change Models." *Journal of Applied Econometrics* 18(1),
1–22.

Basseville, M., and I. V. Nikiforov (1993). *Detection of Abrupt
Changes: Theory and Application*. Englewood Cliffs, NJ: Prentice-Hall.

Bollerslev, T. (1986). "Generalized Autoregressive Conditional
Heteroskedasticity." *Journal of Econometrics* 31(3), 307–327.

Brown, R. L., J. Durbin, and J. M. Evans (1975). "Techniques for
Testing the Constancy of Regression Relationships over Time." *Journal
of the Royal Statistical Society, Series B* 37(2), 149–192.

Chu, C.-S. J., M. Stinchcombe, and H. White (1996). "Monitoring
Structural Change." *Econometrica* 64(5), 1045–1065.

Frisén, M. (2003). "Statistical Surveillance: Optimality and Methods."
*International Statistical Review* 71(2), 403–434.

Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle." *Econometrica*
57(2), 357–384.

Killick, R., P. Fearnhead, and I. A. Eckley (2012). "Optimal Detection
of Changepoints with a Linear Computational Cost." *Journal of the
American Statistical Association* 107(500), 1590–1598.

Kim, C.-J., and C. R. Nelson (1999). *State-Space Models with Regime
Switching: Classical and Gibbs-Sampling Approaches with Applications*.
Cambridge, MA: MIT Press.

Kim, S., N. Shephard, and S. Chib (1998). "Stochastic Volatility:
Likelihood Inference and Comparison with ARCH Models." *Review of
Economic Studies* 65(3), 361–393.

Lorden, G. (1971). "Procedures for Reacting to a Change in
Distribution." *Annals of Mathematical Statistics* 42(6), 1897–1908.

McConnell, M. M., and G. Pérez-Quirós (2000). "Output Fluctuations in
the United States: What Has Changed Since the Early 1980s?" *American
Economic Review* 90(5), 1464–1476.

Montgomery, D. C. (2013). *Introduction to Statistical Quality
Control*, 7th ed. Hoboken, NJ: Wiley.

Moustakides, G. V. (1986). "Optimal Stopping Times for Detecting
Changes in Distributions." *Annals of Statistics* 14(4), 1379–1387.

Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*
41(1/2), 100–115.

Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence
Intervals*. New York: Springer-Verlag.

Stock, J. H., and M. W. Watson (2002). "Has the Business Cycle Changed
and Why?" *NBER Macroeconomics Annual* 17, 159–218.

Wald, A. (1947). *Sequential Analysis*. New York: Wiley.

Willsky, A. S., and H. L. Jones (1976). "A Generalized Likelihood Ratio
Approach to the Detection and Estimation of Jumps in Linear Systems."
*IEEE Transactions on Automatic Control* 21(1), 108–112.

---

## Appendix A. Reproducibility

`make all` regenerates every table and figure from pinned seeds
(Python 3.14, statsmodels/hmmlearn; `make fred` / `make realdata` /
`make realtime` for the data applications, snapshots under `data/`). The
referee-hardening round added six reproducible artifacts to the pack:
`exp07` (ARMA equivalence), `grid_v5` (the q-break channel),
`grid_v6` (the φ sweep), `grid_v7` (the local-level arena),
`grid_v8` (the φ×q amplification cross-grid), and `arl` (ARL₀/ARL₁
table); all are pinned-seed and join the existing grids draw-for-draw.
A second round (2026-07-16) added `exp08` (the PELT localization
benchmark, §8.5) and extended `exp04` with the two windowed-CUSUM
methods (§7) and `realdata` with a fourth series (unemployment,
`unrate`), a third GS10 event (the 2022 hiking cycle), and a
false-alarm-rate sweep (1%, 5%, 10%, 20%) on INDPRO. A third addition,
`exp09` (`experiments/exp09_real_data_fit_check.py`,
`paper_assets/exp09_ljungbox_table.csv`), runs a Ljung-Box residual
check and a model-implied-vs-sample ACF comparison on the fitted AR(1)
filter for every rolling training window of all four real-data series
(§9). A fourth, `exp10` (`experiments/exp10_cusum_ablation.py`,
`paper_assets/exp10_cusum_ablation.csv`), ablates sidedness and
known-vs-estimated parameters for the Table 2 flagship cell's
innovation CUSUM (§4). A fifth, `exp13`/`exp13c` (§9's
multiple-comparisons correction), attempts a joint FWER bound across
the five real-data methods per series tighter than treating all 19
tests as independent: a first implementation
(`experiments/exp13_joint_fwer.py`) redrew each method's alarm months
independently within a shared null draw, which does not model real
cross-method correlation and collapsed the combined statistic onto
`lsc_composite`'s own marginal test; it is kept in the repository as a
documented negative result, not deleted, since the paper's standing
practice is to report failed attempts rather than remove them. A
second implementation (`experiments/exp13c_circular_shift.py`) shifts
all five methods' alarm months by the same random amount per draw — a
rigid rotation that preserves real cross-method timing correlation
exactly, unlike independent redraws. An initial version of this shift
had its own bug, caught before being trusted: it wrapped alarms into
an interval anchored at a fixed 1948 epoch rather than the monitored
window's own true start (1958 for INDPRO), so any event at or beyond
the window's width from that epoch — the 2020-02 COVID NBER peak,
concretely — was structurally unreachable by any shifted alarm, at
any seed, inflating the apparent significance (p = 0.021–0.023 instead
of the corrected 0.027–0.029; two of the seven observed hits were
against exactly that unreachable event). The corrected version shifts
alarms within the window's true absolute bounds
(`window_start_idx + ((a - window_start_idx + s) % n_months)`, per
segment boundaries in `real_data_date_boundaries.csv`) rather than an
arbitrary origin disconnected from where the window actually sits.
With that fix: total hits across methods = 7 against a null with mean
2.52, SD 1.85 over 20,000 draws, p ≈ 0.029, still short of a further
Bonferroni step across the four series and not changing §9's
conclusion.

Extending this to GDP, GS10, and UNRATE
(`experiments/exp13d_export_other_series.py`, which reuses
`real_data_eval.py`'s own `monitored_months()` and event-filtering
directly so each series' inputs match Table 6's own denominators
exactly, and `experiments/exp13d_all_series_circular_shift.py`, which
runs the same test on all four) found a second, different bug in the
same family: GDP's `n_monitor` (from `real_data.py`'s `SERIES` config)
counts *quarterly* observations, since GDPC1 is quarterly, but an
initial version used it directly as a month-count -- 12 segments x 20
"months" (actually quarters) treated as 240 months, when the true
monitored window (1962-04 to 2022-01, per
`real_data_date_boundaries.csv`) is 718 months wide. This silently
truncated GDP's window by roughly 3x, the same failure mode as the
epoch bug above (real events/alarms falling outside the range any
shift could reach) via a different mechanism (a units mismatch instead
of an anchor mismatch). Fixed by deriving `n_months` from the window's
actual start and end dates directly, rounded up to a multiple of the
series' observation step (3 for quarterly GDP, 1 for the three monthly
series). Because two different bugs in this same family have now
escaped notice once each, `exp13c_circular_shift.py` also gained a
`_validate_window` check, called at the start of every test run, that
raises an error if any real event or alarm index falls outside the
stated window -- a structural guard against a third occurrence, not
just a fix for the two already found.

With both fixes and exact (exhaustive, not Monte Carlo) enumeration
throughout: INDPRO p = 0.028, GS10 p = 0.076 (neither survives);
GDP p = 0.0125, exactly equal to the Bonferroni threshold (0.05/4)
to machine precision -- a tie, not a survival, and half of its six
hits trace to one synchronized cross-method co-firing (§9); UNRATE
p = 0.0115, nominally below threshold, but 9 of its 14 hits (64%)
sit in the same φ-clipped windows already flagged in this section's
UNRATE discussion, cross-referenced directly against
`experiments/exp09_real_data_fit_check.py`'s per-segment fitted φ
via the `segment` column already present in `rd_unrate_alarms.csv`.
Neither nominal "survival" holds up as independent evidence once
traced to its source; §9's conclusion is unchanged, now as a checked
result across all four series rather than an INDPRO-only estimate.
A sixth addition, `exp14`
(`experiments/exp14_mixed_channel.py`), tests §10's original practical
recommendation to run both a raw and a whitened variance CUSUM under
real channel uncertainty: a 50/50 population of r- and q-channel
breaks with the channel unknown to the detector, both statistics
jointly recalibrated to hold a common 5% FAR (not run independently at
their own 5% each). The recommendation as originally stated did not
hold up — running both loses to the single better detector at every
SNR tested, with the gap widening from 0.06 (SNR 0.1) to 0.166 (SNR
2.0) — and §10's practical-recipe bullet has been revised accordingly.
A seventh addition, `exp15` (`experiments/exp15_garch_benchmark.py`,
`experiments/garch_detector.py`), fits a GARCH(1,1) on the training
prefix only (via the `arch` package), causally forward-filters
conditional variance over the full series with the fixed fitted
parameters, and runs the same three-arm max-CUSUM used for the raw and
ARIMA rungs on the standardized residuals — reported in Related Work
rather than left deferred. Calibrated at n_reps = 500 (empirical FAR
0.050 confirmed in every cell), GARCH sits at the false-alarm floor on
both variance channels at every SNR tested, contributing nothing over
chance on this DGP — a materially stronger negative result than a
prior placeholder estimate had suggested, and reported at its measured
value rather than softened. An eighth addition, `exp16`
(`experiments/exp16_aic_order_frequencies.py`,
`paper_assets/exp16_aic_order_frequencies.csv`), quantifies Appendix B's
near-unit-root AIC order-selection claim across the same 12 (φ, SNR)
cells as `grid_v6_phisweep`, 500 replicates each, tallying
`lsc.benchmarks.arima.fit_arima_prefix`'s own order choice rather than
a separate re-derivation — see Appendix B for the resulting
frequencies. A ninth, `exp17`
(`experiments/exp17_unrate_phi_gated.py`,
`paper_assets/exp17_unrate_phi_gated.csv`), turns the UNRATE
model-fit caveat (§9) into a direct test: excluding the φ-clipped
windows from both the hit count and the resampling universe (not just
the hit count, which would compare a restricted numerator against an
unrestricted denominator) drops raw_cusum and lsc_kalman_cusum from
4/9 to 1/9 hits each and collapses their significance to p = 0.1474 —
see §9. 98 tests include
bit-identical no-lookahead checks for every feature and detector
(including the raw and ARIMA variance rungs, the two windowed
statistics, and a training-freeze check), DGP ground-truth checks
(including the state-innovation break's SD scaling and null-match),
the ARMA(1,1)/Riccati identity and Kalman-equivalence guards, calibration-
parity checks, composite golden-score regression guards (which pin the
per-time-point-standardized output so a stale artifact cannot recur),
and the windowed-statistic bounded-memory (drain-after-permanent-shift)
property. All post-hoc design changes and pre-registered hypotheses
(including the three falsified ones and the two rejected robust-feature
designs — one falsified, one diluted, §8.3) are in
`experiments/CHANGELOG.md`.

**On "three pre-registered hypotheses falsified" (Contribution 1).**
This refers specifically to the exp05 robust-variance-feature
registration (`experiments/CHANGELOG.md`, 2026-07-11, "registered
before running: robust variance features (exp05)..."): three named
predictions — (a) under t₅ noise, `composite_robust` recovers variance
×1.5 detection to ≥0.5 (from the standard composite's 0.16); (b) under
Gaussian noise the robustness tax on variance scenarios is ≤~10pp; (c)
level and persistence rows are roughly unchanged — all three logged
FALSIFIED in the same-day outcome entry. CHANGELOG documents
pre-registration as a repeated practice through the project, not
confined to this one instance, and other pre-registered predictions
were also falsified elsewhere: both of exp03's registered predictions
(2026-07-10), exp03b's quietness-detection hypothesis (falsified in
its original run and again after a standardization fix, 2026-07-10),
and Outcome A of the varbench claim-adoption decision rule (M0, three
outcomes A/B/C registered before any grid_v4 cell ran; A falsified, C
fired, 2026-07-11/07-12) — so "three" names this specific registration,
not a project-wide total of every falsified pre-registered prediction.
The real-data extension (m6x, 2026-07-11) is a different category:
GDP and GS10 were "chosen for the method's signature cases before
looking at their results," and the evaluation protocol was likewise
fixed in advance, but CHANGELOG frames this explicitly as registering
a design rather than a falsifiable hypothesis ("this entry registers
the design, not power hypotheses") — committed in advance, but not
scored against a stated prediction. Beyond these named cases,
CHANGELOG does not tag each of Appendix C's individual rows as
pre-registered or identified post hoc; absent an explicit
"registered/pre-registered before running" note for a specific claim,
no such status is asserted here, and no complete per-row taxonomy is
attempted.

Full experiment narratives in
`experiments/FINDINGS.md`; theory statements and proofs in Appendix B,
with `experiments/THEORY.md` as the long-form companion. The complete source, pinned data snapshots, and seed
configuration constitute the replication package, to be posted publicly on
acceptance and available from the author on request in the interim.
`paper_assets/reproducibility_check_2026-07-22.txt` is a standing,
checkable record (not a one-time claim) of a fresh `git clone` into a
scratch directory, a fresh virtual environment installed only via the
documented `pyproject.toml` setup, the full test suite, and the two
most recently added benchmark scripts (`exp14`, `exp15`) run in full
from that clean environment -- reproducing the test count and every
cited number in this section bit-for-bit.
`paper_assets/reproducibility_check_2026-07-23.txt` repeats this for
the four-series circular-shift extension above: a separate fresh
clone reproduces the export step's window-bounds validation and the
exact enumeration results for all four series bit-for-bit, including
GDP's exact tie with the Bonferroni threshold.

## Appendix B. Theory: statements and proofs

This appendix gives self-contained statements and proofs of
Propositions 1–2; `experiments/THEORY.md` remains the long-form
companion, and the numerical verification is in
`experiments/exp06_theory_check.py` (1000 replications, §4).

**Setup and assumptions.** Scalar state-space model with *known*
parameters, filter in steady state, Gaussian innovations: S_t =
φS_{t−1} + w_t with w_t ~ N(0, q) and |φ| < 1, and Y_t = S_t + v_t with
v_t ~ N(0, r). The steady-state prediction variance P solves the Riccati fixed point
P = φ²Pr/(P+r) + q; the gain is K = P/(P+r) and the innovation variance
F = P + r. Standardized one-step innovations e_t = (Y_t − φŜ_{t−1})/√F
are iid N(0,1) under the null. A **level break** adds δ to the state
path from t₀ on: S̃_t = S_t + δ·1{t ≥ t₀}, hence Ỹ_t = Y_t +
δ·1{t ≥ t₀} — the DGP used in all experiments. The one-sided Page CUSUM
with drift allowance k and threshold h is g_t = max(0, g_{t−1} + e_t −
k), alarming when g_t ≥ h. (The experiments use fitted training-prefix
parameters, diffuse initialization, and a two-sided CUSUM; §4 explains
why the known-parameter theory nevertheless describes them to first
order.)

> **Proposition 1 (fast-or-never; restated from §4).** After a state
> level shift δ at t₀: (a) the broken path's standardized innovations
> decompose as ẽ_t = e_t + μ_t with e_t the null innovations and μ_t
> deterministic, decaying geometrically at rate ρ = φ(1−K) ∈ (0,1) from
> μ_{t₀} = δ/√F to μ∞ = δ(1−φ)/((1−φ(1−K))√F); (b) if μ_t ≤ μ̃ < k for
> all t ≥ t₁ ≥ t₀ (post-transient) and g_{t₁} = g < h, then for any
> horizon L,
> P( max_{t₁ < t ≤ t₁+L} g_t ≥ h | g_{t₁} = g ) ≤
> (L+1)·exp(−2(k−μ̃)(h−g)).

*Proof of (a).* The steady-state filter is a linear time-invariant map
of Y, so the innovations of the broken path decompose as the null
innovations plus the deterministic innovation response μ_t to the input
δ·1{t ≥ t₀}. Write a_j for the filter's mean state-estimate response
j steps after the break, a_j = E[Ŝ_{t₀+j}] − E[Ŝ⁰_{t₀+j}]. The filter
predicts φa_{j−1} and corrects by K times the mean innovation:
a_j = φa_{j−1} + K(δ − φa_{j−1}) = ρ a_{j−1} + Kδ with a_{−1} = 0 and
ρ = φ(1−K); the mean innovation response is the input minus the
prediction response, μ_{t₀+j} = (δ − φa_{j−1})/√F. Solving the linear
recursion, a_j = (Kδ/(1−ρ))(1 − ρ^{j+1}), so μ decays geometrically at
rate ρ from δ/√F to the limit (δ − φKδ/(1−ρ))/√F =
δ(1−φ)/((1−φ(1−K))√F) = μ∞. ∎

*Proof of (b).* For n > t₁, g_n = max( g + Σ_{i=t₁+1}^n X_i ,
max_{t₁<m≤n} Σ_{i=m}^n X_i ) with increments X_i = e_i + μ_i − k =
z_i − (k − μ_i), z_i iid N(0,1). An alarm by t₁+L requires some anchored
sum Σ_{i=m}^n X_i ≥ h − g for an anchor m ∈ (t₁, t₁+L] (or the
g-anchored sum ≥ h − g). Each X_i is stochastically dominated by
z_i − (k − μ̃), for which θ\* = 2(k − μ̃) solves E[e^{θX}] = 1; the
exponential martingale e^{θ\*Σ} with the maximal inequality gives
P( sup_n Σ_{i=m}^n X_i ≥ h − g ) ≤ e^{−θ\*(h−g)} for each of the ≤ L+1
anchor points, and a union bound finishes. ∎

Interpretation: the filter adapts, so of the full shift δ only the
fraction (1−φ)/(1−φ(1−K)) survives in the innovations per step. The
transient carries total excess mass Σ_j (μ_{t₀+j} − μ∞) =
φa∞/((1−ρ)√F), which is what a "fast" detection consumes; if the alarm
does not fire on the transient, part (b) says it fires later with
probability exponentially small in the threshold — fast or never.

> **Proposition 2 (raw-CUSUM delay: Wald approximation; restated from
> §4).** The raw-Y CUSUM standardizes Y by its frozen training moments;
> after the break the standardized mean shift Δ = δ/σ_Y (σ_Y² =
> q/(1−φ²) + r) persists for all t ≥ t₀. If Δ > k the post-break
> increments have positive drift Δ − k and the alarm is certain as the
> horizon grows, with first-passage (Wald) mean delay E[D] ≈ h/(Δ−k);
> if Δ ≤ k, the bound of Proposition 1(b) applies with μ̃ = Δ.

*Derivation.* Unlike the innovations, the raw standardized series has no
adaptation: the shift δ enters Y permanently and the training moments
are frozen, so every post-break increment is z_i + Δ − k with drift
Δ − k > 0. The CUSUM then behaves as a positive-drift random walk, and
Wald's identity for the first passage of level h − g gives E[D] ≈
(h − g)/(Δ − k) ≈ h/(Δ − k) (Wald 1947). The approximation ignores
boundary overshoot and reflection at 0, so it is an approximation, not
a bound (a corrected version is in Siegmund 1985); against the grids it
runs ≈15–20% conservative at 3σ (§4). If Δ ≤ k the drift is
nonpositive and the never-detect bound applies verbatim. ∎

**ARMA(1,1) equivalence of the whitened rungs (§5).** Applying the AR
operator to the observable, (1 − φL)Y_t = w_t + v_t − φv_{t−1} =: u_t,
which is an MA(1) with autocovariances γ_u(0) = q + r(1 + φ²), γ_u(1) =
−φr, γ_u(h) = 0 (h ≥ 2). Matching u_t to (1 − θL)ε_t (variance σ_ε²)
gives the invertible root θ = (m − √(m²−4))/2 with m = (q + r(1+φ²))/(φr),
and σ_ε² = φr/θ. Two identities close the loop with Proposition 1: σ_ε² =
F and θ = ρ = φ(1 − K). The second follows from the innovation recursion
ν_t = Y_t − φY_{t−1} + φ(1−K)ν_{t−1} (substitute Ŝ_{t−1} = φŜ_{t−2} +
Kν_{t−1} into ν_t = Y_t − φŜ_{t−1}), which is the ARMA(1,1) innovation
recursion once θ = φ(1−K). Hence the steady-state Kalman innovations and
the ARMA(1,1) innovations are the same series; the state layer is not a
distinct information set from ARIMA whitening on this DGP. Both identities
hold to < 10⁻¹² (`lsc.theory.arma11_representation`,
`test_arma11_riccati_identities`), and the numerical equivalence on null
paths — with true and with estimated parameters — is in
`experiments/exp07_arma_equivalence.py` (§5). A near-unit-root caveat: AIC
over the benchmark order grid rarely selects the exact (1,0,1) at φ = 0.95
(it prefers (1,0,0) at low SNR, the differencing (0,1,1) at higher SNR),
but these approximate the ARMA(1,1) closely enough that the median
innovation correlation with the Kalman filter stays at 0.99; forcing
(1,0,1) restores 0.9995. This is now measured directly rather than
asserted from spot checks: `experiments/exp16_aic_order_frequencies.py`
tallies `lsc.benchmarks.arima.fit_arima_prefix`'s own order choice
(the identical call that fits every ARIMA rung elsewhere in the paper)
over 500 null replicates at each of the same 12 (φ, SNR) cells as
`grid_v6_phisweep`. At φ = 0.95, (1,0,1) is selected 12.0% (SNR 0.1),
9.4% (SNR 0.5), and 7.8% (SNR 2.0) of the time; (1,0,0) dominates at
SNR 0.1 (43.0%) and (0,1,1) dominates at SNR 0.5–2.0 (64.8%, 68.6%),
exactly the qualitative pattern above, now with frequencies rather
than an impression. The pattern is not unique to φ = 0.95: at
φ = 0.99, (0,1,1) reaches 72.8% at SNR 2.0, and even at φ = 0.5 the
grid's nominal AIC-optimal choice, (1,0,1), is a minority pick (8.4–
18.4% across SNR) against (1,0,0)'s 61.6–63.0% — the finite-sample AIC
grid favors parsimony over exactness throughout, not only near the
unit root (`paper_assets/exp16_aic_order_frequencies.csv`, all 12
cells).

## Appendix C. Summary of key quantities

| Claim | Number | Source |
|---|---|---|
| Raw CUSUM level 3σ detect, all SNRs | 0.97–0.99 | grid_v1 |
| Innovation CUSUM delay vs raw, 3σ | 24–53 vs 58–91 obs | exp02 |
| Innovation CUSUM detect at 3σ | 0.55–0.67 | exp02/grid_v1 |
| μ∞ at 3σ, SNR 0.5 (knife-edge vs k=0.5) | 0.469 | exp06 |
| **r**-break ×1.5: latent composite, T=500 | 0.82 / 0.87 / 0.91 (SNR 0.1/0.5/2.0) | grid_v1 |
| **r**-break ×1.5: raw rung, T=500 | 1.00 / 0.56 / 0.10 (SNR 0.1/0.5/2.0) | grid_v4_varbench |
| **r**-break ×1.5: ARIMA rung, T=500 | 0.90 / 0.94 / 0.87 (SNR 0.1/0.5/2.0) | grid_v4_varbench |
| **r**-break ×1.5 t₅ (raw/ARIMA/composite), SNR 0.5 | 0.43 / 0.74 / 0.16 | grid_v4_varbench, v2_misspec |
| ARMA≡Kalman innovation ρ̄ (estimated / true params) | 0.99 / 1.000 (max\|Δ\|≈10⁻⁹) | exp07 |
| **q**-break ×1.5: raw rung, T=500 | 0.09 / 0.21 / 0.23 (SNR 0.1/0.5/2.0) | grid_v5_qbreak |
| **q**-break ×1.5: ARIMA rung, T=500 | 0.03 / 0.10 / 0.16 (SNR 0.1/0.5/2.0) | grid_v5_qbreak |
| Pre-registered decision rule (§5) resolved | Outcome B2 (r-channel-specific) | §5, CHANGELOG |
| φ sweep: μ∞ sorts innovation-CUSUM detection | Spearman 0.9415, n=24 (4 φ × 3 SNR × 2 shifts); stratified permutation null (shuffle φ-pairing within SNR×shift, n_perm=20,000) rejects at p<0.00005 | grid_v6_phisweep, exp12 |
| Innovation CUSUM 3σ escape (φ=0.5 / 0.99) | 0.98–1.00 / 0.30–0.63 | grid_v6_phisweep |
| Local-level 3σ level detect (all methods) | ≤ 0.15 (≈ FAR) | grid_v7_llevel |
| Local-level ×1.5 variance (raw / ARIMA rung) | 0.06 / 0.58–0.84 | grid_v7_llevel |
| ARL₀ at 5% window-FAR (L=375) | ≈ 7300 obs | arl_table |
| φ×q: ×1.5 raw edge, φ-swept = SNR-swept | Δ 0.11=0.11 (SNR 0.5) | grid_v8 vs grid_v5 |
| φ×q: subtle Δ at φ→0 / Spearman(amp,Δ) | 0.00 / 0.83; ×3 falsified (−0.60) | grid_v8_phiqbreak |
| Variance ×1.5 across T = 200/500/2000 | 0.11 / 0.87 / 0.99 | grid_v2_T |
| t₅ collapse and repair (×1.5) | 0.16 → 0.75 (tail_cusum) | grid_v2_misspec, v3c |
| Quieting ×⅔ (only tail_cusum) | 0.41 / 0.33 | grid_v3c |
| Persistence-down, best anywhere | 0.33 (SNR 2.0) | grid_v1 |
| Multi-break: raw second-event recall (level→level) | 0.00 | exp04 |
| Composite level→var second event | 0.60 (F1 0.63) | exp04 |
| Windowed-CUSUM fix, level→level 2nd event (raw / innovation) | 0.00→0.68 / 0.01→0.23 | exp04 |
| Windowed-CUSUM fix, level→var / var→var 2nd event | no improvement (≈0.00): mean-shift only | exp04 |
| PELT localization at FAR-matched 5%, level 3σ | 0.83–0.92 (vs. causal raw CUSUM 0.97–0.99) | exp08_pelt |
| PELT localization at FAR-matched 5%, variance ×1.5/×3 | 0.00–0.20 (vs. dedicated raw variance-CUSUM 0.10–1.00) | exp08_pelt |
| INDPRO permutation p (composite) | 0.008 (uncorrected — does not survive Bonferroni/BH-FDR across the 19 tests in Table 6; §9) | rd_eval |
| Circular-shift joint test, INDPRO (5 methods, total hits) | 7 vs. null mean 2.52, max 10 (780 shifts, exact) — p=0.028, does not survive Bonferroni (α/4=0.0125); §9 | exp13c_circular_shift |
| Circular-shift joint test, GDP | 6 vs. null mean 1.50, max 6 (240 shifts, exact) — p=0.0125, exact tie with the threshold, half the hits from one synchronized co-firing; §9 | exp13d_all_series_circular_shift |
| Circular-shift joint test, GS10 | 4 vs. null mean 1.30, max 6 (720 shifts, exact) — p=0.076, does not survive; §9 | exp13d_all_series_circular_shift |
| Circular-shift joint test, UNRATE | 14 vs. null mean 3.57, max 16 (780 shifts, exact) — p=0.0115, nominally survives, but 64% of hits (9/14) sit in the same φ-clipped windows already flagged in §9; §9 | exp13d_all_series_circular_shift |
| GARCH(1,1) benchmark vs. raw/ARIMA rungs, ×1.5, SNR 0.5/2.0 (n_reps=500) | r-channel: 0.098/0.096 (GARCH) vs. 0.56/0.10 (raw), 0.94/0.87 (ARIMA); q-channel: 0.066/0.098 (GARCH) vs. 0.21/0.23 (raw), 0.10/0.16 (ARIMA) — GARCH at the FAR floor throughout | exp15_garch_benchmark |
| AIC order-selection frequency at φ=0.95 (SNR 0.1/0.5/2.0), n=500/cell | (1,0,1): 12.0%/9.4%/7.8%; (1,0,0) dominant at SNR 0.1 (43.0%), (0,1,1) dominant at SNR 0.5-2.0 (64.8%/68.6%) | exp16_aic_order_frequencies |
| UNRATE φ-gated permutation test (raw_cusum, lsc_kalman_cusum) | 4/9→1/9 hits after excluding clipped-φ windows from both numerator and resampling universe (540/780 months); p=0.1474 (both), vs. ungated 0.0002-0.0004; §9 | exp17_unrate_phi_gated |
| Mixed-channel (raw+ARIMA run jointly, unknown channel), SNR 0.1/0.5/2.0 | combined loses to single-better detector at every SNR: 0.493 vs 0.553 / 0.490 vs 0.560 / 0.457 vs 0.623 | exp14_mixed_channel |
| GFC real-time | 2008-09 data, known 2008-12 | rd_realtime |
| COVID real-time | data 2020-03, ~2 mo before NBER | rd_realtime |
