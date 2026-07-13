# When Does Filtering Help You See a Break? Latent-State Diagnostics for Structural Change at Calibrated False-Alarm Rates

## Abstract

A two-layer framework for detecting hidden structural change: a
state-estimation layer (Kalman filtering of a latent state from noisy
observations) and a diagnostics layer that converts features of the
strictly-causal filtered path into alarms. All detectors — ours and
benchmarks — are calibrated on matched null data-generating processes to
the same false-alarm rate (5% per 500 observations), making detection
rates and delays directly comparable. The main finding is a division of
labor, not a victory: for level shifts, a CUSUM on the raw data dominates
detection rates at every signal-to-noise ratio, while the latent-innovation
CUSUM is "fast or never" — a phenomenon we formalize (the post-break
innovation mean decays geometrically to δ(1−φ)/((1−φ(1−K))√F); when this
is below the CUSUM allowance, post-transient detection has an
exponentially small bound) and verify numerically — sweeping the state
persistence φ, we find the asymptotic drift μ∞ sorts detection
(Spearman 0.94) and the innovation CUSUM *escapes* the fast-or-never
regime at low φ, so the negative result is a boundary condition of
persistent states (the empirically relevant case), not a universal law.
For second-moment changes the story is a **two-channel** one. On
observation-noise breaks a whitening ladder shows the edge over raw data
is *prewhitening under autocorrelation*, not the latent state estimate:
the ARIMA-residual and Kalman-innovation rungs are provably the *same
filter* (the observable is exactly ARMA(1,1), so the two whitened rungs
coincide up to estimation error, ρ̄ = 0.99), and both detect the subtle
(×1.5) break at every SNR (ARIMA 0.90/0.94/0.87) while a raw-data
variance CUSUM detects it only when observation noise dominates (0.996 at
SNR 0.1) and falls to chance as the latent signal grows (0.56, 0.10 at
SNR 0.5, 2.0). But this advantage is specific to the observation-noise
channel: on state-innovation (shock-variance) breaks — the channel that
describes the Great Moderation and crisis-volatility events the
application targets — the ordering *inverts*, and a raw variance CUSUM
matches or beats the whitened rungs at every SNR (×1.5: raw
0.09/0.21/0.23 vs ARIMA 0.03/0.10/0.16). Prewhitening reveals a break in
the white component and *removes* the signal of a break in the state,
which is why on real crises a raw variance CUSUM's timing is
indistinguishable from the state-aware composite's. The diagnostics further
survive heavy tails via an exceedance-indicator variant and, via a
shortfall CUSUM, detect variance *quieting* that level-oriented methods
miss. A real-data application
(industrial production, GDP, Treasury yields) reproduces the profile:
every alarm attributes to a second-moment feature, association with NBER
reference dates is significant (permutation p = 0.007), and real-time
(ALFRED vintage) analysis confirms the COVID timing while honestly
downgrading the 2008 timing claim.

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
  scores at earlier times unchanged, exactly).
- Contribution 2 (negative + positive results): the intuition is wrong
  for first moments at high persistence — and we can prove why
  (fast-or-never theorem), then bound the claim by sweeping φ (μ∞ sorts
  detection, and the innovation CUSUM escapes the regime at low φ). For
  second moments the answer is a two-rung, two-channel decomposition. The
  ladder collapses to *two* rungs because the ARIMA and Kalman rungs are
  the same filter — the observable is exactly ARMA(1,1), an equivalence
  we state as theory and confirm to machine precision — so "prewhitening"
  and "state estimation" are not separable here; the state layer adds
  nothing the ARIMA residuals do not. And the prewhitening advantage is
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
- Honest-outcome framing throughout: three pre-registered hypotheses
  were falsified; every post-hoc change is logged (CHANGELOG) and the
  failures are reported as findings.

**Related work.** The paper sits at the intersection of several
literatures. Sequential change detection descends from Page's (1954)
CUSUM and the quickest-detection tradition (Lorden 1971; Moustakides
1986); we use the CUSUM as the common statistic across information sets
rather than proposing a new stopping rule. The idea of monitoring a
*state-space* model through its innovations is not ours: it is the core
of the innovation-based change-detection literature — the generalized
likelihood ratio test on Kalman innovations (Willsky & Jones 1976) and
the systematic treatment of CUSUM/GLR schemes on innovation sequences in
Basseville & Nikiforov (1993). That literature also supplies the
evaluation currency we adopt: the average run length (ARL), with methods
compared at a matched in-control ARL₀ — the statistical-process-control
convention (Page 1954; Montgomery). Our contribution on this axis is
*not* the innovation CUSUM or ARL-matching per se, both of which are
standard in SPC and quickest detection; it is that this matched-error
protocol is essentially *absent* from the applied latent-state
econometrics literature, and that transplanting it there — calibrating
every information set to a common false-alarm rate — is precisely what
produces the negative results (raw data wins on levels; the state
estimate adds nothing over ARIMA whitening on variances). In econometrics,
sequential monitoring of structural change is the CUSUM-of-recursive-
residuals line and its modern monitoring form (Brown, Durbin & Evans
1975; Chu, Stinchcombe & White 1996), which watch *observable* regression
residuals; our object is a *latent* state, and the residuals we monitor
are a filter's innovations. Regime-switching models (Hamilton 1989;
Kim–Nelson 1999) offer a state-aware alternative whose regime
probabilities we include as a benchmark and find saturate under
calibration on nonstationary data. The empirical target of the
second-moment results is the Great Moderation volatility decline
(McConnell–Perez-Quiros 2000; Stock–Watson 2002); as we show, that and
the crisis-volatility events are *state-innovation* (shock-variance)
breaks, the channel on which raw and prewhitened detectors are
interchangeable. Finally, offline changepoint methods (PELT, Killick et
al. 2012) solve a retrospective segmentation problem; our monitoring is
strictly causal and calibrated to a false-alarm rate. What is new here is
not any single detector but the calibrated-parity harness that makes
latent-state and raw-data detectors directly comparable, and a
reduced-form result (the exact ARMA(1,1) equivalence of the ARIMA and
Kalman rungs) that dissolves the "does filtering help?" question for
second moments into a prewhitening question with a known answer.

## 2. Framework and evaluation protocol

**Model layer.** S_t = φS_{t−1} + w_t (var q), Y_t = S_t + v_t (var r);
the estimator sees only Y and fits (φ, q, r) by maximum likelihood on a
training prefix (25% of the sample), then runs a forward-only filter with
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
at 5900–7600 (raw CUSUM calibrates slightly hot). The out-of-control
**ARL₁** is the post-break detection delay (mean delay conditional on
detection, reported beside the detection rate since misses are censored):
e.g. at 3σ level, SNR 0.5, ARL₁ ≈ 77–86 observations. Calibrating a
common ARL₀ is exactly the ARL-matching convention of SPC (Basseville &
Nikiforov 1993); we express it as a window-FAR because our horizon is
finite.

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
parameter). Also: level shifts (0.5, 1, 3 σ_ref), logistic ramps, and
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
from two facts about the detectors' post-break drift. Derivations are in
Appendix B (`experiments/THEORY.md`).

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

> **Proposition 2 (raw-CUSUM delay).** The raw-Y CUSUM sees the full shift
> as a sustained standardized drift Δ = δ/σ_Y, giving an Albert–Wald mean
> delay of approximately h/(Δ−k) once Δ > k, and negligible power when
> Δ ≤ k.

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
persistent. Sweeping φ ∈ {0.5, 0.8, 0.95, 0.99} at fixed SNR (Figure
`grid_v6_muinf_scatter.png`) confirms it and turns the theory into a
falsifiable ordering: μ∞ sorts the innovation-CUSUM detection rate across
all cells (Spearman 0.94), fast-regime cells (μ∞ ≥ k = 0.5) detect
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

**Assumptions and estimation error.** Propositions 1–2 assume the
steady-state filter with *known* parameters. Two facts keep this from
being a limitation in practice. First, the filter reaches its steady
state within the 125-observation training prefix, well before monitoring
begins (the innovation autocorrelation is flat by then). Second, the
error from estimating (φ, q, r) rather than knowing them is second-order:
in the M1 equivalence check (§5, `experiments/exp07`) the ARIMA and
Kalman standardized-innovation series computed with *estimated*
parameters correlate at ρ̄ = 0.99 with each other and, computed with
*true* parameters, at ρ = 1.000 to a max discrepancy of ≈10⁻⁹ — so the
entire estimated-vs-true gap is the small residual that leaves the
median correlation at 0.99. The theory describes the estimated filter to
within that gap.

## 5. Second moments: a whitening ladder

Is the latent layer's second-moment advantage about the *state estimate*,
or merely about *prewhitening* the autocorrelated observations before
applying a variance statistic? We answer this directly by running the
identical variance CUSUM (up-arm Page CUSUMs of z²−1 at allowances k =
0.25 and 0.05, a down-arm CUSUM of 1−z², max over arms, no per-time
standardization) at three levels of prewhitening, calibrated by the same
routine on the same matched-null seed blocks:

- **raw** — z from the raw observations, standardized by frozen
  training-prefix moments;
- **ARIMA** — the same statistic on the standardized one-step residuals
  of an AIC-selected, training-prefix-frozen ARIMA model (whitened, but
  not state-aware);
- **latent** — the composite's e²-based variance features on the Kalman
  innovations (state-aware).

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

**The ladder, both channels (detection rate at T = 500, 5% calibrated
FAR; MC SEs ≤ 0.02 in `paper_assets/ladder_table.csv`, `break_channel`
column).**

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
(0.90 / 0.94 / 0.87) and tracks the *latent* rung step for step (0.82 /
0.87 / 0.91) — as it must, since the equivalence above makes them the
same filter. On this channel the advantage over raw is *prewhitening under
autocorrelation*, decisive where the latent signal masks the noise change.

*State-innovation (q) breaks — raw wins, and the SNR-dependence flips.*
Here the raw rung's dependence on SNR *reverses sign* — it *rises* with
SNR (0.09 → 0.21 → 0.23 at ×1.5) — and the raw rung matches or beats the
whitened rungs at every SNR (×1.5 ARIMA 0.03 / 0.10 / 0.16; ×3 raw 0.72 /
0.96 / 0.96 vs ARIMA 0.26 / 0.79 / 1.00, the whitened rung catching up
only at the ×3 high-SNR ceiling). The mechanism is the mirror image: a
q-break inflates the *state's own* variance, which dominates the marginal
variance of Y at high SNR, so a raw z² statistic sees it directly —
whereas prewhitening *strips out* the state-carried signal along with the
autocorrelation. Prewhitening reveals a break in the white component and
removes a break in the state. Quieting (×⅔, i.e. reduced q) is
undetectable by every rung (≤ 0.07, at FAR): a low-q state contributes
too little to Y to register its own reduction.

This resolves the pre-registered decision rule (`experiments/CHANGELOG.md`)
as **Outcome B2**: the "prewhitening beats raw" result is *specific to
the observation-noise channel*. It matters because the events the
application targets — the Great Moderation, crisis volatility — are
state-innovation (shock-variance) breaks, not observation-noise breaks;
on that channel a raw variance CUSUM is at least as good as any amount of
whitening. This is not a weakness to hide but the explanation for a
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
anywhere in the grid) and 0.17 at T = 2000. Scale-quieting (×⅔) is
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
chatter, binds. Bounded-memory (windowed) statistics are the identified
fix and are left to future work.

## 8. Robustness

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
(`configs/grid_v7_llevel.yaml`, Figure `grid_v7_llevel_degeneracy.png`),
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
0.97–1.00 while raw stays at chance. This is the exact complement of §5's
r-channel AR(1) result — where a raw variance CUSUM could *win* when
observation noise dominates — and it sharpens the paper's thesis:
prewhitening is not merely helpful but *mandatory* once the observable is
nonstationary, because the raw statistic has no stationary baseline to
calibrate against.

**Protocol lessons (each cost us a wrong result before it was fixed).**
plain-HMM regime probabilities saturate and cannot be FAR-calibrated on
nonstationary data; probability-scale scores need log-odds; EM needs
persistent-initialization restarts; composite features must be
standardized per-time-point, not pooled; order-statistic thresholds have
Beta(n+1−k, k) noise regardless of distribution, so heavy-tailed
detectors need larger calibration budgets.

## 9. Real data (illustrative)

Three FRED series, pinned snapshots (2026-07-11), rolling causal
monitoring (train 120 months / monitor 60), per-segment parametric
bootstrap calibration at 5% FAR per window, alarms attributed to the
feature that crossed.

**Industrial production (INDPRO, 1948–2026).** Composite alarms: 2008-09
and 2020-04 (both variance_pressure), 1990-12 (variance_quiet), 1969-08
(variance_quiet; within the false-alarm budget and reported as such).
Hits 3/9 NBER peaks within 12 months, 1 stray vs 0.7 expected;
permutation p = 0.007 (innovation CUSUM p = 0.018; raw CUSUM 1 hit,
p = 0.15). The **raw variance CUSUM** — the bottom rung of the ladder,
added here to test real-data uniqueness — does catch the GFC (2008-09,
up-arm, the same month as the composite) but embeds it among four stray
quieting alarms (1967, 1968, 1988, 2019) and misses COVID, so its
NBER association is not significant (5 alarms, 1 hit, p = 0.56). The
crisis *is* detectable by a raw variance statistic; what the latent layer
buys on real data is a *clean* alarm profile — a significant, low-stray
association — not the crisis catch itself.

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
ZLB event is missed within 12 months. The raw variance CUSUM also flags
Volcker (1980-02, up-arm) but is the noisiest detector on this series
(9 alarms, 8 stray, p = 0.28) — on a series that is *all* volatility
regime, a raw z² statistic fires on every regime shift, which is exactly
why its event-association washes out.

**Sensitivity.** FAR = 10% behaves as expected. Training window 180
months breaks both variance-based detectors' bootstrap calibration on
nonstationary real data — the composite and the raw variance CUSUM each
fire 14 alarms / 21 windows — while the level (raw CUSUM, 2), innovation
(3), and tail (5) detectors stay sane. It is the *second-moment* statistic,
not the composite machinery, that is sensitive to a training window long
enough to straddle a volatility regime; training windows must be short
enough to be locally stationary (120 months worked).

## 10. Discussion

Four points summarize what the harness bought us.

- The latent-state diagnostics layer is not a better level-shift
  detector; it is a *different instrument*, reading second moments and
  dynamics. But the whitening ladder (§5) sharpens the claim past the
  point where "the state estimate" survives as an ingredient: on this DGP
  the Kalman innovations *are* the ARMA(1,1) innovations (an exact
  reduced-form equivalence, verified to machine precision), so whatever
  second-moment power the latent layer has is prewhitening, full stop —
  ARIMA residuals capture it identically. What the state framing still
  buys is dynamics features (persistence, quieting) and the fast-or-never
  speed edge on levels. The division of labor is now measured (grids),
  reduced (raw vs. whitened, the two whitened rungs proven identical), and
  derived (fast-or-never with its φ boundary).
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
  knows the channel, running *both* a raw and a whitened variance CUSUM
  (cheap, and they agree on the q channel) is the robust default. Use the
  exceedance-indicator variant under heavy tails and the composite for
  breadth. Calibrate everything on matched nulls at a common FAR (a common
  ARL₀) and report empirical FARs.
- The calibrated-parity protocol itself is a contribution: it exposed
  every failure mode above, and it is what makes the negative results
  informative rather than anecdotal.
- Limitations / future work: bounded-memory statistics for multiple
  breaks; adaptive composite weighting (breadth tax); switching-SSM (Kim
  filter) model layer; formalizing the persistence-break mechanisms; a
  vol-regime reference set for scoring the exceedance detector on real
  data.

---

## Appendix A. Reproducibility

`make all` regenerates every table and figure from pinned seeds
(Python 3.14, statsmodels/hmmlearn; `make fred` / `make realdata` /
`make realtime` for the data applications, snapshots under `data/`). The
referee-hardening round adds five reproducible artifacts to the pack:
`exp07` (ARMA equivalence, M1), `grid_v5` (the q-break channel, M2),
`grid_v6` (the φ sweep, M3), `grid_v7` (the local-level arena, M4), and
`arl` (ARL₀/ARL₁ table, M5); all are pinned-seed and join the existing
grids draw-for-draw. 95 tests include bit-identical no-lookahead checks
for every feature and detector (including the raw and ARIMA variance
rungs and a training-freeze check), DGP ground-truth checks (now
including the state-innovation break's SD scaling and null-match),
the ARMA(1,1)/Riccati identity and Kalman-equivalence guards, calibration-
parity checks, and composite golden-score regression guards (which pin the
per-time-point-standardized output so a stale artifact cannot recur). All
post-hoc design changes and pre-registered hypotheses (including the
three falsified ones and the two rejected robust-feature designs — one
falsified, one diluted, §8.3) are in `experiments/CHANGELOG.md`; full experiment narratives in
`experiments/FINDINGS.md`; theory derivations in `experiments/THEORY.md`
(Appendix B). The complete source, pinned data snapshots, and seed
configuration constitute the replication package, to be posted publicly on
acceptance and available from the author on request in the interim.

## Appendix B. Theory derivations

Full derivations of Proposition 1 (the geometric decay of the
standardized innovation mean to μ∞ and the resulting fast-or-never
alarm bound) and Proposition 2 (the Albert–Wald delay of the raw-Y CUSUM
under a sustained standardized drift) are in `experiments/THEORY.md`,
with the numerical verification in `experiments/exp06_theory_check.py`
(1000 replications). The steady-state Kalman gain K and innovation
variance F are the fixed-point solutions of the scalar Riccati recursion
for the AR(1) state model of §2; μ∞, ρ, and the bound follow by taking
the post-break innovation as a deterministic geometric transient plus
mean-zero noise and applying a one-sided Hoeffding bound to the CUSUM
increments.

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
(1,0,1) restores 0.9995.

## Appendix C. Summary of key quantities

| Claim | Number | Source |
|---|---|---|
| Raw CUSUM level 3σ detect, all SNRs | 0.97–0.99 | grid_v1 |
| Innovation CUSUM delay vs raw, 3σ | 24–53 vs 58–91 obs | exp02 |
| Innovation CUSUM detect at 3σ | 0.55–0.67 | exp02/grid_v1 |
| μ∞ at 3σ, SNR 0.5 (knife-edge vs k=0.5) | 0.469 | exp06 |
| Composite variance ×1.5, T=500 | 0.82 / 0.87 / 0.91 (SNR 0.1/0.5/2.0) | grid_v1 |
| Ladder ×1.5: raw rung, T=500 | 1.00 / 0.56 / 0.10 (SNR 0.1/0.5/2.0) | grid_v4_varbench |
| Ladder ×1.5: ARIMA rung, T=500 | 0.90 / 0.94 / 0.87 (SNR 0.1/0.5/2.0) | grid_v4_varbench |
| Ladder ×1.5 t₅ (raw/ARIMA/composite), SNR 0.5 | 0.43 / 0.74 / 0.16 | grid_v4_varbench, grid_v2_misspec |
| ARMA≡Kalman innovation ρ̄ (estimated / true params) | 0.99 / 1.000 (max\|Δ\|≈10⁻⁹) | exp07 (M1) |
| **q**-break ×1.5: raw rung, T=500 | 0.09 / 0.21 / 0.23 (SNR 0.1/0.5/2.0) | grid_v5_qbreak (M2) |
| **q**-break ×1.5: ARIMA rung, T=500 | 0.03 / 0.10 / 0.16 (SNR 0.1/0.5/2.0) | grid_v5_qbreak (M2) |
| Decision-rule outcome fired | r-channel provisional C, resolved **B2** (q inverts) | CHANGELOG (M2) |
| φ sweep: μ∞ sorts innovation-CUSUM detection | Spearman 0.94 | grid_v6_phisweep (M3) |
| Innovation CUSUM 3σ escape at low φ (φ=0.5 / 0.99) | 0.98–1.00 / 0.30–0.63 | grid_v6_phisweep (M3) |
| Local-level 3σ level detect (all methods) | ≤ 0.15 (≈ FAR) | grid_v7_llevel (M4) |
| Local-level ×1.5 variance (raw / ARIMA rung) | 0.06 / 0.58–0.84 | grid_v7_llevel (M4) |
| ARL₀ at 5% window-FAR (L=375) | ≈ 7300 obs | arl_table (M5) |
| Variance ×1.5 across T = 200/500/2000 | 0.11 / 0.87 / 0.99 | grid_v2_T |
| t₅ collapse and repair (×1.5) | 0.16 → 0.75 (tail_cusum) | grid_v2_misspec, grid_v3c |
| Quieting ×⅔ (only tail_cusum) | 0.41 / 0.33 | grid_v3c |
| Persistence-down, best anywhere | 0.33 (SNR 2.0) | grid_v1 |
| Multi-break: raw second-event recall | 0.00 | exp04 |
| Composite level→var second event | 0.60 (F1 0.63) | exp04 |
| INDPRO permutation p (composite) | 0.007 | rd_eval |
| GFC real-time | data month 2008-09, known 2008-12 | rd_realtime |
| COVID real-time | data 2020-03, ~2 mo before NBER | rd_realtime |
