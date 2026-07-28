# When Does Filtering Help You See a Break? Latent-State Diagnostics for Structural Change in Scalar Linear-Gaussian AR(1) State-Space Models at Calibrated False-Alarm Rates

## Abstract

We ask when filtering a latent state helps detect structural change,
calibrating every detector to the same false-alarm rate on matched
null data. For scalar linear-Gaussian AR(1) state-space models, the
answer is a trichotomy — *no, yes, no* — across three break types.
(i) Level shifts in a persistent state: no — a raw-data CUSUM
dominates at every SNR, and the innovation CUSUM's post-break drift
decays geometrically to an asymptote that sorts detection across a
persistence sweep (Spearman 0.94) — the μ∞ mechanism we prove
analytically, though the "fast or never" shape it motivates does not
hold cleanly at the paper's own flagship arena when tested directly.
(ii) Observation-noise variance changes: yes, but the advantage is
prewhitening, not the state estimate — at true parameters the
observable is exactly ARMA(1,1), so ARIMA and Kalman whitening are
provably the same filter; the equivalence does not survive
estimation, where a Kalman-MLE variance statistic reliably beats its
ARIMA-MLE counterpart. (iii) State-innovation (shock) variance
changes — the Great-Moderation/crisis-volatility channel: no, a
partial null — raw matches or beats whitening on coarse breaks, while
every rung sits near the false-alarm floor on subtle ones. On real
data (industrial production, GDP, unemployment, Treasury yields),
every alarm attributes to a second-moment feature, but the headline
NBER association does not survive multiple-testing correction. Two
further checks — an offline changepoint method and a multi-break
extension — probe the protocol's edges and are reported in the body.
Read together, the results are deflationary for the
latent layer's detection power — a raw or whitened single-feature
benchmark matches or beats the state-aware detector on nearly every
break type studied — with one exception: a multi-feature composite
built on the genuine filtered state decisively beats the same
composite built from ARIMA inputs, so filtering does buy real
detection power once parameters must be estimated (both for a single
statistic and in the multi-feature composite), just not at the
true-parameter single-statistic idealization the underlying identity
was proven for.

**JEL classification:** C12, C22, C52.
**MSC classification:** 62L10, 62M10, 93E11.
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

- Contribution 1 (protocol): a calibrated false-alarm-rate (FAR) parity
  harness. Every
  method's threshold comes from the same routine on the same matched-null
  draws; empirical FAR is re-verified on fresh nulls; causality is
  enforced structurally (parameters fit on a training prefix only;
  forward-only filtering; alarm scores are NaN on training data) and
  tested bit-identically (perturbing future observations leaves all
  scores at earlier times unchanged, exactly). The protocol extends to
  the honesty of the research process itself: one registration alone
  (exp05, three named sub-predictions) saw all three falsified
  (Appendix A), and pre-registered predictions were falsified at
  several other points across the project besides — every post-hoc
  change is logged
  (CHANGELOG) and the failures reported as findings rather than
  suppressed.
- Contribution 2 (negative + positive results): the intuition is wrong
  for first moments at high persistence — and we can prove why the
  innovation CUSUM's post-break drift decays to an asymptote μ∞ that
  sorts detection (Proposition 1(a), exact), then bound the claim by
  sweeping φ (the innovation CUSUM escapes the regime at low φ). A
  companion finite-horizon tail bound (Proposition 1(b)) also holds but
  is numerically vacuous at every 3σ cell audited (30/30; informative
  at smaller shift magnitudes), and the
  qualitative "fast or never" shape it motivates does not hold cleanly
  at the paper's own flagship arena either — so the theorem that
  survives scrutiny is the μ∞ mechanism, not a provable per-cell
  fast-or-never guarantee. For
  second moments the answer is a two-rung, two-channel decomposition. The
  ladder collapses to *two* rungs at true parameters, because the ARIMA
  and Kalman rungs are
  the same filter — the observable is exactly ARMA(1,1), a known identity
  from structural time-series theory (Harvey 1989; Hamilton 1994, ch. 13)
  that we confirm to machine precision here — so "prewhitening"
  and "state estimation" are not separable there; under estimation the
  collapse does not fully hold (a Kalman-MLE variance rung beats its
  ARIMA-MLE counterpart on the same single statistic). At true
  parameters, though, the state layer's
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
  the real-data claims — including a self-correction on the headline
  GFC (Global Financial Crisis, 2008) timing.

**Related work.** The paper sits at the intersection of several
literatures.

**Quickest detection and SPC (statistical process control).** Sequential change detection descends
from Page's (1954) CUSUM and the quickest-detection tradition (Lorden
1971; Moustakides 1986); we use the CUSUM as the common statistic across
information sets rather than proposing a new stopping rule. Aue &
Kirch (2024) survey seven decades of CUSUM-based sequential
changepoint testing specifically — the family of statistics this paper
uses throughout, on progressively more processed information sets
rather than a new member of that family — and Aue & Horváth (2013)
survey the broader structural-break-in-time-series literature this
paper's application (§9) sits inside, independent of any particular
test statistic. That
literature also supplies the evaluation currency we adopt: the average
run length (ARL), with methods compared at a matched in-control ARL₀ —
the statistical-process-control convention (Page 1954; Montgomery 2013).
The broader statistical-surveillance research program this evaluation
convention sits inside is surveyed in Frisén (2003), a complement to
the Basseville & Nikiforov (1993) treatment cited below for the
innovation-CUSUM axis specifically. A related but distinct
methodological tradition frames sequential monitoring through
always-valid, anytime p-values and e-processes rather than
ARL-calibrated thresholds (Ramdas, Grünwald, Vovk & Shafer 2023); our
calibrated-FAR harness is closer in spirit to the classical
SPC/quickest-detection convention above than to that game-theoretic
framing, but the two are natural peers, and translating the
calibrated-parity protocol into an e-value formulation is a direction
we have not pursued here.

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
(Brown, Durbin & Evans 1975) and its moving-window ("MOSUM") form (Chu,
Hornik & Kuan 1995), generalized into an online-monitoring framework by
Chu, Stinchcombe & White (1996) and Leisch, Hornik & Kuan (2000),
alongside the canonical unknown-breakpoint testing framework of Andrews
(1993), which watch *observable* regression residuals; our object is a
*latent* state, and the residuals we monitor are a filter's
innovations. The bounded-memory fix we give the multi-break re-arm
failure in §7 is in this same MOSUM/moving-estimates family (Chu,
Hornik & Kuan 1995; Leisch, Hornik & Kuan 2000): a moving two-window
mean-shift statistic, rather than a CUSUM accumulated against a fixed
training-prefix baseline.

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
stochastic-volatility state-space models (Kim, Shephard & Chib 1998;
Jacquier, Polson & Rossi 1994); its object is *estimation* of a
volatility process, not distribution-free monitoring at a calibrated
false-alarm rate, so a full stochastic-volatility comparison remains
future work. A GARCH(1,1) benchmark is reportable now: fit on the
training prefix, causally forward-filtered, and run through the same
three-arm max-CUSUM as the raw and ARIMA rungs on its standardized
residuals (`experiments/garch_detector.py`, `exp15_garch_benchmark.py`).
Calibrated at n = 500 across the full 2×2×3 grid (both variance
channels, subtle/coarse breaks, three SNRs — not just the four-cell
subtle-break subset originally checked), with calibration independently
confirmed via a disjoint fresh-null FAR check
(`exp24_garch_fresh_far_check.py`): 5.4%/5.0%/4.8% at SNR 0.1/0.5/2.0,
all within 0.6pp of target.

| channel | vol_mult | SNR 0.1 | SNR 0.5 | SNR 2.0 |
|---|---|---|---|---|
| r | ×1.5 | 0.498 | 0.098 | 0.096 |
| r | ×3 | 0.962 | 0.708 | 0.548 |
| q | ×1.5 | 0.038 | 0.066 | 0.098 |
| q | ×3 | 0.186 | 0.344 | 0.338 |

GARCH sits at the false-alarm floor only for the subtle ×1.5 break at
moderate-to-high SNR (as originally reported), but clears it
substantially at every SNR on the coarse ×3 break (r: 0.55–0.96; q:
0.19–0.34) and shows real power even at r-channel/SNR 0.1/subtle
(0.498) — so "GARCH contributes nothing over chance" holds only for
the four originally-checked cells, not the DGP in general. GARCH is
nonetheless dominated by raw and/or ARIMA in 11 of 12 cells (the
twelfth, r-channel/subtle/SNR 2.0, is tied with raw specifically —
ARIMA still dominates there — raw 0.102 vs. GARCH's 0.096, a gap well
inside the ≈1.3pp MC SE), never strictly the best of the three.

The floor result at subtle, moderate-SNR breaks reflects a generative
mismatch — GARCH(1,1) targets conditional heteroskedasticity, not this
paper's permanent step-change-in-variance DGP — rather than the
CUSUM-wrapper simply being underpowered: a follow-up
(`exp32_garch_mechanism.py`) correlates each rung's own
variance-tracking quantity against the true regime indicator (Spearman
ρ, AUC) on the five floor cells:

| channel | SNR | GARCH ρ | GARCH AUC | raw ρ | raw AUC | ARIMA ρ | ARIMA AUC |
|---|---|---|---|---|---|---|---|
| q | 0.1 | 0.036 | 0.522 | 0.026 | 0.516 | 0.022 | 0.514 |
| q | 0.5 | 0.104 | 0.564 | 0.085 | 0.552 | 0.042 | 0.526 |
| q | 2.0 | 0.190 | 0.616 | 0.141 | 0.587 | 0.071 | 0.543 |
| r | 0.5 | 0.208 | 0.628 | 0.149 | 0.591 | 0.183 | 0.612 |
| r | 2.0 | 0.180 | 0.610 | 0.087 | 0.553 | 0.168 | 0.603 |

GARCH's conditional-variance path has the *highest* AUC of the three
rungs in every one of the five floor cells — the information is
present in the fitted path, so the fixed calibrated CUSUM wrapper on
top of it, not the GARCH fit itself, is what fails to use it (this
rules out the more severe structural-mismatch explanation without
fully isolating why the wrapper underuses the information).

A companion oracle check (`exp37_garch_oracle_break_aware.py`, GARCH
refit separately on the true pre-/post-break segments — only
computable with the break time already known, hence an oracle
diagnostic like the known-parameter columns of `exp10`/`exp26`/`exp30`,
not a fair causal-grid entry) confirms this: post-break z² elevation
collapses to null-level noise (−0.14 to +0.06) versus the plain fit's
+0.16 to +6.4, and the oracle crosses its own threshold only
0.2–5.8% of the time. Perfect break-knowledge reveals no signal the
plain fit was missing — the CUSUM wrapper is the whole bottleneck, at
least for a literal two-segment refit. A genuinely causal break-aware
GARCH (online-estimated break location, refit without lookahead)
remains open infrastructure; Berkes, Gombay, Horváth & Kokoszka (2004)
and Andreou & Ghysels (2002) give related sequential/retrospective
GARCH change-point tests, and the CUSUM-of-squares-for-GARCH
literature following Kokoszka & Leipus (1999, 2000) contextualizes
this benchmark's limitations, but none has been run against this
paper's calibrated-FAR protocol.

**Offline changepoint detection.** Offline changepoint methods (PELT,
Killick et al. 2012) solve a retrospective segmentation problem; our
monitoring is strictly causal and calibrated to a false-alarm rate, so
the two are not comparable on delay (§8.5). We nonetheless calibrate
PELT to the same false-alarm rate on null paths and ask whether it
*localizes* a break at all, given the full sample: it is competitive
with the raw-Y CUSUM on an obvious 3σ level shift (localization rate
0.83–0.92 across SNRs vs. the causal detector's 0.97–0.99) but far
weaker on pure variance breaks (0.00–0.20 vs. the dedicated raw
variance-CUSUM's 0.09–1.00 across the same grid), because PELT's default
cost model is a mean-shift detector — a concrete reason a dedicated
variance statistic, not an off-the-shelf offline method, is the right
tool for the channel this paper is about. The canonical retrospective
alternative for multiple mean breaks specifically, rather than PELT's
general segmentation cost, is the dynamic-programming estimator of Bai
& Perron (1998, 2003); we do not benchmark against it directly since our
question is causal monitoring at a calibrated false-alarm rate, not
retrospective break-date estimation, but it is the natural reference
point for §7's multi-break discussion.

**Online Bayesian changepoint detection.** A separate causal-monitoring
literature frames change detection as online Bayesian inference over
the run length since the last change (Adams & MacKay 2007), rather than the
frequentist calibrated-threshold approach we use throughout; we do not
benchmark against it directly — a proper comparison would need to
calibrate its posterior-probability alarm rule to the same false-alarm
rate as every other detector here, which is a different exercise from
this paper's threshold-calibration protocol. For a broader view of
where the sequential change-detection field has moved since the
classical quickest-detection results this paper builds on, Xie et al. (2021)
surveys both the classical theory and more recent directions,
including the kind of causally-constrained, distribution-free
monitoring this paper's protocol is an instance of.

What is new here is not any single detector but the calibrated-parity
harness that makes latent-state and raw-data detectors directly
comparable, and the channel decomposition it enables. The exact
ARMA(1,1) equivalence of the ARIMA and Kalman rungs (§5) is a known
identity from structural time-series theory (Harvey 1989; Hamilton
1994, ch. 13), not a result established here; what the harness
contributes is using that identity to dissolve the "does filtering
help?" question for second moments into a prewhitening question with a
known answer, and then decomposing that answer by variance channel (r
vs. q).

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
standardization is a design flaw that blunted the composite; §8.3). The
composite score is the max of standardized features; the composite is
itself calibrated on nulls, which prices in the multiple-feature testing
automatically.

**Calibration parity.** Threshold = the (1 − FAR)-quantile of the
per-replication maximum score over matched-null draws; identical routine,
budgets, and seed layout for LSC (Latent-State Structural Change,
the filtered-state detector family this paper studies) detectors and
benchmarks (raw-Y CUSUM,
ARIMA+CUSUM, plain-HMM regime flips). Calibration, evaluation, FAR-check,
and feature-scale seeds are disjoint by construction.

**Metrics.** Detection rate, pre-break false alarms, censored delay; for
multi-break: event-level precision/recall/F1 with one-to-one greedy
matching in a 100-observation window. To connect the protocol to the
statistical-process-control and quickest-detection literatures we also
report the two average-run-length quantities those fields are built
around (`paper_assets/arl_table.csv`). The in-control **ARL₀** is the
mean observations between false alarms implied by the calibrated
per-observation false-alarm hazard, under the simplifying assumption
that this hazard is constant across the monitored window — a CUSUM
accumulated against a frozen training-prefix baseline does not
actually have a memoryless, constant-hazard alarm process, so ARL₀
here is a convenient SPC-style summary of the calibrated window-FAR,
not a directly measured mean run length: with monitored window length
L = 375 (T = 500, 25% training) and empirical window-FAR α, the hazard
is p = 1 − (1−α)^{1/L} and ARL₀ = 1/p ≈ L/α; our 5% window target
corresponds to ARL₀ ≈ 7300 observations, and the empirical detectors sit
at 4383–10841 across the 12 arena×method rows in Table 1 (raw CUSUM
calibrates hot, increasingly so at higher SNR:
4.0% → 6.2% → 8.2% vs. the 5% target, and lsc_state_cusum's SNR=2.0
cell is hot too at 7.8% — together these drive the range's low extreme,
4383 and 4618; lsc_kalman_cusum's cold SNR=0.1 cell, 3.4%, alone drives
the high extreme, 10841). A follow-up
(`experiments/exp38_raw_cusum_far_correction.py`) checks whether this drift
is a calibration-sample-size artifact rather than a genuine confound
on Table 2's ordering: raw_cusum is already calibrated per-SNR
separately (not against a single pooled threshold), so the question
is whether a much larger calibration sample (5000 vs. 500 reps)
tightens the threshold. It does: thresholds rise (27.49→29.30,
103.19→116.93, 213.89→250.17 across the three SNRs) and out-of-sample
FAR moves toward 5% at every SNR (5.4%→4.6%, 6.4%→4.7%, 7.3%→4.7%).
But raw_cusum's detection rate barely moves under the tighter
threshold (≤0.008 change at any SNR), and its advantage over the
innovation CUSUM is essentially unchanged (+0.312→+0.310,
+0.436→+0.434, +0.314→+0.306). The FAR drift is real — a small-sample
calibration effect, now quantified rather than only noted — but it
does not threaten Table 2's headline ordering. The out-of-control
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
(n_reps = 500, √(p(1−p)/n_reps)). lsc_state_cusum is tracked here for
FAR/ARL₀ purposes as one of the composite's constituent features run
standalone; its own detection rate is not separately carried through
Table 2 and later grids, which focus on lsc_kalman_cusum (the
innovation CUSUM Proposition 1 describes) and lsc_composite.*

| Arena | Scenario | Method | Detect rate | SE (detect rate) | ARL₁ (mean delay) |
|---|---|---|---|---|---|
| SNR 0.1 | level 3σ | lsc_composite | 0.766 | 0.019 | 112.6 |
| SNR 0.1 | level 3σ | lsc_kalman_cusum | 0.654 | 0.021 | 63.9 |
| SNR 0.1 | level 3σ | raw_cusum | 0.966 | 0.008 | 72.3 |
| SNR 0.1 | variance ×3 (r) | lsc_composite | 0.990 | 0.004 | 29.3 |
| SNR 0.1 | variance ×3 (r) | lsc_kalman_cusum | 0.870 | 0.015 | 80.4 |
| SNR 0.1 | variance ×3 (r) | raw_cusum | 0.728 | 0.020 | 104.0 |
| SNR 0.5 | level 3σ | lsc_composite | 0.530 | 0.022 | 86.4 |
| SNR 0.5 | level 3σ | lsc_kalman_cusum | 0.554 | 0.022 | 77.2 |
| SNR 0.5 | level 3σ | raw_cusum | 0.990 | 0.004 | 81.8 |
| SNR 0.5 | variance ×3 (r) | lsc_composite | 0.992 | 0.004 | 25.2 |
| SNR 0.5 | variance ×3 (r) | lsc_kalman_cusum | 0.224 | 0.019 | 107.2 |
| SNR 0.5 | variance ×3 (r) | raw_cusum | 0.076 | 0.012 | 137.6 |
| SNR 2.0 | level 3σ | lsc_composite | 0.670 | 0.021 | 63.2 |
| SNR 2.0 | level 3σ | lsc_kalman_cusum | 0.674 | 0.021 | 49.0 |
| SNR 2.0 | level 3σ | raw_cusum | 0.988 | 0.005 | 96.6 |
| SNR 2.0 | variance ×3 (r) | lsc_composite | 0.976 | 0.007 | 17.2 |
| SNR 2.0 | variance ×3 (r) | lsc_kalman_cusum | 0.268 | 0.020 | 114.6 |
| SNR 2.0 | variance ×3 (r) | raw_cusum | 0.058 | 0.010 | 172.9 |

*Table 2. ARL₁ (detection rate and mean delay conditional on detection)
at the canonical level-3σ and variance-×3 breaks, T = 500. The
variance break is observation-noise (r-channel); the state-innovation
(q-)channel is introduced separately in Table 3. Per-cell MC SEs on
detect rate reported directly above (n_reps = 500, √(p(1−p)/n_reps));
all ≤ 0.023.*

**Every detector, not just raw CUSUM, resolves under the same
protocol.** The recalibration check above covers raw_cusum alone,
against a single larger calibration sample. Re-running raw_cusum
alongside every other
Table 1–3 detector — raw_cusum, lsc_kalman_cusum, lsc_state_cusum,
lsc_composite,
raw_var_cusum, arima_var_cusum, est_kalman_var_cusum (§5), and
lsc_tail_cusum, all 8 — with a stricter protocol (bisect the threshold
against a disjoint 2000-replicate fresh-null block rather than only
enlarging the calibration sample; arima_var_cusum alone used a reduced
1000-calibration/1000-fresh budget for cost reasons, `experiments/
CHANGELOG.md`)
confirms the finding generalizes:
all 24 (detector, arena) combinations land within [4.5%, 5.5%] of the
5% target by construction (`experiments/exp46_far_parity.py`). This
includes lsc_kalman_cusum's cold SNR = 0.1 cell (Table 1's 3.4% at
n = 500; 4.7% once calibrated on 5000 reps and checked against a fresh
block) and raw_cusum's hot SNR = 2.0 cell (6.95% checked against this
run's own fresh block at the n = 500 threshold; 4.4% once calibrated
on 5000 reps) — both close toward 5% with a larger calibration sample
rather than needing a hot- or cold-specific fix, and the bisected
threshold lands every detector at exactly 5.0% FAR by construction.
Table 1's originally reported 3.4–8.2% spread is therefore small-sample
calibration noise, symmetric in cause across detector families even
though asymmetric in the magnitude first reported — not evidence that
one family calibrates structurally hot and another structurally cold.
Nor does it threaten this paper's headline ordering: at the FAR-matched
threshold, raw_cusum's level-3σ advantage over lsc_kalman_cusum changes
by at most 0.036 at any SNR (0.312 → 0.286, 0.436 → 0.462, and
0.314 → 0.350 at SNR 0.1/0.5/2.0 respectively — narrowing at one SNR,
widening slightly at the other two; `exp46_detect_matched.csv`), well
inside the ±0.10 tolerance a materially different ordering would need
to exceed.

## 3. Simulation design

Arenas: AR(1) latent state (φ = 0.95) at spec-SNR (stationary state
variance / observation variance) ∈ {0.1, 0.5, 2.0}, with a persistence
sweep φ ∈ {0.5, 0.8, 0.95, 0.99} (SNR held fixed by q = SNR·(1−φ²),
the observation variance r ≡ 1 throughout the base DGP; §5 introduces
r as an independently varying quantity for the variance-channel
scenarios and states the fuller q = SNR(1−φ²)r)
used to map the fast-or-never boundary (§4); and a local-level
(random-walk state) arena, analyzed in §8.4, where — as we now
demonstrate rather than assert — level detection is degenerate for
every method and variance detection becomes possible only after
whitening. Breaks at mid-sample come in **two variance channels**
whose distinction is central to §5. An *observation-noise* (r) break
scales the white-noise component's standard deviation (×1.5, ×3, ×⅔
quieting); a *state-innovation* (q) break scales the state shock's
standard deviation by the same factors — the identical SD convention,
so "×1.5" means SD×1.5 in both channels. The two are structurally
different: an r-break changes only the marginal variance of Y, whereas
a q-break changes both the marginal variance and the autocorrelation
structure (it shifts the reduced-form ARMA(1,1) MA parameter). Also:
level shifts (0.5, 1, 3 σ_ref, where σ_ref = √(q/(1−φ²)) is the latent
state's own stationary standard deviation), logistic ramps, and pure
persistence changes (φ → 0.995 or 0.80 with the stationary variance
held fixed). T = 500 baseline with a T ∈ {200, 2000} sweep;
misspecification arenas with t₅ observation noise and a nonlinear tanh
state drift.

## 4. First moments: raw data wins, and we can say exactly why

**Empirics.** At matched FAR, raw-Y CUSUM has the best level-shift
detection rate at every SNR (0.97–0.99 at 3σ) and every T; the
innovation CUSUM approaches but never overtakes it (0.55–0.67 at 3σ across
the SNR grid, non-monotonic in SNR — Table 2).
A pre-registered SNR-sweep hypothesis — that the innovation CUSUM should
beat raw_cusum on both detection rate and delay as SNR falls
(`experiments/CHANGELOG.md`, 2026-07-10, exp02) — was falsified on
detection rate at every SNR tested; the delay side is more nuanced.
The
innovation CUSUM is the speed champion conditional on firing:
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

Part (a) of this bound — the geometric decay of the post-break
innovation mean — restates a standard linear time-invariant (LTI)
filter step-response computation (Harvey 1989; Hamilton 1994, ch. 13);
part (b) — the tail bound itself — is a standard exponential-martingale
(Wald-type) argument from sequential analysis, of the kind used
throughout quickest detection and SPC (Wald 1947; Siegmund 1985;
Basseville & Nikiforov 1993). Appendix B states and proves both parts
explicitly for this DGP. What is new here is not either building block
but combining them into this specific bound for the Kalman/ARMA(1,1)
innovation CUSUM and using it, below, to explain the raw-vs-innovation
detection-rate gap.

If the calibrated threshold h clears the regime the bound in part (b)
describes, the detector fires during the adaptation transient or, with
exponentially small probability, never. At the thresholds this paper
actually calibrates to, that regime is not reached: across every 3σ
cell in the two grids audited (30 total: φ ∈ {0.5, 0.8, 0.95} at 6–12
cells each, plus φ = 0.99 at 6 cells), the bound is either ≥ 1
(vacuous, 24 cells — including all 12 at the paper's own φ = 0.95) or,
at φ = 0.99, orders of magnitude below the 1/500 resolution this
project's replicate budget could ever falsify (the remaining 6 cells)
— so all 30/30 audited 3σ cells are empirically idle one way or the
other (`experiments/exp52_prop1_bound.py`). At the flagship cell (SNR 0.5, 3σ)
we tested the qualitative claim directly rather than relying on the
vacuous bound: splitting observed lsc_kalman_cusum alarms at the
transient's end (T* = 13 post-break observations, the time for μ_t to
decay within 5% of μ∞) shows that of the 477/500 replicates that
survive the transient without alarming, 54.3% go on to alarm during
the remaining post-transient window (`experiments/exp52b_flagship_curve.py`)
— not the exponentially rare behavior a binding bound would predict.
With φ = 0.95 and k = 0.5, μ∞ > k
would require shifts of order 10σ, so the innovation CUSUM's
post-transient drift is asymptotically negative — the mechanism that
makes it the speed champion but never the detection champion — but
"asymptotically negative drift" and "exponentially rare to ever fire"
are different claims, and only the first is supported at this paper's
own operating point; the second is a boundary-condition property that
holds at low persistence (φ sweep, below), not a guarantee at φ = 0.95,
SNR 0.5.

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
Carlo innovation path matches μ_t within MC error; the Proposition 1(b)
bound is never violated — trivially so at 3σ, where it is vacuous or
below this project's resolution floor at every audited cell (above) —
but more informative at smaller magnitudes: at δ = 0.5σ the bound is
≤0.06% and observed detection (4.4–5.4%) matches ≈FAR, as the
near-zero bound predicts; at δ = 1σ the bound is larger (≤0.71%) and
still small, but observed detection (11.2–13.0%) is roughly double
FAR — the bound at 1σ is non-vacuous but not tight enough to predict
the observed rate itself, only that it stays well below 1. At the actual calibrated thresholds,
μ∞ still sorts every
observed detection rate; δ = 3σ is a knife-edge (μ∞ = 0.43–0.48 vs k = 0.5) matching the observed
0.55–0.67, though the vacuity of the 3σ bound means this sorting is a
property of Proposition 1(a)'s exact decay rate, not a check of
Proposition 1(b)'s tail bound. Proposition 2's Wald delays 68/84/110 across SNRs bracket the
observed raw medians 58/75/91 (≈12–21% conservative: 17%/12%/21% at
SNR 0.1/0.5/2.0) and explain the
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
so SNR and shift size specifically are ruled out as the driver, though
the design cannot separately distinguish μ∞ from other quantities that
are themselves monotone in φ within a stratum (e.g. the Kalman gain K,
or 1−φ directly) — isolating μ∞ from those would need an iso-μ∞
contour design, not run here; rejects at p<0.00005 over 20,000 draws;
`experiments/exp12_spearman_null_test.py`), fast-regime cells (μ∞ ≥ k = 0.5) detect
0.83–1.00 while never-regime cells (μ∞ < 0.5) detect 0.07–0.67, and at
3σ the detector escapes the trap at low persistence (0.98–1.00 across
SNRs at φ = 0.5, 0.93–0.97 at φ = 0.8) but is caught at high
persistence (0.55–0.67 at φ = 0.95, 0.30–0.63 at φ = 0.99) — raw
CUSUM's own detection also declines at φ = 0.99
(to 0.75–0.78 across SNRs, `grid_v6_phisweep_results.csv` — plausibly
the same near-unit-root threshold-inflation mechanism documented for
the variance rung at §5), but far less steeply than the innovation
CUSUM's own φ=0.99 range (the high end, 0.63, being the SNR=0.1 cell
discussed as an exception just below): the
φ-dependence is sharper for the filtered detector, not exclusive to
it. The negative
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
trap at low persistence (0.93–1.00 at φ = 0.5–0.8) and is caught at high
persistence (0.30–0.67 at φ = 0.95–0.99).](paper_assets/grid_v6_muinf_scatter.png)

**Assumptions and estimation error.** Propositions 1–2 assume the
steady-state filter with *known* parameters. Two facts keep this from
being a limitation in practice: the filter reaches its steady state
within the 125-observation training prefix (25% of T = 500), well
before monitoring begins; and the error from estimating (φ, q, r)
rather than knowing them is second-order for the innovation *series* —
the ARMA(1,1)/Kalman equivalence check (§5, `exp07`) finds estimated-
parameter ARIMA and Kalman innovations correlate at ρ̄ = 0.99 with each
other, against ρ = 1.000 at true parameters (max discrepancy ≤2×10⁻⁷).
This establishes series-level agreement under estimation; the
detection-rate consequences are shown empirically, not proved
analytically, and on at least one margin the gap is NOT second-order:
at the Table 2 flagship cell (SNR 0.5, φ = 0.95, level 3σ), holding
sidedness and calibration fixed, true parameters raise the two-sided
innovation-CUSUM detection rate from 0.554 to 0.970, and known
parameters with a one-sided CUSUM (a common alternative convention,
not what Table 2 reports) reach 0.990; the fourth corner — one-sided/
estimated — is 0.636 (`exp10_cusum_ablation.py` computes all four in
one run). Sidedness alone buys +0.082 over the two-sided/estimated
baseline, known parameters alone buy +0.416 — close to additive
(0.554+0.082+0.416=1.052 vs. observed 0.970/0.990, both within the
ceiling's range) rather than strongly interacting — so a reader
reimplementing this cell under either convention should expect a
substantially higher rate than 0.554, which reflects the estimated,
two-sided construction actually used, not an error.

**The same ablation on the variance ladder.** exp10 above covers one
cell of the level-shift ladder; the identical known-vs-estimated
question on the variance ladder (Table 3, §5) had not been checked.
We add two known-parameter counterparts
(`lsc.benchmarks.variance.known_raw_var_cusum_score`,
standardizing by the DGP's analytic stationary SD instead of the
training-prefix sample SD; `known_kalman_var_cusum_score`, the same
three-arm variance CUSUM on steady-state rather than MLE-fit Kalman
innovations — the natural "known" reference point for
`arima_var_cusum`'s estimated whitening) and run them across the
identical 12-cell grid Table 3 uses
(`experiments/exp26_known_param_variance.py`).

| channel | vol_mult | SNR | raw: est. → known (gap) | Kalman/ARIMA: est. → known (gap) |
|---|---|---|---|---|
| r | ×1.5 | 0.1 | 0.996 → 0.988 (−0.008) | 0.900 → 0.986 (+0.086) |
| r | ×1.5 | 0.5 | 0.560 → 0.964 (+0.404) | 0.942 → 0.984 (+0.042) |
| r | ×1.5 | 2.0 | 0.102 → 0.168 (+0.066) | 0.868 → 0.984 (+0.116) |
| r | ×3 | 0.1 | 1.000 → 0.988 (−0.012) | 0.980 → 0.986 (+0.006) |
| r | ×3 | 0.5 | 0.998 → 0.980 (−0.018) | 0.998 → 0.984 (−0.014) |
| r | ×3 | 2.0 | 0.852 → 0.976 (+0.124) | 0.998 → 0.984 (−0.014) |
| q | ×1.5 | 0.1 | 0.094 → 0.198 (+0.104) | 0.032 → 0.084 (+0.052) |
| q | ×1.5 | 0.5 | 0.212 → 0.446 (+0.234) | 0.100 → 0.266 (+0.166) |
| q | ×1.5 | 2.0 | 0.230 → 0.498 (+0.268) | 0.158 → 0.722 (+0.564) |
| q | ×3 | 0.1 | 0.724 → 0.950 (+0.226) | 0.262 → 0.816 (+0.554) |
| q | ×3 | 0.5 | 0.962 → 0.980 (+0.018) | 0.794 → 0.984 (+0.190) |
| q | ×3 | 2.0 | 0.960 → 0.976 (+0.016) | 0.996 → 0.984 (−0.012) |

*Table 2b. Known-parameter minus estimated-parameter detection-rate
gap, variance ladder, n = 500, 5% calibrated FAR
(`exp26_known_param_variance`).*

**The Table 2 flagship gap is not a one-off — it generalizes, unevenly,
to the variance channel.** Nine of 12 raw-rung cells and 9 of 12
Kalman/ARIMA-rung cells show the known-parameter variant at or above
the estimated one; the negative cells sit at or near the detection
ceiling, where there is little room to differ in either direction.

**Paired-SE correction to the negative cells (Experiment F).** A
systematic paired-SE reconstruction of all 12 Table 2b cells
(`experiments/exp43_table2b_paired_se.py`, the same bit-identical-
reproduction-first methodology as exp40; all 12 cells reproduced their
published aggregates exactly) shows the six negative gaps are not
uniformly "within MC noise" as an eyeballed independence-bound SE
(≈0.01–0.02) would suggest. Ranked by |gap|/paired-SE: raw
r×1.5/SNR0.1 (−0.008, 1.4 SE) is genuinely small; the other five are
2.1–3.0 paired SEs from zero — raw r×3/SNR0.1 (−0.012, 2.5 SE), raw
r×3/SNR0.5 (−0.018, 3.0 SE), Kalman/ARIMA r×3/SNR0.5 (−0.014, 2.3 SE),
r×3/SNR2.0 (−0.014, 2.3 SE), and q×3/SNR2.0 (−0.012, 2.1 SE). These are
small in absolute magnitude (all |gap| ≤ 0.018) and every one sits with
both compared rates ≥ 0.95, so the substantive reading is unchanged —
real but tiny reversals confined to near-ceiling cells, not evidence
against the broader known-parameter-helps pattern — but "within MC
noise" overstated the case for five of the six; "small and
ceiling-bound but not attributable to noise alone" is the more precise
description. Pairing tightens the SE relative to the independence
bound in 17 of the 24 rung-cell combinations tested; in the other 7 the
two rungs are anticorrelated enough per-replicate that pairing widens
rather than narrows the SE, reported as found rather than smoothed to a
uniform direction.

Two findings are large enough to change how a cited result
should be read. (Throughout this paper, "Outcome A/B/C" labels the
possible results of a pre-registered decision rule — the outcomes are
named and their consequences fixed *before* the relevant grid is run,
per this project's pre-registration discipline (Appendix A;
`experiments/CHANGELOG.md`). A given decision rule's own outcomes are
defined where it is first resolved, e.g. "Outcome C" below refers to
the r-channel's SNR-dependent collapse mechanism, first named where
§5 resolves this ladder's channel-inversion decision rule.) **First**, r ×1.5/SNR 0.5 — a cell on the steep part of
Outcome C's SNR-dependent collapse (raw_var_cusum: 0.996 / 0.560 / 0.102
at SNR 0.1/0.5/2.0, §5) — nearly closes under known parameters (0.560 →
0.964, +0.404, the largest gap in the raw-rung column — the
Kalman/ARIMA-rung column has larger gaps still, up to +0.564, see
below); SNR 2.0 still collapses
even with known parameters (0.102 → 0.168), so the collapse itself is
not purely an estimation artifact, but its steepness at the SNR 0.5
midpoint substantially is. Outcome C's mechanism (state-driven
autocorrelation masking a shrinking noise-variance signal as SNR rises)
remains the right explanation for the SNR 2.0 floor; it is not the
whole explanation for the SNR 0.5 cell specifically. **Second**, the
q-channel Kalman/ARIMA-rung gaps are large and one-sided (+0.052 to
+0.564, never meaningfully negative) — consistent with, and
sharpening, exp20/exp21's finding (§5) that the ARIMA composite's
underperformance traces mostly to the innovation series itself. A
follow-up (`experiments/exp30_order_known_arima.py`) isolates which
part of this gap is AIC picking the wrong order versus MLE noise on
whichever order is picked: forcing the true ARMA(1,0,1) order while
still fitting its coefficients by MLE (`order_known`), scored on the
identical simulated paths as both the `estimated` and `known` rungs
above so the three form a genuine decomposition rather than three
independent Monte Carlo draws, gives, across all six q-channel cells:

| SNR | vol_mult | estimated | order_known | known | gap: order-selection | gap: coefficient noise |
|---|---|---|---|---|---|---|
| 0.1 | ×1.5 | 0.032 | 0.030 | 0.084 | −0.002 | +0.054 |
| 0.5 | ×1.5 | 0.100 | 0.112 | 0.266 | +0.012 | +0.154 |
| 2.0 | ×1.5 | 0.158 | 0.176 | 0.722 | +0.018 | +0.546 |
| 0.1 | ×3 | 0.262 | 0.230 | 0.816 | −0.032 | +0.586 |
| 0.5 | ×3 | 0.794 | 0.814 | 0.984 | +0.020 | +0.170 |
| 2.0 | ×3 | 0.996 | 0.998 | 0.984 | +0.002 | −0.014 |

*Table 2c. Decomposition of the q-channel known-minus-estimated gap
into order-selection and coefficient-estimation components (n = 500,
5% calibrated FAR, same seeds as exp26; `exp30_order_known_arima`).
Fresh-null FAR check on `order_known`: 3.2–6.4% across the three
SNRs. All six splits reconstruct the corresponding gap above exactly,
since `order_known` is scored on the same paths as `estimated` and
`known`.*

The order-selection component is within ±0.03 of zero in five of six
cells, and only marginally outside it in the sixth (q, ×3, SNR 0.1:
−0.032);
essentially the entire known-minus-estimated gap is MLE
coefficient-estimation noise on the correctly specified model, not
AIC choosing the wrong order. This sharpens rather than overturns the
practical recommendation — raw wins the q-channel when the whitening
filter's coefficients must be estimated — by attributing the "why"
to a specific, measured source instead of a bundled
estimation-and-order-selection hand-wave. One asymmetry this does not
resolve: `known_raw_var_cusum` is handed only the analytic stationary
SD while `known_kalman_var_cusum` is handed the full model, so the
known-parameter whitened rung's gains are plausibly inflated relative
to raw's by that asymmetry regardless of the order-known result here.
A reader reimplementing any single variance-ladder cell against a
known-parameter
or exact-filter baseline should expect materially higher detection
rates at several SNR/channel combinations, not just at the Table 2
flagship cell.

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
CUSUM's detection rate crosses 50% (ratio ≈ 2.1–2.9 across the three
SNRs: 2.13/2.88/2.40 at SNR 0.1/0.5/2.0) sits well below the ratio at which the *theoretical* μ∞ crosses
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
+ noise DGP the observable Y has an *exact* ARMA(1,1) reduced form — a
known signal-plus-noise / structural-time-series identity (Harvey 1989;
Hamilton 1994, ch. 13), not a new result — which we use here to collapse
the ladder: differencing by (1 − φL) leaves an MA(1), whose invertible
root gives an MA parameter θ and innovation variance σ_ε² satisfying two
identities we verify to machine precision (Appendix B; `lsc.theory
.arma11_representation`) — σ_ε² = F, the Kalman innovation variance, and
θ = ρ = φ(1 − K), the Proposition-1 decay rate. So the steady-state
Kalman innovations *are* the ARMA(1,1) innovations: the ARIMA and latent
rungs are the same filter, not two competitors. We confirm this
numerically (`experiments/exp07_arma_equivalence.py`, ≥200 null paths):
with true parameters the two standardized-innovation series agree to a
median correlation of 1.000 (max discrepancy ≤2×10⁻⁷, ≈10⁻⁹ at SNR
0.5/2.0 and ≈1.5×10⁻⁷ at SNR 0.1), and with each rung
fit on the training prefix — the actual operating condition — they still
correlate at ρ̄ = 0.99, the small wedge being pure estimation error
(forcing the ARIMA order to the true (1,0,1) tightens it to 0.9995).
(AIC, incidentally, rarely selects (1,0,1): near the unit root at φ = 0.95
it prefers (1,0,0) or a differencing (0,1,1); this is a benign near-unit-
root artifact — those orders approximate the ARMA(1,1) closely enough to
preserve ρ̄ ≥ 0.95 — reported in full in Appendix B.) The practical
consequence: the ladder is really **raw vs. whitened**, and "does the
*state estimate* help beyond ARIMA whitening?" has the answer *no, by
construction* — at the equivalence's own scope, true parameters — for
the single break-pressure statistic built directly
on the innovation series. Under estimation this construction argument
does not transfer cleanly (below).
Whether the same holds for the full 11-feature composite, several of
whose features are built from the Kalman *filtered state* rather than
the innovation series, is a separate empirical question with a
different, more qualified answer — taken up directly at the end of
this section. What remains here is the genuinely empirical question —
when does whitening help at all? — which turns out to depend on *which
variance channel* breaks.

**The equivalence is exact at true parameters; it does not survive
estimation, and the estimation-side gap runs in the direction that
weakens "no, by construction."** The identity above (σ_ε² = F, θ = ρ)
holds for the population Kalman/ARMA(1,1) filters; what an analyst
actually runs is a Kalman MLE fit or an AIC-selected ARIMA fit on a
125-observation training prefix, two different estimation procedures
applied to the same population target, not two readings of the same
fitted object. A fourth rung, `est_kalman_var_cusum` — the identical
three-arm variance CUSUM statistic used by `raw`/`ARIMA` above, applied
to a Kalman filter's MLE-estimated (not true-parameter) innovations
rather than an ARIMA model's AIC-selected-and-fit residuals — isolates
exactly this gap on the same 12-cell r/q × ×1.5/×3 × SNR grid Table 3
uses (`experiments/exp44_est_kalman_rung.py`; all 12 cells reproduce
the published raw/ARIMA aggregates bit-for-bit before the new rung is
trusted). The result: est_kalman_var_cusum is never worse than
arima_var_cusum at any of the 12 cells (11 strictly better, one exact
tie at ceiling) and decisively better at the q-channel, ×3, SNR 0.1
cell (0.524 vs. 0.262, Δ = +0.262, 12.0 paired SEs from zero) — a gap
large enough on its own to establish the direction, independent of the
three near-boundary cells (0.098–0.100) that individually would not.
Splitting the gap's source (`exp44_innovation_tails.csv`): at SNR 0.1
the two rungs' per-replicate max scores — the quantity the calibrated
threshold actually acts on — correlate at only 0.14–0.20, far below
the ≈0.99 median-innovation correlation the identity above
establishes, rising to 0.87–0.98 only at SNR 2.0 — where the detect-rate
gap is correspondingly small at the coarse ×3 break (≤0.002) but not
at the subtle ×1.5 break, where it is still 0.08–0.10 despite the
high max-score correlation, since both rungs sit far from the ceiling
there and have more room to differ. **The mechanism is not AIC order
selection.** Table 2c's own decomposition — run on ARIMA's
estimated-vs-known-order-vs-fully-known ladder, same q-channel grid —
shows that forcing the true (1,0,1) order at this identical cell
leaves ARIMA's own detection rate unchanged or slightly *worse*
(order_known 0.230 vs. estimated 0.262, gap −0.032; §4, Table 2c), and
that essentially the entire known-minus-estimated gap on ARIMA's own
ladder is MLE coefficient-estimation noise on the correctly specified
model, not order misselection (exp16's 7.8–12.0% true-order recovery
rate notwithstanding). So the same explanation cannot be the reason
`est_kalman_var_cusum` beats `arima_var_cusum` here: both are
continuous MLE fits, and ARIMA's own order-selection cost is measured
at essentially zero on this cell. The more likely source is that the
two MLE procedures, though targeting the same underlying ARMA(1,0,1)
structure, land at different points in coefficient space under
estimation noise, and the max-over-arms CUSUM's dependence on tail
excursions specifically — rather than median-innovation agreement —
amplifies whichever fit's estimation noise is larger into a
detection-rate gap at low SNR. Which fit is more tail-robust under
estimation noise, and why, is not established by this check and is
flagged as open rather than resolved. What the check does establish is
the *result*: the *state estimate* — the Kalman filter's MLE fit —
does buy real detection power over ARIMA whitening for this single
statistic once parameters must be estimated; "no, by construction" is
correct only at the equivalence's own true-parameter scope, and
understates the latent layer's contribution once estimation is
accounted for. This is a distinct fourth rung with its own numbers,
not a stand-in for arima_var_cusum. (This grid is φ = 0.95 only,
matching Table 3's own scope; the φ = 0.99 r-channel reversal noted
below is attributed to ARIMA's estimation fragility specifically, and
whether est_kalman also degrades there is not established by this
check and is not asserted either way.)

**Table 3. The ladder, both channels** (detection rate at T = 500, 5%
calibrated FAR; MC SEs ≤ 0.023 (worst-case √(p(1−p)/500), matching
Table 2's convention) in `paper_assets/ladder_table.csv`,
`break_channel` column). Rounded to 2 decimals here to match this
table's own MC SE; later tables reporting the same or related cells
(e.g. Table 2b, 4, 4b, Appendix C) carry the full 3-decimal precision
of the underlying CSVs — the values agree, only the display precision
differs (e.g. this table's "1.00" is Table 4's "0.996"). Bold cells
mark the SNR-extreme values discussed in the prose immediately
following the table.

**Paired, not independence-assuming, SEs — checked systematically for
all 46 raw-vs-ARIMA cells across Tables 3, 3b, 3c, and 4** (every cell
reproduces its published aggregate exactly first;
`experiments/exp40_systematic_paired_se.py`,
`paper_assets/exp40_systematic_paired_se.csv`). Pairing tightens the SE
in 31 of 46 cells; every one of the other 15 is a coarse ×3 scenario
where both rungs sit near the detection ceiling (e.g. r-channel
SNR 0.5: 0.998 vs. 0.998, paired SE 0.002831 vs. independence-bound
0.002826) — a ceiling effect (little per-replicate variance left to
exploit), not a breakdown of the pairing logic. Every subtle ×1.5 cell
without exception, and most coarse ×3 cells away from the ceiling,
show genuine tightening. Full per-cell figures are in the CSV; prose
below uses the paired SE where it materially changes the reading.

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
| 0.10 | 1.01 | 0.000 (0.014) | 0.344 (0.024) |
| 0.50 | 1.33 | 0.016 (0.014) | 0.526 (0.024) |
| 0.70 | 1.96 | 0.038 (0.014) | 0.528 (0.023) |
| 0.85 | 3.60 | 0.104 (0.015) | 0.204 (0.018) |
| 0.95 | 10.26 | 0.112 (0.018) | 0.168 (0.018) |
| 0.99 | 50.25 | 0.074 (0.017) | 0.302 (0.025) |

*Table 3a. Raw's detection-rate advantage over the ARIMA rung (Δ),
swept over φ at fixed q, r (`grid_v8_phiqbreak`). On the subtle break Δ
tracks the amplification factor and peaks at φ = 0.95 before receding
at the unit-root edge; on the coarse break Δ stays large at every φ,
including φ = 0.1 where amplification is negligible. Each Δ is a
difference of two detection rates at n_reps = 500; SEs in parentheses
are the TRUE paired-per-replicate SE(Δ), not a conservative
independence-assuming bound (raw and ARIMA
are scored on the SAME simulated path per replicate). Per-replicate
outcomes were not persisted by the original run, so
`experiments/exp19_paired_se_grid_v8.py` reconstructs them via a
determinism argument — identical seed and code path reproduce
bit-identical scores, verified directly (re-running twice, re-drawing
the same seed) rather than assumed — with reconstructed aggregates
matching all 12 published cells exactly; this checks the mechanism
that should make the reconstruction correct, not the original
individual-replicate pairing directly, since no such record survives.
SE(d̄) is reported, where
d = 1{raw detects} − 1{ARIMA detects} per replicate. The paired SEs
(0.014–0.025) run 20–56% below a worst-case, p = 0.5,
independence-assuming bound of 0.032, and
mostly below (roughly −0.5% to 20%) an
independence bound computed from the actual observed
rates — consistent with positive
correlation between the two detectors' per-replicate outcomes on most
cells. Re-reading the
subtle-break "recedes at the unit-root edge" claim against the PAIRED
SEs: 0.112 (φ = 0.95, SE 0.018) vs. 0.074 (φ = 0.99, SE 0.017), a
difference of 0.038 against a combined SE of
√(0.018² + 0.017²) ≈ 0.025 — about 1.5 SE, tighter than the ≈0.8 SE
the old conservative bound implied, but still short of a conventional
significance threshold. The qualitative shape is supported by the
broader sweep, but the specific 0.95-vs-0.99 ordering should still not
be read as a precise, noise-free finding — now a better-characterized
"suggestive but not conclusive" rather than "indistinguishable from
noise."

**Finer resolution around the peak.** Adding φ = 0.90 and φ = 0.97 to
the sweep (`experiments/exp39_phi_peak_joint_test.py`; Δ: φ=0.90 →
0.082, φ=0.95 → 0.112, φ=0.97 → 0.110,
φ=0.99 → 0.074) shows a plateau across φ = 0.95–0.97, not a sharp peak
at φ = 0.95 specifically — 0.97's value is nearly tied with 0.95's. A
joint bootstrap test of whether Δ(0.95) exceeds both neighbors
(φ = 0.90 and φ = 0.99) simultaneously gives p = 0.149 — not
significant at conventional levels, consistent with the "suggestive,
not conclusive" reading above rather than a precise peak location.

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

**A second operating point: φ = 0.99.** Every result above uses φ = 0.95
as the body arena; φ = 0.99 is more persistent and arguably more
empirically realistic (real macro/financial series often sit closer to
the unit root), and the theory's own apparatus (μ∞, the fast-or-never
boundary, the 1/(1−φ²) amplification) is explicitly φ-dependent, so
whether the ladder's trichotomy survives φ = 0.99 was a real, open
question for the r channel, pre-registered before running
(`experiments/CHANGELOG.md` 2026-07-24; `exp28_known_param_phi99.py`,
`phi99_robustness_table.py`). SNR held fixed across φ by the
grid_v4/grid_v6 convention q = SNR(1−φ²)r.

| channel | break | rung | φ=0.95, SNR 0.1 | 0.99 | φ=0.95, SNR 0.5 | 0.99 | φ=0.95, SNR 2.0 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| **r** | ×1.5 | raw | 1.00 | 0.98 | 0.56 | **0.24** | 0.10 | 0.07 |
|       |      | ARIMA | 0.90 | **0.58** | 0.94 | **0.20** | 0.87 | **0.27** |
|       | ×3   | raw | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | **0.56** |
|       |      | ARIMA | 0.98 | 0.97 | 1.00 | 0.99 | 1.00 | 0.99 |
| **q** (closest φ=0.95 comparator: SNR 2.0; φ=0.99's fixed-q induced SNR ≈ 2.45) | ×1.5 | raw | — | — | — | — | 0.23 | 0.14 |
|                     |      | ARIMA | — | — | — | — | 0.16 | 0.06 |
|                     | ×3   | raw | — | — | — | — | 0.96 | **0.58** |
|                     |      | ARIMA | — | — | — | — | **1.00** | **0.28** |

*Table 3b. The ladder at φ = 0.99 vs. φ = 0.95, estimated rungs* (T =
500, 500 reps, 5% calibrated FAR; MC SEs ≤ 0.023, matching Table 3's
protocol; paired SEs for all 6 cells are part of the systematic
check at Table 3 above, `exp40_systematic_paired_se`; bold marks the
cells the pre-registered φ=0.99 decision rule's prediction did
not anticipate; full data `paper_assets/grid_v9_r_phi99_results.csv`,
`paper_assets/phi99_robustness_estimated.csv`).

**φ = 0.5, 0.8, 0.95, and 0.99, not just two points.** Table 3b's
φ = 0.99 cell is a single additional point rather than a sweep, unlike
the level-shift persistence result (Table 3a/Fig. 2), which this table
should be held to the same standard as. A follow-up
(`experiments/r_phi_sweep_analyze.py`, `configs/grid_v9b_r_phi_lo.yaml`
for the new φ = 0.5/0.8 cells, joined with the already-published
φ = 0.95/0.99 columns) fills in the r-channel sweep:

| φ | subtle ×1.5, SNR 0.1 | SNR 0.5 | SNR 2.0 | coarse ×3, SNR 0.1 | SNR 0.5 | SNR 2.0 |
|---|---|---|---|---|---|---|
| | raw / ARIMA | raw / ARIMA | raw / ARIMA | raw / ARIMA | raw / ARIMA | raw / ARIMA |
| 0.5 | 0.996 / 0.410 | 0.936 / 0.900 | 0.436 / 0.580 | 1.00 / 0.996 | 1.00 / 1.00 | 1.00 / 1.00 |
| 0.8 | 0.996 / 0.842 | 0.856 / 0.954 | 0.212 / 0.826 | 1.00 / 0.988 | 0.998 / 1.00 | 0.992 / 1.00 |
| 0.95 | 0.996 / 0.900 | 0.560 / 0.942 | 0.102 / 0.868 | 1.00 / 0.980 | 0.998 / 0.998 | 0.852 / 0.998 |
| 0.99 | 0.976 / 0.580 | 0.240 / 0.200 | 0.072 / 0.266 | 1.00 / 0.974 | 1.00 / 0.988 | 0.558 / 0.992 |

*Table 3c. Full r-channel φ-sweep, estimated rungs, both break sizes
(n = 500, 5% calibrated FAR; `paper_assets/r_phi_sweep_full.csv`;
paired SEs for all 12 cells are part of the systematic check at
Table 3 above, `exp40_systematic_paired_se`).*

The breakdown is not gradual — it is a sharp reversal specific to
φ = 0.99, not a trend that emerges progressively across persistence.
On the subtle break at every SNR, raw's advantage over ARIMA
(raw − ARIMA) shrinks steadily from φ = 0.5 through φ = 0.95 (at
SNR 0.1: 0.586 → 0.154 → 0.096), then reverses sharply at φ = 0.99
(back up to 0.396) — the same up-then-sharply-recovering shape holds at
SNR 0.5 and SNR 2.0, though at those two SNRs the advantage crosses
into ARIMA's favor before φ = 0.95 (SNR 0.5: 0.036 → −0.098 → −0.382 →
0.040; SNR 2.0: −0.144 → −0.614 → −0.766 → −0.194, never crossing back
to positive even at φ = 0.99) — the sharp φ = 0.99 recovery is a shared
feature of the shape, not a shared sign. The
coarse break tells a different story entirely: ARIMA stays high and
stable across the whole sweep, including at φ = 0.99 (0.974/0.988/0.992
across the three SNRs there), while it is raw that degrades near the
unit root (1.00 → 0.558 at SNR 2.0), matching the nonstationarity-penalty
diagnosis already given below rather than a new mechanism. So: the
subtle-break whitening reversal is a unit-root-edge phenomenon specific
to φ = 0.99, not something that trends gradually across persistence —
a sharper and more defensible characterization than "the trichotomy
does not survive φ = 0.99" would be on its own.

**The estimated-rung r-channel story does not survive intact.** The
pre-registered prediction was that ARIMA would "stay flatter" than raw
at φ = 0.99, as it does at φ = 0.95 (0.90/0.94/0.87). It does not: the
φ = 0.99 ARIMA rung on the subtle ×1.5 break is low and *non-monotone*
(0.58/0.20/0.27) — worse than raw at SNR 0.1 (0.58 vs 0.98) and SNR 0.5
(0.20 vs 0.24), only pulling ahead at SNR 2.0 (0.27 vs 0.07, both weak
in absolute terms). This is a genuine falsifier of the "prewhitening
wins the r channel" claim as it would be read off the estimated Table 3
rung alone — the ladder's headline ordering is not φ-invariant.

**The known-parameter counterpart isolates why: estimation, not
mechanism, and the pre-registered decision rule resolves CONFIRMED IN
PART, FALSIFIED IN PART.** Standardizing by the DGP's true stationary
moments (`known_raw_var_cusum_score`, `known_kalman_var_cusum_score`,
exp26's method) at φ = 0.99: the known *Kalman* rung is flat at 0.984
across all three SNRs — matching its φ = 0.95 flatness (0.986/0.984/
0.984) — so whitening's *population-level* r-channel case is intact,
not weakened. Known-raw, by contrast, falls sharply with SNR even at
true parameters (0.990/0.390/0.062 at φ = 0.99, vs. mostly-flat
0.988/0.964/0.168 at φ = 0.95) — a real population-level effect of the
1/(1−φ²) amplification shrinking the noise-variance break's share of
Y's marginal variance (the same mechanism Table 3a documents for the
q-channel's raw *advantage*, here working against raw's own power).
Layered on top is a much larger *estimation* effect: the φ = 0.99
known-Kalman/estimated-ARIMA gap (+0.40 to +0.78,
`exp28_known_param_phi99.csv`) dwarfs the φ = 0.95 table's same-column
r-channel max (+0.12, 0.868→0.984 at SNR 2.0 — the q-channel's own max
there, +0.564, is a different channel and not commensurable) —
consistent with the near-unit-root AIC/MLE fragility already
documented (AIC rarely selects the true order at φ = 0.95, "benign"
there but not at φ = 0.99). At the coarse ×3 r-break, estimated ARIMA
stays flat at φ = 0.99 (0.97/0.99/0.99, matching φ = 0.95's
0.98/1.00/1.00) while estimated raw visibly falls (1.00/1.00/0.56 vs.
1.00/1.00/0.85) — but known-raw stays near-ceiling there too
(0.99/0.98/0.97), so raw's estimated coarse-break decline is *also*
substantially the training-prefix nonstationarity penalty (§8.4), not
a population collapse; the ARIMA ≥ raw ordering survives here more
because ARIMA's estimation holds up than because its population
advantage grows. The q-channel ordering (raw ≥ ARIMA) not only
survives φ = 0.99 but strengthens (coarse ×3/SNR 2.0: ARIMA 0.996 vs.
raw 0.960 at φ = 0.95, but raw 0.58 vs. ARIMA 0.28 at φ = 0.99) — though
known-Kalman beats known-raw on the q channel at both φ (φ = 0.95/
SNR 2.0: 0.722 vs. 0.498 subtle, 0.984 vs. 0.976 coarse; φ = 0.99:
0.236 vs. 0.152 subtle, 0.982 vs. 0.760 coarse), so the population-level
q-channel story was never "raw beats whitening" — only the estimated
one is, driven even more at φ = 0.99 by ARIMA's near-unit-root fragility
(known/estimated gap up to +0.70) than by the ARMA-θ-shift mechanism
Table 3 attributes it to at φ = 0.95.

**Resolution:** Confirmed — the q-channel estimated ordering (raw ≥
ARIMA) and the coarse r-break ordering (ARIMA ≥ raw) both survive
φ = 0.99, the latter more decisively. Falsified — the subtle r-break's
estimated ordering does not survive: ARIMA is not "flatter," it is
lower and non-monotone, losing to raw at two of three SNRs. The
known-parameter ablation resolves *why* rather than leaving it as
noise: near-unit-root AIC/MLE difficulty, not a population-level
change in what whitening buys, drives essentially all of the reversal
— whitening's population case is, if anything, stronger at φ = 0.99.
Exactly the boundary-condition finding Proposition 1 predicts (μ∞ and
the fast-or-never regime are both φ-dependent), reported as a genuine
scope qualifier, not smoothed over: **the r-channel "prewhitening
wins" claim is an estimated-rung, not a population-level, statement,
and it degrades specifically because ARIMA estimation — not the
whitening mechanism — struggles near the unit root.**

**Does the state help beyond whitening — for the full composite, not
just the innovation series?** The ARMA(1,1) equivalence above proves
"no" for the single break-pressure statistic on the innovation series
*at true parameters*; under estimation the single-statistic answer is
also not a clean "no" (§5, est_kalman_var_cusum vs. arima_var_cusum).
The equivalence, exact or otherwise, says nothing about the 11-feature composite: 5 of its features
(`break_pressure`, `variance_pressure`, `variance_pressure_slow`,
`variance_quiet`, `innovation_ac`) act on the innovation series and
inherit the equivalence directly, but 6 (`level_change`, `slope`,
`acceleration`, `instability`, `persistence`, `state_shift_pressure`)
act on the Kalman *filtered state*, which has no innovation-series
analog. We test this directly (`lsc.models.ARIMAModel`,
`experiments/exp20_composite_on_arima.py`): feed the SAME 11-feature
computation and composite machinery
(`lsc.diagnostics.features.compute_features`,
`lsc.eval.detectors.make_composite_detector`) an ARIMA model's
one-step-ahead fitted value in place of the Kalman filtered state
(direct substitution for the 5 innovation-based features, a disclosed
judgment call for the 6 state-based ones — an ARIMA model has no state
distinct from the series it fits, so "filtered-state slope" becomes
"one-step-ahead-forecast slope"), same calibration pipeline, same
seeds, same per-time standardization, same max-score rule.

**Table 4. Composite built on ARIMA inputs vs. the Kalman composite**
(detection rate, T = 500, 5% calibrated FAR;
`paper_assets/exp20_composite_on_arima.csv`).

| channel | break | detector | SNR 0.1 | SNR 0.5 | SNR 2.0 |
|---|---|---|---|---|---|
| **r** | ×1.5 | raw | 0.996 | 0.560 | 0.102 |
|   |   | ARIMA-CUSUM | 0.900 | 0.942 | 0.868 |
|   |   | composite (Kalman) | 0.818 | 0.868 | 0.910 |
|   |   | composite (ARIMA) | **0.226** | **0.416** | **0.632** |
| **r** | ×3 | raw | 1.000 | 0.998 | 0.852 |
|   |   | ARIMA-CUSUM | 0.980 | 0.998 | 0.998 |
|   |   | composite (Kalman) | 0.990 | 0.992 | 0.976 |
|   |   | composite (ARIMA) | 0.978 | 0.990 | 0.984 |
| **q** | ×1.5 | raw | 0.094 | 0.212 | 0.230 |
|   |   | ARIMA-CUSUM | 0.032 | 0.100 | 0.158 |
|   |   | composite (Kalman) | 0.064 | 0.106 | 0.234 |
|   |   | composite (ARIMA) | 0.044 | 0.104 | **0.096** |
| **q** | ×3 | raw | 0.724 | 0.962 | 0.960 |
|   |   | ARIMA-CUSUM | 0.262 | 0.794 | 0.996 |
|   |   | composite (Kalman) | 0.438 | 0.760 | 0.976 |
|   |   | composite (ARIMA) | 0.248 | **0.380** | 0.964 |

The answer is *not* "no, by construction," and it is not uniform. Where
the r-channel subtle ×1.5 break leaves room to differ (i.e. away from
the ceiling), the Kalman composite decisively beats the ARIMA
composite — 0.818 vs. 0.226 (SNR 0.1), 0.868 vs. 0.416 (SNR 0.5), 0.910
vs. 0.632 (SNR 2.0). A follow-up
(`experiments/exp35_composite_paired_se.py`) replaces the conservative
independence-assuming bound with the true paired per-replicate SE,
since both composites are scored on the same simulated path per
replicate, for three of these cells:

| cell | Δ | SE(Δ) paired | z | SE(Δ) independence bound |
|---|---|---|---|---|
| r ×1.5 / SNR 0.1 | +0.594 | 0.0229 | 25.96 | 0.0254 |
| r ×1.5 / SNR 2.0 | +0.278 | 0.0214 | 12.99 | 0.0251 |
| q ×3 / SNR 0.1 | +0.190 | 0.0213 | 8.93 | 0.0294 |

(r ×1.5/SNR 0.1's own reconstruction gets detect_kalman = 0.820 against
the published 0.818 — a 1-of-500-replicate difference, most plausibly
MLE/BLAS-order nondeterminism noted elsewhere in Appendix A, not a bug;
Δ = 0.594 above uses the reconstructed 0.820, internally consistent
with this table's paired SE though not bit-for-bit with the published
aggregate. The other two cells reproduce exactly.) None of these paired
SEs include variance from the composite's own per-time-point null-scale
estimation (`n_scale_reps = 50`, a single fixed scale seed at every
call site in this codebase) — the calibration step protects the false-
alarm rate against this source but the reported detection-rate SEs do
not account for it; a multi-seed sensitivity check on the scale
estimate itself is not run here and is flagged as an open gap rather
than resolved.
All three are decisively significant, and — as with Table 3a's own
paired-SE check (`exp19`) — the paired SE runs slightly tighter than
the independence bound. The same pattern holds, smaller in
magnitude, on the q-channel coarse ×3 break at SNR 0.1/0.5 (0.438 vs.
0.248; 0.760 vs. 0.380) and at q ×1.5
SNR 2.0 (0.234 vs. 0.096). Only in the near-ceiling cells — r ×3 at
every SNR, q ×3 at SNR 2.0 — do the two composites converge (within
0.01–0.02), and that convergence is because there is almost no room
left for either to differ, not evidence of equivalence. A second,
sharper anti-result: in every r ×1.5 cell and at q ×3/SNR 0.5, the
ARIMA composite is not just worse than the Kalman composite but worse
than the *single* ARIMA-CUSUM statistic feeding it (e.g. r ×1.5 SNR
0.1: composite 0.226 vs. the same ARIMA rung's own single-feature
statistic at 0.900) — bolting the other 10 features onto ARIMA inputs
is actively counterproductive there, a stronger form of the
max-over-features dilution already documented for a different
composite variant (COMPOSITE_ROBUST2, §8.3(ii)): the composite's
calibrated threshold is set by whichever feature has the heaviest null
tail, and the ARIMA-based state-analog features evidently have
worse-behaved null distributions than their Kalman counterparts,
taxing the whole composite's FAR budget more heavily.

**Isolating the source: innovation-only features vs. filtered-state
features.** Table 4's 11-feature composite mixes two kinds of
features: 5 (`break_pressure`, `variance_pressure`,
`variance_pressure_slow`, `variance_quiet`, `innovation_ac`) act on
the innovation series alone and inherit the ARMA(1,1) equivalence
directly; 6 (`level_change`, `slope`, `acceleration`, `instability`,
`persistence`, `state_shift_pressure`) act on the Kalman filtered
state, which has no innovation-series analog — ARIMA's one-step-ahead
fitted value stands in for it, a disclosed judgment call, not an
equivalence. Running the SAME 5-feature, innovation-only subset
through the unmodified composite machinery on both models, across the
identical 12-cell grid (`lsc.diagnostics.features.COMPOSITE_INNOV5`,
`experiments/exp21_composite_innov5.py`), isolates how much of the
11-feature gap survives with the 6 filtered-state features removed
entirely.

*Table 4b. 5-feature innovation-only composite (Kalman vs. ARIMA) vs.
the 11-feature composite of Table 4 (detection rate, T = 500, 5%
calibrated FAR; `paper_assets/exp21_composite_innov5.csv`).*

| channel | break | detector | SNR 0.1 | SNR 0.5 | SNR 2.0 |
|---|---|---|---|---|---|
| **r** | ×1.5 | composite-5 (Kalman) | 0.960 | 0.874 | 0.912 |
|   |   | composite-5 (ARIMA) | 0.518 | 0.416 | 0.632 |
|   |   | composite-11 (Kalman) | 0.818 | 0.868 | 0.910 |
|   |   | composite-11 (ARIMA) | 0.226 | 0.416 | 0.632 |
| **r** | ×3 | composite-5 (Kalman) | 0.994 | 0.992 | 0.978 |
|   |   | composite-5 (ARIMA) | 0.980 | 0.990 | 0.984 |
|   |   | composite-11 (Kalman) | 0.990 | 0.992 | 0.976 |
|   |   | composite-11 (ARIMA) | 0.978 | 0.990 | 0.984 |
| **q** | ×1.5 | composite-5 (Kalman) | 0.108 | 0.108 | 0.236 |
|   |   | composite-5 (ARIMA) | 0.050 | 0.104 | 0.096 |
|   |   | composite-11 (Kalman) | 0.064 | 0.106 | 0.234 |
|   |   | composite-11 (ARIMA) | 0.044 | 0.104 | 0.096 |
| **q** | ×3 | composite-5 (Kalman) | 0.596 | 0.778 | 0.978 |
|   |   | composite-5 (ARIMA) | 0.330 | 0.380 | 0.964 |
|   |   | composite-11 (Kalman) | 0.438 | 0.760 | 0.976 |
|   |   | composite-11 (ARIMA) | 0.248 | 0.380 | 0.964 |

Comparing gap = detect(Kalman) − detect(ARIMA) for the 5-feature
composite against the published 11-feature composite, the two agree to
within ~0.02–0.03 (the n = 500 noise floor) in 9 of 12 cells — e.g.
r/SNR 0.5/×1.5: 0.458 (5-feature) vs. 0.452 (11-feature); r/SNR 2.0/
×1.5: 0.280 vs. 0.278; q/SNR 0.5/×3: 0.398 vs. 0.380. Only r/SNR 0.1/
×1.5 shows the full composite pulling meaningfully further ahead
(0.592 vs. 0.442, the six filtered-state features adding a real ~0.15
to the gap there); two q-channel cells (q/SNR 0.1/×1.5 and q/SNR 0.1/
×3) show the *innovation-only* gap exceeding the full-composite gap —
adding the filtered-state features narrows the Kalman/ARIMA difference
there, not widens it. **Reading: for 9 of the 12 cells, the gap
traces almost entirely to the innovation series, not to the composite
losing genuinely state-specific information** — since Y is exactly
ARMA(1,1)-equivalent to the Kalman innovations on the null path,
ARIMA's own standardized one-step residual is a measurably worse
detection *input* under a break, and that alone reproduces the gap.

**Threshold and attribution diagnostic, r ×1.5/SNR 0.1 specifically
(exp22).** That cell is the one exception above, and the one where
Table 4's composite-on-ARIMA is furthest behind its own single ARIMA-
CUSUM feature (0.226 vs. 0.900). Reconstructing both
composites with the exact recipe behind Table 4
(`experiments/exp22_composite_threshold_argmax.py`): the ARIMA
composite's calibrated threshold (45.49) is 28.9% higher than the
Kalman composite's (35.28) at the same 5% FAR target — a genuine
"noisy substitute" signature (an ARIMA-fed feature diluting the shared
threshold, the same dilution mechanism documented for a different
composite variant in §8.3(ii)). But the argmax-feature distribution at
alarm time tells a more specific story: of the Kalman composite's 415
alarms, 400 (96%) are attributed to
`variance_pressure` — an innovation-only feature, not one of the 6
filtered-state features under dispute; of the ARIMA composite's 124
alarms, 96 (77%) are `variance_pressure` and another 19 (15%) are
`break_pressure`, also innovation-only — the 6 filtered-state-analog
features account for only 9/124 (7%) of ARIMA alarms and 11/415 (3%)
of Kalman alarms. Both readings hold simultaneously: the 6 ARIMA-fed
filtered-state-analog features rarely fire directly on a true break in
either composite (consistent with the innovation-only isolation
above), but still measurably inflate the null max-score distribution
the shared threshold is calibrated against, taxing detection power
without ever being the useful signal themselves — a real, now-measured
threshold tax (+28.9%) distinct from what the innovation-only
comparison alone could surface.

**This narrows §5's headline claim.** "The ladder is really raw vs.
whitened" is a proven identity for the single innovation-series
statistic, but only an approximation for the full composite — mostly
(9 of 12 cells) because ARIMA's whitened innovation series is itself a
worse detection input, compounded by the smaller exp22 threshold-tax
effect above. The honest summary: not "the state adds nothing beyond
whitening" but "...*for the single break-pressure statistic*; for the
richer composite it adds a measured, smaller-than-the-raw-gap-suggests
but nonzero amount, concentrated in the subtle-break regime where
detection is hardest."

**Does a smoothed (two-sided) state proxy close any of that gap
(Experiment D)?** The one-step ARIMA `fittedvalues` used above as the
state analog for the 6 filtered-state-analog features is itself a
disclosed judgment call, not a controlled substitution — a
fixed-interval (two-sided) Kalman *smoother* run over the same frozen
training-prefix fit is a different, more information-rich analog, at
the cost of no longer being causal (`lsc.models.SmoothedARIMAModel`,
`experiments/exp41_smoothed_arima_composite.py`; NOT a proposed
deployable rung, the same oracle-status caveat already given to
exp37's break-aware GARCH refit and the known-parameter scores above).
Re-running the r×1.5/SNR 0.1 cell (this section's own headline cell,
not the level-shift "flagship" cell of §2/§4) and its two SNR
neighbors (n = 500, same seeds as exp20):

| SNR | raw | ARIMA-CUSUM | composite (Kalman) | composite (ARIMA, one-step) | composite (ARIMA, smoothed) |
|---|---|---|---|---|---|
| 0.1 | 0.996 | 0.900 | 0.818 | 0.226 | **0.382** |
| 0.5 | 0.560 | 0.942 | 0.868 | 0.416 | 0.416 |
| 2.0 | 0.102 | 0.868 | 0.910 | 0.632 | 0.632 |

At SNR 0.1 the smoothed proxy closes part of the gap (0.226 → 0.382,
still short of the Kalman composite's 0.818); at SNR 0.5 and 2.0 it
makes no difference at all — threshold and detection rate are
identical to the one-step composite to full precision (thresholds
35.9996 and 33.2922, matching exp20 exactly). This is exactly what
exp22's attribution diagnostic predicts: smoothing only changes the 6
`est.filtered`-based features (which exp22 found drive just 7% of
ARIMA-composite alarms at SNR 0.1, and the bit-for-bit match at
SNR 0.5/2.0 shows they're fully non-binding there), not the 5
innovation-based ones a smoother can't alter. A genuinely better state
estimate helps exactly where the filtered-state features already had
some purchase, and not at all where they didn't — a further narrowing
of, not a reversal of, "destructive substitution, not missing state
information."

## 6. Dynamics: near the information floor

Pure persistence changes (marginals preserved) are close to undetectable:
at SNR 0.5, persistence-down is undetectable by every method (all at
FAR); persistence-up is not uniformly so (a pre-registered hypothesis
wrong in both directions) — raw CUSUM spuriously reaches 0.16 (against
its own 6.2% FAR) via a
conditional "level-freeze" artifact we dissect, and lsc_state_cusum
mildly exceeds FAR too (6.8% vs. 5.4%), while the other methods stay
at FAR. Quieting changes
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

This section studies causal, online detection of a second break given
a fixed re-arm protocol — a different question from retrospectively
estimating the number and location of multiple breaks in a full
sample, which is what the dynamic-programming approach of Bai &
Perron (1998, 2003) solves; we do not benchmark against it here for
the same reason given in Related Work for PELT (causal monitoring at
a calibrated false-alarm rate, not retrospective break-date
estimation), so its absence from the comparisons below is a framing
choice, not an oversight.

Under a re-arm protocol applied identically to all methods (re-arm when
the score drains below half threshold plus a 20-obs refractory), raw
CUSUM never detects a second event (recall 0.00 — its fixed-baseline
statistic saturates and cannot drain), but the LSC CUSUMs rarely re-arm in
time either at 150-observation spacing (level→level second-event recall
≤ 0.05): first-alarm delay plus drain time exceeds the gap. The exception
is cross-channel pairs: level→variance is caught principally by the
composite
(second-event recall 0.60, F1 0.63; lsc_state_cusum also catches it
partially, recall 0.42) because its variance features were
never saturated by the level event. Re-arming costs almost nothing under
the null (≤ 1.2% of null paths give a second alarm) — saturation, not
chatter, binds.

**A bounded-memory fix, and its limits.** The diagnosis
above — a fixed-baseline statistic compares every observation against
the *original* training-prefix reference and so never drains after a
permanent shift — points to a specific repair: replace the fixed
reference with a moving one. We build a MOSUM-style two-window
statistic (Chu, Hornik & Kuan 1995; the same family generalized for
online monitoring by Chu, Stinchcombe & White 1996 and Leisch, Hornik &
Kuan 2000) that compares a
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
scenarios, both windowed MEAN statistics stay at second-event recall ≈
0.00, because a pure variance change carries no mean signal for a
moving-window *mean* comparison to see. Closing that gap needs a
windowed *variance* statistic — a moving-window analogue of the r/q-
channel CUSUMs of §5 rather than of the mean-shift ones — which we
build and test on the identical `var_up_down` scenario (obs-noise ×3
at t = 200, ×⅓ back to baseline at t = 350, 150-observation spacing,
same arena/seeds/re-arm protocol as above): a two-window
log-variance-ratio statistic, `windowed_raw_var_score`
(`lsc.benchmarks.variance`, `experiments/exp27_windowed_variance.py`),
comparing a trailing window's mean-square to the window before it
rather than to the training-prefix baseline, exactly mirroring the
mean-shift fix's two-window design but on squared, not raw,
standardized observations. **It closes the gap.** Where the fixed-
baseline raw variance CUSUM gets recall_break1 = 0.998 but
recall_break2 = 0.000 (never drains) and the existing windowed
mean-shift statistic gets 0.000/0.000 on this scenario (no mean signal
at either break for it to see), the windowed variance statistic
reaches recall_break1 = 0.932, recall_break2 = 0.948, F1 = 0.958, at
precision 0.997 (n = 500, calibrated 5% FAR) — both events well
detected, at high precision, with no channel asymmetry between the
first and second event. Multi-break detection therefore needs the
statistic's channel matched to the break at BOTH levels — mean vs.
variance, and fixed-baseline vs. windowed — exactly as the single-break
results of §5 already required for the channel dimension alone; with
the matching statistic on each axis and the break sequence known in
advance to be single-channel, the "double-dip" failure mode this
section originally left open is not a structural limit of
bounded-memory monitoring, only of applying the wrong bounded-memory
statistic to a variance-channel break.

**But naively combining both windowed statistics for a mixed sequence
does not work.** The one real economic setting motivating this section
— a recession (level break) followed by a shock-variance regime
change, or the reverse — is a *mixed* ordering, where neither break is
known in advance to be the matched channel. A follow-up
(`experiments/exp36_combined_windowed_multibreak.py`) tests both
orderings with both windowed statistics running simultaneously
(`windowed_combined`, calibrated jointly by taking the max over both
statistics' scores against a common threshold, the same
max-over-statistics design as the main composite):

| ordering | method | recall break 1 | recall break 2 | precision | F1 | FAR |
|---|---|---|---|---|---|---|
| level→var | raw_var_cusum | 0.644 | 0.140 | 0.784 | 0.523 | 6.6% |
| level→var | windowed_raw_cusum | 0.692 | 0.002 | 0.991 | 0.461 | 6.8% |
| level→var | windowed_raw_var | 0.594 | 0.234 | 0.985 | 0.516 | 4.6% |
| level→var | windowed_combined | 0.692 | 0.004 | 0.991 | 0.462 | 6.8% |
| var→level | raw_var_cusum | 0.998 | 0.000 | 0.998 | 0.665 | 6.6% |
| var→level | windowed_raw_cusum | 0.000 | 0.000 | n/a | 0.000 | 6.8% |
| var→level | windowed_raw_var | 0.932 | 0.006 | 1.000 | 0.623 | 4.6% |
| var→level | windowed_combined | 0.044 | 0.000 | 1.000 | 0.029 | 6.8% |

This is a clean negative result, not a further fix. The joint
calibration sets `windowed_combined`'s threshold identically to
`windowed_raw_cusum` alone's — the variance arm's own well-calibrated
threshold never binds in the null-max calibration, so during
evaluation it is held to the mean-shift arm's much higher bar. In the
var→level ordering this is severe: `windowed_raw_var` alone catches
the first (variance) break 93.2% of the time, but under the naive
max-combination that collapses to 4.4%, because the mean-shift arm's
threshold dominates the max and the variance arm's own well-calibrated
signal is discarded even when it is the one that fired. This is the
same max-over-statistics dilution mechanism already documented for the
11-feature composite (§8.3(ii), Table 4), now shown to bind for a
two-statistic combination as well, not just an 11-feature one. So:
each windowed statistic alone genuinely closes its own matched-channel
gap, but naively combining them for a channel-unknown mixed sequence
does not — the likely actual fix is per-arm standardization (as the
composite's own per-t z-scoring already does across its 11 features),
not attempted here. The one real economic setting where this matters —
a recession followed by a shock-variance regime change, or the
reverse — is therefore still an open problem for bounded-memory
monitoring under channel uncertainty, not a solved one.

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
per-time-point as defined in §2, it is *diluted, not dead*: it reaches 0.58 at
×1.5 (SNR 0.5, Gaussian) and 0.21 at ×1.5 under t₅ — where it even edges
the e²-based composite's heavy-tail collapse (0.16, §8.2) — and ≈0.97 at
×3 under t₅. But it trails the standalone detector of (iii) at the subtle
break, because a max-over-≈10-features composite spreads the calibrated
5% false-alarm budget across all features: the exceedance feature's per-t
standardized score climbs past z ≈ 19 on a ×1.5 break, so bounded
increments *do* reach discriminating ratios (an earlier pooled-scale
build, before the per-time-standardization fix above, wrongly
suggested otherwise), yet the shared
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
Kalman-innovation 0.04–0.15, raw variance ≤ 0.10, ARIMA variance and
composite ≤ 0.07). A
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

**ICSS, the purpose-built variance-changepoint counterpart.** PELT's
poor showing on variance breaks (above) is a cost-model mismatch, not
evidence that offline retrospective methods are inherently unsuited to
this problem — the natural offline benchmark for variance changepoints
specifically is Inclán & Tiao's (1994) ICSS (iterative cumulative sums
of squares), not PELT with a mean-shift cost. We implement the
standard binary-segmentation approximation to ICSS — one-pass
recursive partitioning at the point of maximal normalized
cumulative-sum-of-squares deviation, `lsc.benchmarks.changepoint.
icss_breakpoints` — omitting Inclán & Tiao's own iterative refinement
step (re-checking each accepted candidate against its neighbors and
re-splitting until convergence); the omission is disclosed here rather
than left implicit, since the omitted step generally *improves*
localization, so the ICSS numbers below plausibly understate the true
algorithm's performance. Calibrated by simulation to the same 5% FAR
via the same bisection protocol used for PELT's penalty
(`experiments/exp25_icss_benchmark.py`), on the same standardized
post-training segment, at the same FAR-matched localization criterion
and window (±25 obs) — restricted to the variance scenarios only,
both channels, since ICSS has no mean-shift claim.

| Arena (SNR) | Scenario | ICSS localize rate | causal raw_var_cusum |
|---|---|---|---|
| 0.1 | r ×1.5 | 0.74 | 0.996 |
| 0.1 | r ×3 | 1.00 | 1.000 |
| 0.1 | q ×1.5 | 0.02 | 0.094 |
| 0.1 | q ×3 | 0.30 | 0.724 |
| 0.5 | r ×1.5 | 0.06 | 0.560 |
| 0.5 | r ×3 | 0.98 | 0.998 |
| 0.5 | q ×1.5 | 0.02 | 0.212 |
| 0.5 | q ×3 | 0.42 | 0.962 |
| 2.0 | r ×1.5 | 0.00 | 0.102 |
| 2.0 | r ×3 | 0.04 | 0.852 |
| 2.0 | q ×1.5 | 0.00 | 0.230 |
| 2.0 | q ×3 | 0.34 | 0.960 |

*Table 5b. ICSS localization rate at a FAR-matched (5%) operating
point, n = 500 per cell (`exp25_icss`), against the causal raw
variance CUSUM's detection rate on the identical cells (Table 3,
`grid_v4_varbench`/`grid_v5_qbreak`) for reference. MC SEs ≤ 0.023
(n = 500).*

ICSS clears the FAR floor far more often than PELT does on these same
scenarios (up to 1.00 vs. PELT's 0.00–0.20 ceiling on variance breaks)
— confirming the mismatch above was PELT's mean-shift cost model, not
an inherent offline-method weakness. But ICSS is dominated by the
*causal* raw variance CUSUM in 11 of 12 cells (tying only at r ×3/
SNR 0.1, both at ceiling) despite having an unfair advantage the causal
detector does not: ICSS sees the full 375-observation post-training
segment at once, with no online/causal constraint, while raw_var_cusum
only ever sees data up to its current alarm time. The gap is largest
exactly where the paper's other results predict it should be — the
r-channel, where §5's amplification mechanism (state-driven
autocorrelation swamping a shrinking noise-variance signal as SNR
rises) already explains raw_var_cusum's own SNR-dependence (Outcome C,
§5): ICSS collapses even faster with rising SNR than the causal
statistic does (0.74→0.06→0.00 at r ×1.5, vs. raw_var_cusum's
0.996→0.560→0.102 over the same SNR sweep), consistent with ICSS's
search over candidate breakpoints across the *whole* segment being
more exposed to that same autocorrelation-driven dilution than a
CUSUM accumulated causally against a fixed training-prefix baseline.
This is further evidence for the paper's calibrated-parity causal
framing generally: the offline advantage of seeing the future does not
rescue a variance-changepoint method here, even a correctly-specified
one, once the DGP's own state dynamics work against it.

On the canonical 3σ level break PELT is competitive with the causal
raw-Y CUSUM (0.83–0.92 here vs. 0.97–0.99 in Table 2's arena, seeing
the whole path rather than detecting online) — an off-the-shelf offline
method genuinely can find an obvious level shift. But on variance
breaks it is close to useless (0.00–0.20, against the dedicated raw
variance-CUSUM's ladder-table numbers of 0.09–1.00, §5), because PELT's
default l2 cost model is a mean-shift statistic: it responds to a
variance change only through the second-order effect that within-
segment sum-of-squares grows, a far weaker signal than a statistic
built for variance directly. The comparison sharpens rather than
undercuts the paper's thesis: an off-the-shelf offline segmentation
method is not a substitute for a channel-matched statistic, even when
it is given the unfair advantage of seeing the whole sample.

**Implementation lessons for practitioners.**
plain-HMM regime probabilities saturate and cannot be FAR-calibrated on
nonstationary data; probability-scale scores need log-odds; EM needs
persistent-initialization restarts; composite features must be
standardized per-time-point, not pooled; order-statistic thresholds have
Beta(n+1−k, k) noise regardless of distribution, so heavy-tailed
detectors need larger calibration budgets.

**8.6 A second-order DGP: does the trichotomy need AR(1)?** Every result
above — §4's level-shift finding, §5's whitening ladder — is derived and
tested on AR(1)+noise specifically; the paper's theoretical apparatus
(Propositions 1–2, the ARMA(1,1) equivalence of §5) is an exact
algebraic statement about that model, not a general one. We test
whether the *empirical* trichotomy survives a second-order persistence
structure that the theory does not directly cover: `AR2StateDGP`,
S_t = φ₁S_{t−1} + φ₂S_{t−2} + w_t (pre-registered
`experiments/CHANGELOG.md` 2026-07-24, before implementation;
`lsc/dgp/continuous.py`, `experiments/exp29_ar2_trichotomy.py`), under
two parameterizations chosen for qualitatively different persistence —
real, well-separated poles (φ₁ = 1.4, φ₂ = −0.45 → poles ≈ {0.5, 0.9})
and a complex pair (φ₁ = 1.6, φ₂ = −0.9 → complex poles of modulus ≈
0.949, oscillatory/quasi-cyclical persistence) — at one representative
cell each (SNR 0.5, matching the paper's most-discussed subtle-break
case; level 1σ, r/q ×1.5), rather than the full grid. **Disclosed
modeling choice:** the q-channel break scales the SD of the single
shock w_t — the direct structural analogue of AR1StateDGP's q-break —
because AR(2) has two AR coefficients and there is no unambiguous
single choice of "the state-innovation channel" the way there is for
AR(1); a persistence-type break on φ₁ or φ₂ is a different, separate
question this DGP does not implement.

| parameterization | channel | comparison | detect(a) | detect(b) | ordering |
|---|---|---|---|---|---|
| real roots {0.5, 0.9} | level | raw_cusum vs. innovation_cusum | 0.376 | 0.156 | raw wins ✓ |
| real roots | r (×1.5) | raw_var_cusum vs. arima_var_cusum | 0.660 | 0.968 | ARIMA wins ✓ |
| real roots | q (×1.5) | raw_var_cusum vs. arima_var_cusum | 0.276 | 0.184 | raw wins ✓ |
| complex roots, mod. 0.949 | level | raw_cusum vs. innovation_cusum | 0.928 | 0.760 | raw wins ✓ |
| complex roots | r (×1.5) | raw_var_cusum vs. arima_var_cusum | 0.810 | 0.938 | ARIMA wins ✓ |
| complex roots | q (×1.5) | raw_var_cusum vs. arima_var_cusum | 0.420 | 0.280 | raw wins ✓ |

*Table 6. AR(2)+noise core trichotomy check* (T = 500, n_train = 125,
500 reps, 5% calibrated FAR, seeds disjoint from every other grid —
calibration 110000+, evaluation 210000+;
`paper_assets/exp29_ar2_trichotomy.csv`). ✓ marks agreement with the
AR(1)-derived ordering.

**All six cells are consistent with the pre-registered prediction.** The level
break: raw beats the model-based innovation CUSUM under both
parameterizations, the same "first moments: raw wins" result as §4's
entire AR(1) grid. The r channel: ARIMA-whitened beats raw under both
parameterizations, the same "prewhitening wins" ordering as §5's Table
3 body arenas (φ = 0.95; recall §5's new φ = 0.99 extension shows this
specific ordering is *not* φ-invariant at the *estimated* rung — the
AR(2) check here uses the AR(1) grid's SNR/persistence range, not the
unit-root edge, so it is not in tension with that finding). The q
channel: raw matches or beats ARIMA-whitened under both
parameterizations, the inverted ordering §5 attributes to prewhitening
stripping state-carried signal. This is consistent with the
pre-registered decision rule on the two parameterizations
tested — the only one of this round's three extensions (§5's φ = 0.99,
this section, §9's appendix) to come back clean, though "consistent
with, on two points" is a weaker claim than "confirmed" implies, and is
stated that way deliberately. The qualitative caveat is scope, not
mechanism: one cell per parameterization at one SNR/break-size is an
existence check ("the ordering survives leaving AR(1)"), not a
characterization of *how* it varies across the AR(2) parameter space
the way §5's φ-sweep characterizes AR(1) — a natural next step this
paper does not take.

## 9. Real data (illustrative)

Four FRED series — industrial production (INDPRO), GDP (GDPC1), and
10-year Treasury yields (GS10) (pinned snapshots, 2026-07-11) plus the
unemployment rate (UNRATE)
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
alone would produce (`real_data_eval.py`). A separate, wider 24-month
window (`real_data.py`'s `_summary.csv` export) is used only for the
narrative "caught N months after" delay figures reported in prose
below (e.g. GDP's ~17-month GFC delay, Volcker's 4-month delay) —
those figures are descriptive, not inputs to the 12-month hit/miss
count the permutation p-values above are computed from; a delay figure
outside 12 months (as with GDP's GFC alarm) is therefore compatible
with that same event registering as a miss, not a hit, in the p-value
above. These four series are the
only ones ever pulled for this section. INDPRO, GDP, and GS10 were
chosen for the method's signature cases before looking at their
results (GDP for the 1984Q1 Great Moderation, GS10 for the 1979–82
Volcker episode and the post-2008 ZLB (zero-lower-bound) quieting, `experiments/
CHANGELOG.md` 2026-07-11 design entry). UNRATE was added five days
later (2026-07-16), after the first three series' results were already
known — a genuine fourth look, not an a priori design choice, disclosed
here rather than left implicit. It is included in the four-way
multiple-testing correction below regardless of its own outcome, and no
series was ever dropped, reweighted, or excluded from that correction
based on its results — the selection risk this timing raises is that a
fourth series *could* have been added because the first three looked
promising, not that any series' inclusion was outcome-contingent; we
cannot rule out the former from the record alone. GS10's
three registered events (Volcker 1979-10, the post-2008 ZLB, and a
2022-03 hiking-cycle onset added later) are author-selected from known
monetary-policy/yield-curve episodes, not drawn from a systematized
external date list the way the other three series' NBER-peak and
McConnell–Perez-Quiros (2000) Great-Moderation events are — a real
difference in evidentiary weight the permutation test itself cannot
capture, disclosed here rather than left implicit.

**Look-ahead boundary of the per-segment bootstrap calibration.** Every
segment's alarm threshold is set by a parametric bootstrap whose null
DGP is fit *only* on that segment's training prefix
(`experiments/real_data.py:162`, `null = fitted_null(Y[:NT])`) — never
on the monitored months, and never on the full train+monitor segment.
For INDPRO's GFC segment (train 1998-01..2007-12, monitor
2008-01..2012-12) the two differ substantially (φ = 0.954, q = 0.0046
fit on the training prefix alone vs. φ = 0.892, q = 0.0385 on the full
segment, `experiments/exp23_realdata_lookahead_check.py`) — confirming
the two are not close enough that the distinction would be
immaterial even if it were violated. The existing bit-identical
no-lookahead test (`tests/test_no_lookahead.py`) already covers
filtered estimates, innovations, every feature, and every detector
SCORE on simulated DGPs, but had never been run against the real-data
pipeline's threshold-SETTING step specifically — the thing this
section's causality claim actually depends on. Checked directly:
corrupting the monitored window's values beyond a point t and
re-running the full per-segment procedure (bootstrap null fit,
calibration, all five detectors) on both the original and corrupted
segment, both the calibrated THRESHOLD and every detector's score up
to t are bit-identical for all five real-data detectors
(`lsc_composite`, `lsc_tail_cusum`, `lsc_kalman_cusum`, `raw_cusum`,
`raw_var_cusum`). The real-data pipeline's causality claim holds at
the threshold, not just at the filter.

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
| UNRATE | lsc_composite | 3 | 2/9 | 1 | 0.057 |
| UNRATE | lsc_kalman_cusum | 4 | 4/9 | 0 | 0.0002 |
| UNRATE | lsc_tail_cusum | 6 | 2/9 | 4 | 0.205 |
| UNRATE | raw_cusum | 4 | 4/9 | 0 | 0.0004 |
| UNRATE | raw_var_cusum | 7 | 2/9 | 5 | 0.257 |

*Table 7. Real-data alarm summary at 5% FAR, 120-month training for
INDPRO/UNRATE (monthly) and 180-month/60-quarter training for GDP
(quarterly) (`rd_eval.csv`), for the three NBER/McConnell–Pérez-Quirós-registered
series (INDPRO, GDP, UNRATE). GS10 is treated separately, below, as an
exploratory illustration rather than part of this corrected family —
see "GS10 (exploratory)" later in this section for why. GDP's
raw_cusum fired zero alarms across
all 12 windows, so no permutation test applies — plausibly a
training-window-length artifact rather than a channel-specific null:
GDP's usable history (~300 quarters) is a fraction of the T = 500
simulation grid's horizon, and the 180-month (60-quarter) training
window this section uses for GDP specifically (quarterly frequency, so
120 months of INDPRO/UNRATE's monthly cadence would be only 40 GDP
observations — too few, hence the longer window) is itself short
relative to what the main grid's calibration assumes, so raw_cusum may
simply be underpowered
here at this specific window length rather than genuinely uninformative
about GDP-level shifts in general. UNRATE, added later as a fourth series, shows
the largest raw_cusum/lsc_kalman_cusum association by p-value in the
table (4/4 hits, zero strays, out of 9 peaks in range) — but see the
model-fit discussion below, which complicates a straightforward
reading of that association. Alarm and hit counts are exact given the fixed historical series,
not Monte Carlo estimates; only the permutation p-values carry
resampling uncertainty, from the 20,000-draw test — SE = √(p(1−p)/20000)
≤ 0.0035 across every p-value reported here. All alarms/hits/p-values
in this table are computed on today's revised data; only INDPRO's GFC
and COVID alarms have also been checked against real-time ALFRED
vintages (below) — everything else here, including all of GDP
and UNRATE, is a revised-data illustration, not a real-time-verified
timing claim.*

**Multiple-comparisons correction.** Bottom line first: no real-data
series in this section clears both the multiple-testing bar and the
model-fit bar at once, under either the per-method corrections below
or the joint circular-shift test later in this section (the
multiple-testing problem addressed here is the one Harvey, Liu & Zhu
2016 raise for empirical finance broadly: many candidate "discoveries"
tested against a single-test threshold). The corrected family is 39
tests: Table 7's 14 valid permutation tests (5 methods × 3 series —
INDPRO, GDP, UNRATE, less GDP's zero-alarm raw_cusum cell; GS10 is
reported separately as an exploratory illustration, not part of the
corrected family, since its events are partly author-selected) plus
Table 7b's sensitivity sweep, which re-thresholds the same alarm
machinery and contributes 25 additional distinct tests (5 of its 30
rows duplicate Table 7's INDPRO entries exactly). A Bonferroni
threshold across all 39 (α/39 ≈ 0.00128) leaves only UNRATE's
raw_cusum (p = 0.0004) and lsc_kalman_cusum (p = 0.0002) standing. A
Benjamini–Hochberg FDR procedure at q = 0.05 admits a third — INDPRO's
FAR = 1% sensitivity variant (Table 7b, lsc_composite, p = 0.003,
clearing the rank-3 cutoff ≈0.00385) — but this is a caution about
pooling, not a third confirmed finding: FAR = 1/5/10/20% on the same
series and method are four re-thresholdings of one alarm process, not
four independent looks, overstating how much independent evidence the
sweep contributes. It is also the noisiest single threshold anywhere
in this section by two measures stated elsewhere in the paper but not
cross-referenced here: at FAR = 1% and the n_cal = 200 calibration
budget used throughout §9, the alarm threshold is the 2nd-largest of
200 null draws — exactly the small-order-statistic regime the
Implementation Lesson (§8.5) warns needs a larger calibration budget
for heavy-tailed detectors — and with only 3 alarms against 9 events,
its permutation p-value has just four attainable outcomes (0–3 hits),
the same coarseness disclosed for GS10 below but not, until now, for
this cell. INDPRO's headline association (p = 0.008 at the
baseline FAR = 5%/120-month setting, the number featured in the
abstract) survives neither correction (Bonferroni needs ≤0.00128; its
own BH rank, 4th of 39, needs ≤0.00513). The two associations that DO
survive both corrections are exactly the ones the UNRATE model-fit
discussion below flags as resting on φ-clipped, misspecified windows
(three of UNRATE's four hits) — so no single series clears both bars:
INDPRO clears model-fit but not multiple-testing; UNRATE's raw_cusum
and lsc_kalman_cusum clear multiple-testing but not model-fit. We
report INDPRO/composite at its nominal p = 0.008 throughout, but it
should be read as a suggestive single-series association, not a
family-wise-significant finding. All three corrections treat their
test families as independent, which they are not (shared alarm
machinery across methods and across the sensitivity sweep) — a valid
but conservative approximation, not an exact one. Stepwise procedures
that account for this dependence directly (Romano & Wolf 2005) would
be the principled fix rather than a conservative independence
approximation, but are not run here; we flag rather than apply them,
since the correlation structure across the sensitivity sweep's
re-thresholdings of one alarm process is not itself the kind of
resampling-friendly design their method targets without adaptation.

A genuinely joint test, robust to that cross-method correlation,
confirms the same conclusion: shifting all five methods' alarm months
by the same random amount each draw preserves real cross-method timing
correlation (a common shift is a rigid rotation, unlike independent
per-method redraws) and gives an *exact* permutation p-value per series
(`experiments/exp13c_circular_shift.py`, `exp13d_all_series_circular_
shift.py`; the shift space is small and discrete, so every possible
shift is enumerated, not Monte Carlo sampled; a first attempt at a
joint statistic, `exp13_joint_fwer.py`, collapsed onto the composite's
own marginal test and was abandoned for this circular-shift design
instead). Bonferroni across the three corrected-family series is
α/3 = 0.0167. INDPRO: 7 hits vs. null mean 2.52 (780 shifts, exact),
p = 0.028, does not survive. GDP: 6 hits vs. null mean 1.50 (240
shifts, quarterly), p = 0.0125, nominally clears — but half its hits
are one synchronized co-firing (lsc_composite, raw_var_cusum,
lsc_kalman_cusum all hitting the same 2020-Q2 COVID event), not
independent agreements, so not read as a real finding despite clearing
the threshold. UNRATE: 14 hits vs. null mean 3.57 (780 shifts),
p = 0.0115, nominally clears — but 64% of its hits sit in the same
φ-clipped windows flagged below, including a GFC window where all five
methods fire together in one misspecified window rather than
independently; only the remaining 36% (the well-estimated 2020 window)
reflect a model the paper's own diagnostics would trust. GS10: 4 hits
vs. null mean 1.30 (720 shifts), p = 0.076, reported for context only.
Read together: no series clears both bars here either, the same
conclusion as the per-method corrections above.

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
above). p-values in this section use a per-series, per-method random
seed; every value is within ≤ 0.002 of a shared-seed estimate,
consistent with Monte Carlo noise at 20,000 draws and not a change in
any underlying alarm (`experiments/CHANGELOG.md` records the
reproducibility fix behind this choice of seeding).

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
real-time crisis detection. **Extending the vintage check: UNRATE and
GS10.** This ALFRED protocol was originally run for INDPRO's GFC and
COVID alarms only; it has since been extended to UNRATE and GS10 —
both monthly like INDPRO, and mechanically closer extensions than
quarterly GDP — verified to reproduce every published INDPRO value
exactly before trusting the new series
(`experiments/realtime_check_multi.py`). **UNRATE:** GFC — raw_cusum
alarms at the 2008-09 vintage, raw_var_cusum at 2008-11, the composite
at 2009-01, and the exceedance-indicator detector at 2009-02; COVID —
raw_var_cusum at the 2020-04 vintage, the composite and raw_cusum both
at 2020-05, the exceedance detector at 2020-10 — a similar
ahead-of-or-coincident-with-NBER pattern to INDPRO, spread across a
wider range of vintages than INDPRO's tighter clustering. **GS10:**
one of the three registered episodes turned out to be untestable —
direct ALFRED queries for 1979-10 (and 1990, 1994, 1996-06, as further
checks) return no vintage data; GS10's vintage history is only
available from approximately 1997-01 onward. This narrows an earlier,
weaker claim in this section that ALFRED "serves vintage histories for
all three series," which had only confirmed vintage data exists for
GS10 in general, not that it covers the Volcker episode specifically —
caught here rather than carried forward silently. With that episode
dropped, the two testable GS10 episodes (the zero-lower-bound onset and
the 2022 hiking cycle) give a weak result: only raw_var_cusum alarms on
the 2022 episode (2022-11 vintage, 2022-10 data month), and no method
alarms within the tested window for the zero-lower-bound episode. This
is plausibly a genuine finding — the zero-lower-bound event is a
quieting event, and this paper's own simulations already show quieting
is the hardest signal for every rung to catch (§5, §8) — rather than a
bug, but it is a weak result and is reported as such rather than
oversold. **GDP remains deferred**: it is quarterly, not monthly, so
extending the protocol needs an adapted training/monitoring window
length and a quarterly (not monthly) vintage grid, a materially larger
change than the monthly-to-monthly UNRATE/GS10 extensions above, and
this project's own history (`experiments/CHANGELOG.md`) shows that
rolling-window protocol extensions done quickly have twice introduced
real bugs (a window-anchoring error and a GDP quarter/month units
mismatch) that were only caught by a subsequent dedicated check — a
reason for deferring this specific extension rather than rushing it.
GDP's real-time claims below, and the sensitivity variants of Table 7b,
remain computed on today's revised data, not the vintage a real-time
analyst would have had; read those as illustrative of the detectors'
behavior, not as real-time-verified timing claims.

**GDP (GDPC1, quarterly).** GFC and COVID caught (variance_pressure,
2008Q4 and 2020Q2). The raw variance CUSUM again fires around both
crises
(2009Q2 and 2020Q2, up-arm) with fewer alarms than on INDPRO, though
only the COVID alarm falls inside the 12-month post-event hit window —
2009Q2 is ~17 months after the Dec-2007 NBER peak (2 alarms,
1 hit, p = 0.31) — the same "catches the crisis, misses the clean
association" pattern. The registered Great Moderation event (1984Q1) is an
honest miss: causal rolling monitoring detects the quieting only in
1992Q2 (composite) / 1997Q1 (tail shortfall), because 60-quarter training
windows contain 1970s volatility until the early 1990s. Retrospective
full-sample break dates are not reproducible by honest monitoring.

**GS10 (exploratory).** Unlike INDPRO, GDP, and UNRATE, GS10's three
registered events (Volcker, the 2008 ZLB onset, the 2022 hiking cycle)
are author-selected from known monetary episodes rather than
NBER/McConnell–Pérez-Quirós-dated, and only three events are
achievable — a materially weaker evidentiary basis than the other
three series. GS10 is therefore treated as an illustrative,
exploratory case throughout this section: excluded from Table 7's
corrected family and the multiplicity correction above, its own
permutation p-values reported and discussed here on their own terms,
and its ALFRED vintage and circular-shift results (below) read the
same way. This exclusion was decided on the evidentiary grounds just
stated alone, before computing what effect it would have on any other
series' threshold; one such effect exists and is disclosed rather than
smoothed over — removing GS10 shrinks the joint circular-shift test's
family from four series to three, which loosens the Bonferroni
threshold (α/4 = 0.0125 → α/3 ≈ 0.0167) enough that GDP's own
circular-shift p = 0.0125, an exact tie under the four-series
threshold, nominally clears the three-series one (§9, below) — the
synchronized-co-firing caveat given there already prevents this from
being read as a real finding, but the dependency itself is worth
naming rather than leaving for a careful reader to reconstruct across
two subsections. Volcker regime caught 4
months after the registered 1979-10 event (variance_pressure; raw and
innovation CUSUM the same month); the post-Volcker disinflation
quietings are flagged (1989–1996, shortfall/variance_quiet); the 2008
ZLB event is missed within 12 months. A third registered event, the
2022-03 hiking-cycle onset, is missed by *every* method within the
window this series' final segment actually monitors — 13 months, to
2023-04 (`real_data_date_boundaries.csv`; shorter than the 12-month hit
window used elsewhere in this section only because 2022-03 falls in
the last available segment) — unlike Volcker, a persistent
directional yield rise without a matching shock-variance regime shift
falls outside every detector's reach in what data is available; whether
a longer post-event window would also miss it is not established here.
The raw variance CUSUM also
flags Volcker (1980-02, up-arm) but is the noisiest detector on this
series (9 alarms, 8 stray, 1/3 events hit, p = 0.40) — on a series that
is *all* volatility regime, a raw z² statistic fires on every regime
shift, which is exactly why its event-association washes out. (The
Volcker-catch window itself is well-fit — Ljung-Box p=0.42/0.19,
φ=0.24, unclipped — so unlike UNRATE below, this alarm is not an
artifact of degenerate model fit; the intervening windows bracketing
the shock, 1973–1988, fit far worse, but neither produces the reported
alarm.) GS10's permutation p-values rest
on only 3 registered events, so only four hit-counts (0–3) are achievable
and the resulting p's are far coarser than INDPRO's or UNRATE's (n=9);
read the raw hit/alarm counts as more informative than the exact
p (the 20,000-draw permutation test is, in effect, approximating a
discrete Fisher's-exact-type test with only four attainable outcomes
on this series): lsc_composite 7 alarms/1 of 3 hit/p=0.325,
lsc_kalman_cusum 2/1/0.109, lsc_tail_cusum 4/0/1.000, raw_cusum
2/1/0.104, raw_var_cusum 9/1/0.402 (`rd_eval.csv`, GS10 rows). None of
these approach significance even uncorrected, consistent with GS10's
exploratory rather than confirmatory role in this section.

**Unemployment rate (UNRATE, monthly).** Table 7's raw_cusum and
lsc_kalman_cusum results on UNRATE look like the strongest association
in the real-data application — 4/4 hits, zero strays, p=0.0002–0.0004
— but a model-fit check (`experiments/exp09_real_data_fit_check.py`,
in the spirit of Harvey & Koopman's (1992) diagnostic-checking
program for unobserved-components/state-space models — parameter
estimates and filter residuals should be checked, not just used, on
every window separately, not only on average across windows)
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
`experiments/exp17_unrate_phi_gated.py` excludes the four φ-clipped
windows (segments where the pipeline's clip bound, |φ| at 0.01 or
0.99, actually bound — segments 3, 8, 9, and 10 of 13; segment 9
produces no hit, so it does not appear among the three clipped windows
discussed above that DO produce hits, but it is still excluded here)
from both the hit count and the resampling
universe — a gated test cannot compare a restricted numerator against
an unrestricted denominator without biasing the result. Gating drops
both detectors from 4/9 to 1/6 hits (only the well-estimated 2020
window survives, out of 6 NBER-type events whose window falls inside
the gated universe) against a resampling universe of 540 of the original
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
| Window 180 mo, monitor 36 | lsc_composite | 14 | 2/8 | 0.556 |
| Window 180 mo, monitor 36 | lsc_kalman_cusum | 3 | 2/8 | 0.046 |
| Window 180 mo, monitor 36 | lsc_tail_cusum | 5 | 3/8 | 0.015 |
| Window 180 mo, monitor 36 | raw_cusum | 2 | 2/8 | 0.017 |
| Window 180 mo, monitor 36 | raw_var_cusum | 14 | 1/8 | 0.878 |
| Window 180 mo, monitor 60 | lsc_composite | 5 | 2/8 | 0.138 |
| Window 180 mo, monitor 60 | lsc_kalman_cusum | 2 | 2/8 | 0.019 |
| Window 180 mo, monitor 60 | lsc_tail_cusum | 6 | 0/8 | 1.000 |
| Window 180 mo, monitor 60 | raw_cusum | 2 | 1/8 | 0.268 |
| Window 180 mo, monitor 60 | raw_var_cusum | 8 | 1/8 | 0.722 |

*Table 7b. INDPRO sensitivity to the false-alarm target (1/5/10/20%,
120-month training) and to a longer training window (180 months). The
first 180-month variant (monitor 36, `--tag _w180`) changes the monitor
window and segment count (13→21) at the same time as training length —
a confound disclosed and then isolated by the second 180-month variant
(monitor 60, held at the baseline's own value, `--tag _w180m60`,
n_segments=12, comparable to baseline's 13), which changes training
length alone. Alarm and hit counts are exact given the fixed historical
series, not Monte Carlo estimates; only the permutation p-values carry
resampling uncertainty, from the 20,000-draw test — SE = √(p(1−p)/20000)
≤ 0.0035 across every p-value reported here. Hit denominators are x/9
throughout except the 180-month-training rows (x/8, both variants): the
longer training window pushes monitoring's start date later, dropping
one NBER peak out of the monitored range regardless of monitor length.*

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
on nonstationary real data — in the monitor-36 variant the composite
and the raw variance CUSUM each fire 14 alarms / 21 windows (p = 0.556
and 0.878, both uninformative) — while the level (raw CUSUM),
innovation, and tail detectors stay sane (2–5 alarms, p as low as
0.015–0.046). Because that variant also shrinks the monitor window
(60→36 months) and raises the segment count (13→21), the monitor-60
isolation splits the two effects: raw_var_cusum's alarm rate is
essentially unchanged by holding monitor fixed (8/12 = 0.67/window vs.
14/21 = 0.67/window, p = 0.722), so its blowup is genuinely a
training-window/regime-straddling effect as originally claimed and not
a monitor-window artifact. The composite's alarm rate, by contrast,
drops substantially once monitor is held fixed (5/12 = 0.42/window vs.
14/21 = 0.67/window, still somewhat above baseline's 4/13 = 0.31/
window) — so part, but not all, of the composite's original 14/21
blowup was the monitor-window/segment-count confound rather than the
training-window effect alone; the training-window effect is real for
the composite too, just smaller than the confounded number suggested.
It is the *second-moment* statistic, not the composite machinery, that
is sensitive to a training window long enough to straddle a volatility
regime; training windows must be short enough to be locally stationary
(120 months worked).

## 10. Discussion

**What, then, is the latent layer actually for?** The results invite a
deflationary reading, and we take it seriously rather than deflect it. For
raw *detection power*, the state-space layer is largely redundant: raw
CUSUM owns level shifts in the scalar AR(1)-plus-noise DGP class studied
here (§4) — though at the flagship benchmark cell this ownership is
itself convention-dependent, tying exactly under a one-sided,
known-parameter construction, so the robust claim is the μ∞ ordering
(Proposition 1(a), exact) and its φ-boundary, not an unconditional
numerical advantage under every reasonable convention or a guarantee
that the innovation CUSUM's qualitative "fast or never" shape holds —
it does not at the paper's own flagship arena, where over half of the
paths that survive the adaptation transient go on to alarm later
anyway (§4); ARIMA whitening owns observation-noise
variance and, at true parameters, by the exact ARMA(1,1) equivalence,
*is* the Kalman
innovation filter — for this single statistic the state estimate
contributes nothing beyond it only in that idealized limit, reversing
once parameters must be estimated
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
  buys is dynamics features (persistence, quieting), the speed edge on
  levels conditional on firing, and — once parameters must be estimated
  rather than known — real detection power on the single-statistic
  variance ladder itself (§5, est_kalman_var_cusum vs. arima_var_cusum).
  The division of labor is now measured (grids),
  reduced (raw vs. whitened, the ARIMA and Kalman filters proven identical
  at the innovation-series level only at true parameters — not the
  composite's detection numbers, and not the estimated single-statistic
  case either), and derived (the μ∞ ordering with its φ boundary, though
  the qualitative fast-or-never shape it motivates does not hold cleanly
  at the paper's own φ = 0.95 arena, §4).
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
  tested — 0.550 (raw) vs. 0.500 (combined) at SNR 0.1 (a 0.050 loss,
  1.2 SE), 0.537 (ARIMA) vs. 0.500 at SNR 0.5 (0.037, 0.9 SE), and
  0.560 (ARIMA) vs. 0.470 at SNR 2.0 (0.090, 2.2 SE) — because jointly
  calibrating two statistics to one FAR budget raises the bar each
  individually must clear, and that tax outweighs the benefit of
  channel-agnosticism in this test. (An earlier version of this
  comparison miscalibrated `arima_var_cusum`'s threshold in this script
  specifically — a calibration-seed bug, `experiments/CHANGELOG.md`
  2026-07-23 — which read as a monotonically widening gap; the
  corrected numbers above still show a loss at every SNR but not a
  monotone one, and the SNR-0.5 gap is within 1 SE of noise.) The
  honest recommendation is weaker than "run both, it's free": if
  forced to pick one detector under real channel uncertainty, ARIMA
  was the better single choice in two of the three SNRs tested here;
  running both is only clearly justified if the analyst has a specific
  reason to think the channel mix is skewed toward the regime where
  raw wins (low SNR). That per-SNR comparison is itself an oracle a
  real practitioner facing *unknown* SNR cannot make after the fact —
  the practically relevant question is what a FIXED rule scores,
  pooled across the SNR range tested. Pooling exp14's three per-SNR
  rates under an explicit equal-thirds weighting (the simplest
  defensible default absent a claimed population mix of SNRs, not a
  derived ground truth; `experiments/exp18_pooled_baseline.py`) gives
  always-raw 0.374 (SE 0.015), always-ARIMA 0.526 (SE 0.017), the
  jointly-calibrated combined statistic 0.490 (SE 0.017), and an
  oracle that picks whichever of raw/ARIMA is better *at each SNR*
  0.549 (SE 0.017; by construction this oracle rate is ≥ both fixed
  rules at every one of the three SNRs, confirmed exactly). Two things
  follow: (1) pooled, always-ARIMA beats the jointly-calibrated
  combined statistic by 0.036 (1.5 SE) and always-raw by 0.151 —
  under channel uncertainty with this SNR mix, simply always running
  ARIMA is a stronger fixed rule than either "run both" or "always
  raw", though the margin over "run both" specifically is modest;
  (2) always-ARIMA already captures all but 0.023 of the oracle's
  advantage, so knowing the SNR in advance (on top of not knowing the
  channel) buys little here — the earlier per-SNR "ARIMA wins in two
  of three" reading and the pooled fixed-rule reading agree. Use the
  exceedance-indicator variant under heavy tails and the composite for
  breadth. Calibrate everything on matched nulls at a common FAR (a common
  ARL₀) and report empirical FARs.
- The calibrated-parity protocol itself is a contribution: it exposed
  every failure mode above, and it is what makes the negative results
  informative rather than anecdotal.
- Limitations / future work: a bounded-memory (MOSUM-style) statistic
  fixes the multiple-breaks re-arm failure for level-type second events
  (§7); a windowed *variance* statistic (§7, `exp27`) closes the
  matching gap for variance-type second events (recall_break2 0.00 →
  0.948 on the `var_up_down` scenario) — but naively combining both
  windowed statistics for a channel-unknown mixed sequence (level then
  variance, or the reverse) does not work (§7, `exp36`): joint
  max-calibration holds the variance arm to the mean-shift arm's higher
  bar, collapsing a 93.2% single-arm recall to 4.4% under combination,
  so channel-matched multi-break monitoring under real channel
  uncertainty is still open, likely needing per-arm standardization
  rather than a shared max-threshold; adaptive composite weighting (breadth tax);
  switching-SSM (Kim filter) model layer; formalizing the
  persistence-break mechanisms; a vol-regime reference set for scoring
  the exceedance detector on real data; a plain GARCH(1,1) benchmark
  is now reported over the full 2×2×3 channel×break-size×SNR grid
  (Related Work): it is dominated by raw and/or ARIMA in 11 of 12
  cells (the twelfth, r-channel/subtle/SNR 2.0, is tied with raw
  specifically —
  raw 0.102 vs. GARCH 0.096, inside the ≈1.3pp MC SE), sits at the false-alarm floor specifically at the subtle
  ×1.5 break and moderate-to-high SNR, but is NOT at the floor
  elsewhere — it clears it substantially at the coarse ×3 break on
  both channels (0.19–0.96) and at low-SNR subtle r-channel breaks
  (0.50) — so "GARCH contributes nothing over chance" holds only for
  the originally-checked subset, not the DGP in general; "dominated in
  all 12 cells" is itself a claim about this specific CUSUM-on-
  standardized-residuals wrapper, not about GARCH's fitted path per se
  — exp32/exp37 (Related Work) show that path tracks the true regime
  about as well as the raw/ARIMA rungs' own tracking quantities even in
  the floor cells, so we tested a different (non-CUSUM) alarm rule on
  the same GARCH fit directly (Experiment E): an exceedance-indicator
  CUSUM applied to log(σ²_t) itself (the same construction as the
  heavy-tail `tail_exceedance`/`tail_shortfall` repair, §8.3), in place
  of a CUSUM on GARCH-standardized residuals, same 2×2×3 grid, same
  calibrated 5% FAR, n = 500
  (`experiments/exp42_garch_variance_exceedance.py`). At exp32's own 5
  floor cells (AUC 0.522–0.628, all subtle ×1.5), the exceedance rule
  does *not* rescue detection: its rate stays within 0.01–0.07 of the
  5% FAR target at all five (q SNR 0.1/0.5/2.0: 0.058/0.052/0.120; r
  SNR 0.5/2.0: 0.050/0.088) — nowhere near the AUC's implied
  discriminative power, so "this specific wrapper is underpowered" does
  not generalize to "any wrapper on this fit would do better" at the
  cells where the objection was originally raised. The real, large
  gains from the exceedance construction show up somewhere the floor
  analysis did not point: the two coarse ×3/SNR 2.0 cells, which were
  already well above the FAR floor under the plain CUSUM wrapper (r:
  0.548 → 0.688; q: 0.338 → 0.746) — a genuine improvement, but at
  cells that were never the ones motivating this experiment. A
  break-aware GARCH variant (allowing its own parameters to shift, in
  the spirit of Bai & Perron 1998, 2003) and a full stochastic-volatility
  state-space comparison remain open — the fuller plain-GARCH grid
  rules out the "just use GARCH" objection at every break size and
  SNR tested for this specific wrapper (dominated everywhere), and
  Experiment E now rules out "just use a different alarm rule on the
  same fit" as a fix for the floor cells specifically, without
  resolving whether a purpose-built regime-shift volatility model
  would fare differently. A separate open item: every comparison in
  this paper is between heuristic CUSUM-family statistics on
  differently-transformed series; a known-parameter GLR detector
  (Willsky & Jones 1976, cited in Related Work but not implemented) at
  matched FAR would give an optimality reference point the raw/ARIMA/
  Kalman comparisons above do not — without it, "raw wins on levels"
  identifies which suboptimal statistic is less suboptimal, not how
  much power any of them leaves on the table. Left to future work.
- Scope, held deliberately. The title and framing above are restricted
  to scalar linear-Gaussian AR(1)-plus-noise state-space models; a
  genuine multivariate or regime-switching DGP class through the full
  2×2×3+ grid is a materially larger undertaking than a revision cycle
  should absorb — a new generative model, new detector calibration
  against it, and a repeated grid, not a drop-in extension of the
  existing infrastructure — and is left to future work rather than
  attempted here. The single-cell AR(2) departure already reported
  (§8.6) shows the trichotomy surviving one deviation from AR(1); a
  fuller multivariate or regime-switching treatment would need to show
  the same at scale, and we do not claim it does.

---

## References

Adams, R. P., and D. J. C. MacKay (2007). "Bayesian Online Changepoint
Detection." *arXiv preprint* arXiv:0710.3742.

Andreou, E., and E. Ghysels (2002). "Detecting Multiple Breaks in
Financial Market Volatility Dynamics." *Journal of Applied
Econometrics* 17(5), 579–600.

Andrews, D. W. K. (1993). "Tests for Parameter Instability and
Structural Change with Unknown Change Point." *Econometrica* 61(4),
821–856.

Aue, A., and L. Horváth (2013). "Structural Breaks in Time Series."
*Journal of Time Series Analysis* 34(1), 1–16.

Aue, A., and C. Kirch (2024). "The State of Cumulative Sum Sequential
Changepoint Testing 70 Years After Page." *Biometrika* 111(2),
367–391.

Bai, J., and P. Perron (1998). "Estimating and Testing Linear Models
with Multiple Structural Changes." *Econometrica* 66(1), 47–78.

Bai, J., and P. Perron (2003). "Computation and Analysis of Multiple
Structural Change Models." *Journal of Applied Econometrics* 18(1),
1–22.

Basseville, M., and I. V. Nikiforov (1993). *Detection of Abrupt
Changes: Theory and Application*. Englewood Cliffs, NJ: Prentice-Hall.

Berkes, I., E. Gombay, L. Horváth, and P. Kokoszka (2004). "Sequential
Change-Point Detection in GARCH(p,q) Models." *Econometric Theory*
20(6), 1140–1167.

Bollerslev, T. (1986). "Generalized Autoregressive Conditional
Heteroskedasticity." *Journal of Econometrics* 31(3), 307–327.

Brown, R. L., J. Durbin, and J. M. Evans (1975). "Techniques for
Testing the Constancy of Regression Relationships over Time." *Journal
of the Royal Statistical Society, Series B* 37(2), 149–192.

Chu, C.-S. J., K. Hornik, and C.-M. Kuan (1995). "MOSUM Tests for
Parameter Constancy." *Biometrika* 82(3), 603–617.

Chu, C.-S. J., M. Stinchcombe, and H. White (1996). "Monitoring
Structural Change." *Econometrica* 64(5), 1045–1065.

Frisén, M. (2003). "Statistical Surveillance: Optimality and Methods."
*International Statistical Review* 71(2), 403–434.

Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle." *Econometrica*
57(2), 357–384.

Hamilton, J. D. (1994). *Time Series Analysis*. Princeton, NJ: Princeton
University Press.

Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and
the Kalman Filter*. Cambridge: Cambridge University Press.

Harvey, A. C., and S. J. Koopman (1992). "Diagnostic Checking of
Unobserved-Components Time Series Models." *Journal of Business &
Economic Statistics* 10(4), 377–389.

Harvey, C. R., Y. Liu, and H. Zhu (2016). "...and the Cross-Section of
Expected Returns." *Review of Financial Studies* 29(1), 5–68.

Inclán, C., and G. C. Tiao (1994). "Use of Cumulative Sums of Squares
for Retrospective Detection of Changes of Variance." *Journal of the
American Statistical Association* 89(427), 913–923.

Jacquier, E., N. G. Polson, and P. E. Rossi (1994). "Bayesian Analysis
of Stochastic Volatility Models." *Journal of Business & Economic
Statistics* 12(4), 371–389.

Killick, R., P. Fearnhead, and I. A. Eckley (2012). "Optimal Detection
of Changepoints with a Linear Computational Cost." *Journal of the
American Statistical Association* 107(500), 1590–1598.

Kim, C.-J., and C. R. Nelson (1999). *State-Space Models with Regime
Switching: Classical and Gibbs-Sampling Approaches with Applications*.
Cambridge, MA: MIT Press.

Kim, S., N. Shephard, and S. Chib (1998). "Stochastic Volatility:
Likelihood Inference and Comparison with ARCH Models." *Review of
Economic Studies* 65(3), 361–393.

Kokoszka, P., and R. Leipus (1999). "Testing for Parameter Changes in
ARCH Models." *Lithuanian Mathematical Journal* 39(2), 182–195.

Kokoszka, P., and R. Leipus (2000). "Change-Point Estimation in ARCH
Models." *Bernoulli* 6(3), 513–539.

Leisch, F., K. Hornik, and C.-M. Kuan (2000). "Monitoring Structural
Changes with the Generalized Fluctuation Test." *Econometric Theory*
16(6), 835–854.

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

Ramdas, A., P. Grünwald, V. Vovk, and G. Shafer (2023). "Game-Theoretic
Statistics and Safe Anytime-Valid Inference." *Statistical Science*
38(4), 576–601.

Romano, J. P., and M. Wolf (2005). "Stepwise Multiple Testing as
Formalized Data Snooping." *Econometrica* 73(4), 1237–1282.

Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence
Intervals*. New York: Springer-Verlag.

Stock, J. H., and M. W. Watson (2002). "Has the Business Cycle Changed
and Why?" *NBER Macroeconomics Annual* 17, 159–218.

Wald, A. (1947). *Sequential Analysis*. New York: Wiley.

Willsky, A. S., and H. L. Jones (1976). "A Generalized Likelihood Ratio
Approach to the Detection and Estimation of Jumps in Linear Systems."
*IEEE Transactions on Automatic Control* 21(1), 108–112.

Xie, L., S. Zou, Y. Xie, and V. V. Veeravalli (2021). "Sequential
(Quickest) Change Detection: Classical Results and New Directions."
*IEEE Journal on Selected Areas in Information Theory* 2(2), 494–514.

---

## Appendix A. Reproducibility

`make all` regenerates every table and figure from pinned seeds
(Python 3.14, statsmodels/hmmlearn; `make fred` / `make realdata` /
`make realtime` for the data applications, snapshots under `data/`).
The pack includes pinned-seed scripts spanning `exp02` through `exp52b`
(45+ scripts as of this draft, growing over the project's revision
history rather than a fixed set): the earliest pre-registration checks
(`exp02`-`exp06`, discussed by name above and in the Introduction —
the SNR-sweep and persistence-change hypotheses, the exp05 robust-
variance registration, and the theory-verification script cited in §4),
the
ARMA(1,1)-equivalence check, the `grid_v5`–`grid_v8` q-break/φ-sweep/
local-level/φ×q grids, the ARL₀/ARL₁ table, the PELT localization
benchmark (§8.5), the real-data fit and CUSUM-ablation checks (§4, §9),
the multiple-comparisons circular-shift tests across all four real-data
series (§9), the mixed-channel and pooled-baseline checks (§10), the
GARCH(1,1) benchmark (Related Work), the AIC order-frequency and
UNRATE φ-gated checks (Appendix B, §9), the paired-SE reconstruction
for Table 3a (§5), the composite-on-ARIMA ablation (§5), the paired-SE
correction to Table 2b's negative gaps (`exp43`, Experiment F), and —
added in
later revision rounds — the estimated-Kalman variance rung
(`exp44`), the AR(2)+noise robustness check (`exp29`, Table 6), the
smoothed-ARIMA and GARCH-exceedance composite variants (`exp41`,
`exp42`), the symmetric FAR recalibration (`exp46`), the Proposition
1(b) idle-bound audit (`exp52`), and the flagship detection-vs-horizon
curve (`exp52b`). All are
pinned-seed and join the existing grids draw-for-draw; every number
these scripts produce is cited at its point of use in §4–§10 and
tabulated in Appendix C. The order in which these were added, and the
bugs found and fixed along the way — including a window-anchoring bug
and a units-mismatch bug in the circular-shift permutation test (§9)
and a stale calibration seed in the mixed-channel check (§10) — are
recorded in `experiments/CHANGELOG.md` in the linked repository;
only the corrected, current numbers are reported in the body. 121 tests
(as of this draft; the suite grows as new checks are added, e.g. the
AR(2) DGP tests added for §8.6) include
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
and Outcome A of the varbench claim-adoption decision rule (three
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
with `experiments/THEORY.md` as the long-form companion. The complete
source, pinned data snapshots, and seed configuration constitute the
replication package and are publicly available now, not deferred to
acceptance: https://github.com/789wethan-wq/lsc.
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
A fourth round (2026-07-23) added the exp15 full 2×2×3 GARCH grid,
`exp18` (pooled exp14 baselines), `exp19` (paired SE for Table 3a), and
`exp20` (composite-on-ARIMA, `lsc/models/arima_model.py`) — full test
suite still green (98 passed) and `exp19` self-verifies by reproducing
all 12 published Table-4 detect_rate cells exactly from the original
config and seeds, but this round has NOT yet been run through the
fresh-clone reproducibility-check protocol the other rounds above have
(`exp20`'s ~8.8-hour wall-clock, driven by a handful of pathologically
slow ARIMA fits documented in `experiments/CHANGELOG.md`, made a full
re-verification pass impractical within this session) — flagged here
rather than silently assumed to meet the same bar.

**A cross-environment reproduction.** Every check above is
a fresh clone on the *same machine* — it tests that pinned seeds
reproduce exactly given repeated setup, not that they survive a change
of operating system, C library, or Python interpreter. We label this
distinction explicitly rather than let a same-machine check stand in
for a stronger claim: this is a **cross-environment** reproduction, run
by the author's own tooling in a different container — not a
**third-party** one, where an independent person clones the repo cold
and runs it following only the README with no other guidance. No such
third-party check has been performed; if a labmate, advisor, or
collaborator becomes available before submission, that is the more
credible addition and should supersede, not just supplement, this
section.

Setup (`Dockerfile.repro`, `.dockerignore`, both committed): the
package's own committed `paper_assets/` snapshot, `pyproject.toml`,
`lsc/`, `tests/`, `experiments/`, `configs/`, and `Makefile` are copied
into a `python:3.12-slim` (Debian, glibc, linux/arm64) container —
deliberately different from the author's development environment
(macOS, Homebrew Python 3.14) on OS, C library, and interpreter minor
version — with dependencies installed *only* from the pinned
`pyproject.toml`, then `make all` (everything except `fred`, which
needs network) is run with no manual intervention beyond that single
command.

*This exercise found and fixed three real reproducibility bugs before
it produced a clean run* — exactly the kind of gap a same-machine
check cannot surface, since the author's own long-lived development
venv silently papers over all three:

1. `pyproject.toml` declared `requires-python = ">=3.11"`, but pinned
   `numpy==2.5.1` requires Python ≥3.12 — the first build (against
   `python:3.11-slim`) failed outright on `pip install`. Fixed by
   correcting the declared minimum to `>=3.12` (the true constraint)
   rather than by pinning an older numpy.
2. `experiments/m2_param_recovery.py`'s `DataFrame.to_latex()` call
   requires `jinja2` internally (pandas 3.x routes `to_latex` through
   its `Styler` machinery) — undeclared in `pyproject.toml`, and the
   build only ever "worked" on the author's machine because `jinja2`
   happened to be present from an unrelated, unrecorded install (`pip
   show jinja2` on the host: `Required-by:` empty — nothing in the
   pinned dependency graph actually needs it). Fixed by adding
   `jinja2==3.1.6` to `pyproject.toml`'s dependencies.
3. The first `Dockerfile.repro` piped `make all` into `tee`, whose exit
   code (always 0) masked a real `make` failure — the container
   reported success while `make all` had actually failed on a missing
   `paper_assets/` directory (`.dockerignore` excluded it, so the
   image had nowhere to write). Both fixed: `paper_assets/` is now
   copied into the image (matching what a real `git clone` provides,
   which `make all` then overwrites in place), and the run redirects
   directly to a log file instead of piping through `tee`, so a `make`
   failure now surfaces as the container's real exit code.

*Result, after those fixes: a clean run, with the substantive findings
intact but literal byte-identity NOT achieved everywhere.* The
container ran to completion (`make all`, exit 0) and its
`paper_assets/` was diffed file-by-file against the host's committed
copy (excluding `m2_param_recovery.csv`, already documented as
BLAS-thread-order nondeterministic on a single machine, §Reproducibility
lesson 11). Of 23 `*_results.csv` grid outputs, 18 are byte-identical;
the other 5 show `detect_rate` differences of at most 0.006 (3 of 500
replications) concentrated in methods that fit a model by MLE
(`lsc_composite`, `lsc_state_cusum`, ARIMA-based rungs) — never in
`raw_cusum` or other closed-form statistics with no iterative fit,
which are exact in every file checked. `mean_delay_*` columns (also
MLE-dependent, continuous rather than discrete) differ by relative
~1e-4–1e-5. The mechanism is almost certainly a different BLAS/LAPACK
backend under an identical pinned `numpy`/`statsmodels` version
(Accelerate/vecLib on the author's macOS vs. the container's OpenBLAS
manylinux wheel) nudging an iterative optimizer's last bit of
convergence — the same class of nondeterminism lesson 11 already
documents for repeated runs on one machine, evidently larger, though
still small, across a genuine change of numerical library.
`experiments/exp29_ar2_trichotomy.csv` (the AR(2) check, §8.6) and
`configs/grid_v9_r_phi99.yaml`'s output (the φ=0.99 r-channel ladder,
§5) — the two genuinely new, non-cached computations this round added
— are both **byte-identical** end-to-end, the strongest single piece
of evidence here: a full DGP simulation → MLE fit → CUSUM →
calibration → detection-rate pipeline reproduced to the last bit across
OS, libc, and Python minor version. (`exp28_known_param_phi99.csv`
shows only last-digit CSV string-formatting differences on values its
own `_already_done` cache reused unchanged from the copied
`paper_assets/` rather than recomputing — a serialization artifact, not
a value difference, and disclosed rather than silently folded into the
"byte-identical" count above.) One further honest note: mid-run, host
CPU load spiked to 50 (unrelated processes on the author's own
machine, not this container) and the run slowed by roughly 20–30x for
about ninety minutes before easing — a real-world confound of running
this check on a shared, live development machine rather than a
dedicated one, noted for transparency though it affects wall-clock
time, not correctness.

**Honest summary.** The claim this section supports is narrower than
"byte-for-byte reproduction survives any environment": closed-form
statistics do, to the last bit; MLE-dependent ones reproduce the
substantive finding (detection rates match to within 0.006, an order
of magnitude below any effect size this paper reports as meaningful)
but not literal bit-identity, because floating-point optimizer
convergence is not portable across BLAS backends even with every
package version pinned. That is a real, disclosed limit on the
reproducibility claim, caught only by actually crossing environments —
and, separately, the exercise surfaced and fixed two genuine dependency
bugs (`requires-python`, missing `jinja2`) that a same-machine check
had never been positioned to find.

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

As noted in §4, Proposition 1(a) restates a standard LTI step-response
computation (Harvey 1989; Hamilton 1994, ch. 13); Proposition 1(b) is a
standard exponential-martingale/Wald-type tail bound (Wald 1947;
Siegmund 1985; Basseville & Nikiforov 1993). Both proofs are given below
for completeness and to pin down the exact constants (ρ, μ∞, θ\*) used
elsewhere in the paper.

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
runs ≈12–21% conservative at 3σ (§4). If Δ ≤ k the drift is
nonpositive and the never-detect bound applies verbatim. ∎

**ARMA(1,1) equivalence of the whitened rungs (§5).** This is the
standard signal-plus-noise reduced form of structural time-series theory
(Harvey 1989; Hamilton 1994, ch. 13); we spell out the derivation here
only to pin down the two identities (σ_ε² = F, θ = ρ) that the rest of
the paper relies on. Applying the AR
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
| Proposition 1(b) bound, 3σ cells audited (grid_v1 + grid_v6_phisweep) | 30/30 empirically idle: 24 vacuous (≥1, incl. all 12 at the paper's own φ=0.95); 6 (all φ=0.99) below the 1/500 resolution floor | exp52 |
| Flagship cell (SNR 0.5, φ=0.95, 3σ) post-transient alarm rate among transient-survivors | 54.3% (477/500 survived to T*=13 post-break obs. without alarming); curve gain h=60→250: +0.254 (lsc_kalman_cusum) vs. +0.698 (raw_cusum) — the pre-registered prediction that the innovation-CUSUM curve gains <0.05 past the transient is falsified | exp52b |
| **r**-break ×1.5: latent composite, T=500 | 0.82 / 0.87 / 0.91 (SNR 0.1/0.5/2.0) | grid_v1 |
| **r**-break ×1.5: raw rung, T=500 | 1.00 / 0.56 / 0.10 (SNR 0.1/0.5/2.0) | grid_v4_varbench |
| **r**-break ×1.5: ARIMA rung, T=500 | 0.90 / 0.94 / 0.87 (SNR 0.1/0.5/2.0) | grid_v4_varbench |
| **r**-break ×1.5 t₅ (raw/ARIMA/composite), SNR 0.5 | 0.43 / 0.74 / 0.16 | grid_v4_varbench, v2_misspec |
| ARMA≡Kalman innovation ρ̄ (estimated / true params) | 0.99 / 1.000 (max\|Δ\|≈10⁻⁹) | exp07 |
| Paired-SE correction to Table 2b's 6 negative gaps (Experiment F) | 1 of 6 genuinely within noise (r×1.5/SNR0.1, 1.4 SE); other 5 are 2.1–3.0 paired SEs from zero, though all \|gap\| ≤ 0.018 at rates ≥0.95 | exp43_table2b_paired_se |
| AR(2)+noise core trichotomy check, real/complex roots, 6 comparisons | all 6 agree with the AR(1) grid's ordering (raw wins level; ARIMA wins r-channel; raw wins q-channel) | exp29 (Table 6) |
| Smoothed (two-sided) ARIMA state proxy vs. one-step, composite-on-ARIMA gap (Experiment D) | closes part of the gap at SNR 0.1 (0.226→0.382, still short of Kalman's 0.818); no change at SNR 0.5/2.0 (identical threshold/rate to one-step) | exp41 |
| GARCH exceedance-indicator alarm rule vs. plain CUSUM wrapper (Experiment E) | no rescue at exp32's 5 floor cells (stays within 0.01–0.07 of 5% FAR); real gains at 2 coarse ×3/SNR 2.0 cells (r: 0.548→0.688; q: 0.338→0.746) not originally motivating the check | exp42 |
| est_kalman_var_cusum vs. arima_var_cusum, 12-cell r/q grid (estimated params) | 11/12 cells strictly better, 1 exact tie at ceiling, 0 worse; decisive at q ×3/SNR 0.1: 0.524 vs. 0.262 (Δ=+0.262, 12.0 paired SEs) | exp44 |
| Composite-on-Kalman vs. composite-on-ARIMA, r ×1.5 (SNR 0.1/0.5/2.0) | 0.818/0.868/0.910 (Kalman) vs. 0.226/0.416/0.632 (ARIMA) — decisive away from ceiling | exp20_composite_on_arima |
| Composite-on-ARIMA vs. its own single ARIMA-CUSUM feature, r ×1.5 | 0.226/0.416/0.632 (composite) < 0.900/0.942/0.868 (single feature) — composite is counterproductive on ARIMA inputs | exp20_composite_on_arima |
| **q**-break ×1.5: raw rung, T=500 | 0.09 / 0.21 / 0.23 (SNR 0.1/0.5/2.0) | grid_v5_qbreak |
| **q**-break ×1.5: ARIMA rung, T=500 | 0.03 / 0.10 / 0.16 (SNR 0.1/0.5/2.0) | grid_v5_qbreak |
| Pre-registered decision rule (§5) resolved | Outcome B2 (r-channel-specific) | §5, CHANGELOG |
| φ sweep: μ∞ sorts innovation-CUSUM detection | Spearman 0.9415, n=24 (4 φ × 3 SNR × 2 shifts); stratified permutation null (shuffle φ-pairing within SNR×shift, n_perm=20,000) rejects at p<0.00005 | grid_v6_phisweep, exp12 |
| Innovation CUSUM 3σ escape (φ=0.5 / 0.99) | 0.98–1.00 / 0.30–0.63 | grid_v6_phisweep |
| Local-level 3σ level detect (all methods) | ≤ 0.15 (≈ FAR) | grid_v7_llevel |
| Local-level ×1.5 variance (raw / ARIMA rung) | 0.06 / 0.58–0.84 | grid_v7_llevel |
| ARL₀ at 5% window-FAR (L=375) | ≈ 7300 obs | arl_table |
| Symmetric FAR recalibration, 8 detectors × 3 arenas (bisect threshold vs. fresh 2000-null block) | 24/24 far_fresh_matched in [4.5%,5.5%]; level-3σ raw-vs-lsc_kalman gap moves by ≤0.036 at any SNR (narrows at 1 of 3, widens slightly at the other 2) | exp46 |
| φ×q: ×1.5 raw edge, φ-swept = SNR-swept | Δ 0.11=0.11 (SNR 0.5) | grid_v8 vs grid_v5 |
| φ×q: subtle Δ at φ→0 / Spearman(amp,Δ) | 0.00 / 0.83; ×3 falsified (−0.60) | grid_v8_phiqbreak |
| φ×q Table 3a: paired vs. conservative SE(Δ), 12 cells | paired 0.014–0.025 vs. old worst-case bound 0.032 (all cells reproduced exactly); φ=0.95-vs-0.99 subtle-Δ gap now ≈1.5 SE (was ≈0.8 SE) | exp19_paired_se_grid_v8 |
| Variance ×1.5 across T = 200/500/2000 | 0.11 / 0.87 / 0.99 | grid_v2_T |
| t₅ collapse and repair (×1.5) | 0.16 → 0.75 (tail_cusum) | grid_v2_misspec, v3c |
| Quieting ×⅔ (only tail_cusum) | 0.41 / 0.33 | grid_v3c |
| Persistence-down, best anywhere | 0.33 (SNR 2.0) | grid_v1 |
| Multi-break: raw second-event recall (level→level) | 0.00 | exp04 |
| Composite level→var second event | 0.60 (F1 0.63) | exp04 |
| Windowed-CUSUM fix, level→level 2nd event (raw / innovation) | 0.00→0.68 / 0.01→0.23 | exp04 |
| Windowed-CUSUM fix, level→var / var→var 2nd event (mean-shift statistic) | no improvement (≈0.00): mean-shift only | exp04 |
| Windowed VARIANCE-ratio fix, var→var 2nd event (`windowed_raw_var`, n=500) | recall_break1=0.932, recall_break2=0.948, F1=0.958, precision=0.997 — closes the gap above | exp27_windowed_variance |
| Naive combined windowed statistic, mixed orderings (level→var / var→level, n=500) | var→level windowed_raw_var alone 0.932 recall_break1 collapses to 0.044 under naive max-combination with windowed_raw_cusum — joint calibration holds the variance arm to the mean-shift arm's higher bar; negative result, not a fix | exp36_combined_windowed_multibreak |
| q-channel known-vs-estimated gap: order-selection vs. coefficient-noise decomposition, 6 cells | order-selection component within ±0.03 of zero in 5/6 cells (sixth: −0.032); ≈ all of the gap (up to +0.586) is MLE coefficient noise on the correctly-specified order | exp30_order_known_arima |
| Flagship cell (SNR 0.5, φ=0.95, 3σ), missing one-sided/estimated corner | 0.636 (two-sided/estimated 0.554, two-sided/known 0.970, one-sided/known 0.990 already reported) — parameter knowledge (+0.416) dominates sidedness (+0.082) | exp10_cusum_ablation |
| Table 4 Kalman-vs-ARIMA composite gaps: paired per-replicate SE, 3 cells | r×1.5/SNR0.1: Δ=0.594, z=25.96; r×1.5/SNR2.0: Δ=0.278, z=12.99; q×3/SNR0.1: Δ=0.190, z=8.93 — paired SE tighter than the independence bound in all three | exp35_composite_paired_se |
| GARCH conditional-variance path vs. true regime, 5 floor cells (AUC) | GARCH 0.522–0.628, highest of the three rungs in every floor cell (raw 0.516–0.591, ARIMA 0.514–0.612) — favors underpowered-wrapper over structural-mismatch explanation | exp32_garch_mechanism |
| GARCH oracle (true-break-time refit), post-break z² elevation, 12 cells | oracle −0.14 to +0.06 (null-level); plain single-fit +0.16 to +6.4 — perfect break-knowledge reveals no signal beyond what the plain fit already shows | exp37_garch_oracle_break_aware |
| Full r-channel φ-sweep (φ = 0.5/0.8/0.95/0.99), subtle ×1.5, raw−ARIMA advantage | 0.586 → 0.154 → 0.096 → reverses to 0.396 at φ=0.99 (SNR 0.1); sharp reversal specific to the unit-root edge, not gradual | r_phi_sweep_full |
| ALFRED vintage extension, UNRATE (GFC / COVID) | raw_cusum 2008-09 / 2020-05; raw_var_cusum 2008-11 / 2020-04; composite 2009-01 / 2020-05; tail_cusum 2009-02 / 2020-10 | rd_realtime_unrate |
| ALFRED vintage extension, GS10 | Volcker (1979-10) untestable — no ALFRED vintage data before ≈1997-01; hike2022 caught by raw_var_cusum only (2022-11 vintage); zero-lower-bound onset: no method alarms in window | rd_realtime_gs10 |
| PELT localization at FAR-matched 5%, level 3σ | 0.83–0.92 (vs. causal raw CUSUM 0.97–0.99) | exp08_pelt |
| PELT localization at FAR-matched 5%, variance ×1.5/×3 | 0.00–0.20 (vs. dedicated raw variance-CUSUM 0.09–1.00) | exp08_pelt |
| ICSS localization at FAR-matched 5%, variance r/q ×1.5/×3 (n=500) | 0.00–1.00, clears PELT's ceiling but dominated by causal raw_var_cusum in 11/12 cells (0.996→0.102 vs. ICSS 0.74→0.00 at r ×1.5 over SNR 0.1→2.0) | exp25_icss |
| INDPRO permutation p (composite) | 0.008 (uncorrected — does not survive Bonferroni (α/39≈0.00128) or its own BH-FDR rank across the combined 39-test family of Table 7 + Table 7b, GS10 excluded as exploratory; §9) | rd_eval |
| Circular-shift joint test, INDPRO (5 methods, total hits) | 7 vs. null mean 2.52, max 10 (780 shifts, exact) — p=0.028, does not survive Bonferroni (α/3≈0.0167); §9 | exp13c_circular_shift |
| Circular-shift joint test, GDP | 6 vs. null mean 1.50, max 6 (240 shifts, exact) — p=0.0125, nominally clears the 3-series threshold (α/3≈0.0167) but half the hits are one synchronized co-firing, not read as a real finding; §9 | exp13d_all_series_circular_shift |
| Circular-shift joint test, GS10 (exploratory, not in corrected family) | 4 vs. null mean 1.30, max 6 (720 shifts, exact) — p=0.076, reported for context only; §9 | exp13d_all_series_circular_shift |
| Circular-shift joint test, UNRATE | 14 vs. null mean 3.57, max 16 (780 shifts, exact) — p=0.0115, nominally survives, but 64% of hits (9/14) sit in the same φ-clipped windows already flagged in §9; §9 | exp13d_all_series_circular_shift |
| GARCH(1,1) benchmark, full 2×2×3 grid (channel × ×1.5/×3 × SNR, n_reps=500) | r ×1.5: 0.498/0.098/0.096; r ×3: 0.962/0.708/0.548; q ×1.5: 0.038/0.066/0.098; q ×3: 0.186/0.344/0.338 (SNR 0.1/0.5/2.0) — floor only at ×1.5 + moderate/high SNR; dominated by raw/ARIMA in 11 of 12 cells (twelfth, r ×1.5/SNR 2.0, is a statistical tie: raw 0.10 vs. 0.096, inside the ≈1.3pp MC SE) | exp15_garch_benchmark |
| AIC order-selection frequency at φ=0.95 (SNR 0.1/0.5/2.0), n=500/cell | (1,0,1): 12.0%/9.4%/7.8%; (1,0,0) dominant at SNR 0.1 (43.0%), (0,1,1) dominant at SNR 0.5-2.0 (64.8%/68.6%) | exp16_aic_order_frequencies |
| UNRATE φ-gated permutation test (raw_cusum, lsc_kalman_cusum) | 4/9→1/6 hits after excluding the 4 clipped-φ windows from both numerator and resampling universe (540/780 months, 6 of 9 events fall inside the gated universe); p=0.1474 (both), vs. ungated 0.0002-0.0004; §9 | exp17_unrate_phi_gated |
| Mixed-channel (raw+ARIMA run jointly, unknown channel), SNR 0.1/0.5/2.0 | combined loses to single-better detector at every SNR: 0.500 vs 0.550 / 0.500 vs 0.537 / 0.470 vs 0.560 (calibration-seed bug fixed 2026-07-23; see CHANGELOG) | exp14_mixed_channel |
| Pooled fixed-rule baselines (equal-thirds over SNR 0.1/0.5/2.0), unknown channel AND SNR | always-raw 0.374, always-ARIMA 0.526, combined 0.490, oracle-best-per-SNR 0.549 — always-ARIMA nearly matches the oracle (gap 0.023) | exp18_pooled_baseline |
| GFC real-time | 2008-09 data, known 2008-12 | rd_realtime |
| COVID real-time | data 2020-03, ~2 mo before NBER | rd_realtime |
