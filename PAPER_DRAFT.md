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
exponentially small bound) and verify numerically. For second-moment and
dynamics changes the ordering reverses — though a whitening ladder (the
same variance CUSUM on raw data, ARIMA residuals, and Kalman innovations)
shows the edge is *prewhitening under autocorrelation*, not the latent
state estimate itself: a raw-data variance CUSUM given identical
calibration detects the subtle (×1.5) break only when observation noise
dominates (0.996 at SNR 0.1) and falls to chance as the latent signal
grows (0.560, 0.102 at SNR 0.5, 2.0), whereas the same statistic on
prewhitened residuals detects it at every SNR (ARIMA 0.90/0.94/0.87,
tracking the latent composite's 0.82/0.87/0.91). The diagnostics further
survive heavy tails via an exceedance-indicator variant and, via a
shortfall CUSUM, detect variance *quieting* that level-oriented methods
miss. A real-data application
(industrial production, GDP, Treasury yields) reproduces the profile:
every alarm attributes to a second-moment feature, association with NBER
reference dates is significant (permutation p = 0.008), and real-time
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
  for first moments at every SNR — and we can prove why (fast-or-never
  theorem). For second moments it is right, but a whitening ladder (raw /
  ARIMA / Kalman, same variance CUSUM) localizes *why*: the advantage
  over raw data is prewhitening under autocorrelation — decisive at high
  SNR, where the latent signal masks the observation-noise change and the
  raw detector sits at chance — and the latent state estimate itself adds
  little once the observations are whitened (ARIMA residuals suffice).
- Contribution 3 (method): a tail-robust exceedance-indicator CUSUM that
  preserves the second-moment advantage under heavy-tailed noise, found
  via two documented failed designs.
- Contribution 4 (application discipline): attribution, permutation
  tests, sensitivity, pinned data snapshots, and real-time vintages for
  the real-data claims — including a self-correction on the headline GFC
  timing.
- Honest-outcome framing throughout: three pre-registered hypotheses
  were falsified; every post-hoc change is logged (CHANGELOG) and the
  failures are reported as findings.

**Related work.** The paper sits at the intersection of four literatures.
Sequential change detection descends from Page's (1954) CUSUM and the
quickest-detection tradition (Lorden 1971; Moustakides 1986); we use the
CUSUM as the common statistic across information sets rather than proposing
a new stopping rule. Regime-switching models (Hamilton 1989; Kim–Nelson
1999) offer a state-aware alternative whose regime probabilities we include
as a benchmark and find saturate under calibration on nonstationary data.
The empirical target of the second-moment results is the Great Moderation
volatility decline (McConnell–Perez-Quiros 2000; Stock–Watson 2002), which
motivates the variance and quieting scenarios. Finally, offline changepoint
methods (PELT, Killick et al. 2012; and the wider changepoint literature)
solve a retrospective segmentation problem; our monitoring is strictly
causal and calibrated to a false-alarm rate, so the comparison is to online
detectors on matched nulls. What is new here is not any single detector but
the calibrated-parity harness that makes latent-state and raw-data
detectors directly comparable, and the whitening ladder that decomposes the
latent layer's second-moment advantage into a prewhitening component and a
(negligible) state-estimation component.

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
matching in a 100-observation window.

## 3. Simulation design

Arenas: AR(1) latent state (φ = 0.95) at spec-SNR (stationary state
variance / observation variance) ∈ {0.1, 0.5, 2.0}; a local-level
(random-walk state) arena is retained as a documented degenerate case
(§8.4). Breaks at mid-sample: level shifts (0.5, 1, 3 σ_ref), logistic
ramps, observation-noise scale changes (×1.5, ×3, and ×⅔ quieting), and
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

**The ladder (detection rate at T = 500, 5% calibrated FAR; MC SEs ≤ 0.02
in `paper_assets/ladder_table.csv`).**

| break | rung | SNR 0.1 | SNR 0.5 | SNR 2.0 |
|---|---|---|---|---|
| variance ×1.5 | raw | **1.00** | 0.56 | **0.10** |
|               | ARIMA | 0.90 | 0.94 | 0.87 |
|               | latent | 0.82 | 0.87 | 0.91 |
| variance ×3   | raw | 1.00 | 1.00 | 0.85 |
|               | ARIMA | 0.98 | 1.00 | 1.00 |
|               | latent | 0.99 | 0.99 | 0.98 |

**Reading the ladder.** The subtle ×1.5 break is the discriminating case.
The *raw* rung is strongly SNR-dependent — it falls monotonically from
0.996 (SNR 0.1) through 0.560 (SNR 0.5) to 0.102 (SNR 2.0, within 5 pp of
the 6.0% empirical FAR, i.e. chance). The mechanism is transparent: as
SNR rises the latent state's variance dominates the marginal variance of
Y, so a ×1.5 change in the (now-small) observation-noise component is a
shrinking fraction of total variance and is masked by state-driven
autocorrelation. This overturns the scoped claim — earlier framed against
"level-oriented detectors standard in this literature" — that a raw-data
variance detector must sit at chance: it does not, *when observation
noise dominates*. But prewhitening removes exactly the autocorrelation
that masks the break: the *ARIMA* rung is essentially flat across SNR
(0.90 / 0.94 / 0.87) and tracks the *latent* rung (0.82 / 0.87 / 0.91)
step for step — the state estimate adds little the ARIMA residuals do not
already provide. The advantage over raw is therefore *prewhitening under
autocorrelation*, not latency or state estimation per se (the recipe in
§10 is revised accordingly, and the fast-or-never first-moment result of
§4 is untouched — that concerns the state CUSUM, not the variance CUSUM).
At the coarser ×3 break every rung is at or near ceiling; the ladder only
separates in the subtle regime. This is the pre-registered **Outcome C**
of the decision rule logged in `experiments/CHANGELOG.md` (Outcomes A and
B, which required the raw rung to be uniformly at chance or uniformly
within 10 pp of the composite, are both falsified by the SNR dependence).
Attribution: the z²/e² variance-pressure arms drive the alarms at every
rung; on real data the same features drive every crisis alarm (§9).

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
detectable — but only by the exceedance detector (§8.3): 0.41/0.33, all
other methods at chance.

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
any distribution) — the raw statistic separates near-perfectly, but dies
inside the composite: a max-over-features composite rewards
break-to-null-IQR ratio, and bounded increments cannot reach the ratios
unbounded features set the threshold with. (iii) The same statistic as a
*standalone* calibrated detector (up-arm k = 0.05, down-arm k = 0.02; k
chosen on non-evaluation seeds, procedure logged): variance ×1.5 at 0.87
Gaussian / 0.75 t₅ (repairing 0.16), ×3 at ~1.0 with ~37-obs delay under
both distributions, and the first successful quieting detection (×⅔:
0.41/0.33). Both headline rates fell 3–5pp short of the pre-registered
bars — reported as such.

**8.4 Protocol lessons (each cost us a wrong result before it was
fixed).** Random-walk-state arenas are degenerate for level-break
ranking; plain-HMM regime probabilities saturate and cannot be
FAR-calibrated on nonstationary data; probability-scale scores need
log-odds; EM needs persistent-initialization restarts; composite features
must be standardized per-time-point, not pooled; order-statistic
thresholds have Beta(n+1−k, k) noise regardless of distribution, so
heavy-tailed detectors need larger calibration budgets.

## 9. Real data (illustrative)

Three FRED series, pinned snapshots (2026-07-11), rolling causal
monitoring (train 120 months / monitor 60), per-segment parametric
bootstrap calibration at 5% FAR per window, alarms attributed to the
feature that crossed.

**Industrial production (INDPRO, 1948–2026).** Composite alarms: 2008-09
and 2020-04 (both variance_pressure), 1990-12 (variance_quiet), 1969-08
(variance_quiet; within the false-alarm budget and reported as such).
Hits 3/9 NBER peaks within 12 months, 1 stray vs 0.7 expected;
permutation p = 0.008 (innovation CUSUM p = 0.018; raw CUSUM 1 hit,
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
  dynamics that raw-data detectors miss once the latent signal masks
  them. The whitening ladder (§5) sharpens the claim: the second-moment
  advantage over raw data is *prewhitening under autocorrelation*, which
  ARIMA residuals capture as well as the Kalman state — the state layer's
  distinctive contribution is dynamics (persistence, quieting) and the
  fast-or-never speed edge on levels, not variance detection per se. The
  division of labor is now both measured (grids), laddered (raw / ARIMA /
  latent), and derived (fast-or-never).
- Practical recipe: run raw CUSUM for levels; for scale changes, *whiten
  first* (ARIMA residuals or Kalman innovations) and run the variance
  CUSUM — prewhitening, not the latent state estimate, is what buys
  detection when the series is autocorrelated and the latent signal is
  strong, while a raw variance CUSUM suffices only when observation noise
  dominates (low SNR). Use the exceedance-indicator variant under
  distributional uncertainty (heavy tails), and the composite for
  breadth. Calibrate everything on matched nulls at a common FAR and
  report empirical FARs.
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
`make realtime` for the data applications, snapshots under `data/`).
84 tests include bit-identical no-lookahead checks for every feature and
detector (including the raw and ARIMA variance rungs and a training-freeze
check), DGP ground-truth checks, and calibration-parity checks. All
post-hoc design changes and pre-registered hypotheses (including the
three falsified ones and the two failed robust-feature designs) are in
`experiments/CHANGELOG.md`; full experiment narratives in
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
| Decision-rule outcome fired | C (raw rung SNR-dependent) | CHANGELOG |
| Variance ×1.5 across T = 200/500/2000 | 0.11 / 0.87 / 0.99 | grid_v2_T |
| t₅ collapse and repair (×1.5) | 0.16 → 0.75 (tail_cusum) | grid_v2_misspec, grid_v3c |
| Quieting ×⅔ (only tail_cusum) | 0.41 / 0.33 | grid_v3c |
| Persistence-down, best anywhere | 0.33 (SNR 2.0) | grid_v1 |
| Multi-break: raw second-event recall | 0.00 | exp04 |
| Composite level→var second event | 0.60 (F1 0.63) | exp04 |
| INDPRO permutation p (composite) | 0.008 | rd_eval |
| GFC real-time | data month 2008-09, known 2008-12 | rd_realtime |
| COVID real-time | data 2020-03, ~2 mo before NBER | rd_realtime |
