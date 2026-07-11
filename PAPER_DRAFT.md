# When Does Filtering Help You See a Break? Latent-State Diagnostics for Structural Change at Calibrated False-Alarm Rates

**Draft skeleton for prose editing.** Every number below is from the final
runs (500 Monte Carlo replications unless noted; MC standard errors in the
results parquets under `paper_assets/`). Sections marked *[EDIT]* need
connective prose or a decision from you; everything else can be reworded
freely — the numbers should not change.

---

## Abstract *[EDIT — 150 words, draft below]*

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
dynamics changes the ordering reverses: the latent diagnostics detect
subtle variance changes (×1.5) that every raw-data method misses at
chance, survive heavy tails via an exceedance-indicator variant, and are
the only detectors of variance quieting. A real-data application
(industrial production, GDP, Treasury yields) reproduces the profile:
every alarm attributes to a second-moment feature, association with NBER
reference dates is significant (permutation p = 0.008), and real-time
(ALFRED vintage) analysis confirms the COVID timing while honestly
downgrading the 2008 timing claim.

---

## 1. Introduction *[EDIT — motivation prose; the skeleton of the argument:]*

- Structural change is usually sought in observables; but many economic
  quantities of interest are latent states observed with noise. Intuition
  says filtering should help detection. Does it?
- Contribution 1 (protocol): a calibrated-FAR parity harness. Every
  method's threshold comes from the same routine on the same matched-null
  draws; empirical FAR is re-verified on fresh nulls; causality is
  enforced structurally (parameters fit on a training prefix only;
  forward-only filtering; alarm scores are NaN on training data) and
  tested bit-identically (perturbing future observations leaves all
  scores at earlier times unchanged, exactly).
- Contribution 2 (negative + positive results): the intuition is wrong
  for first moments at every SNR — and we can prove why (fast-or-never
  theorem). It is right, decisively, for second moments and dynamics.
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

*[EDIT: related literature — CUSUM/Page, Hamilton/Markov-switching,
McConnell–Perez-Quiros (2000) for the Great Moderation, state-space
monitoring, changepoint detection (PELT etc.). One paragraph each.]*

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

**Theory (verified).** After a state level shift δ, the standardized
innovation mean decays geometrically at rate ρ = φ(1−K) to
μ∞ = δ(1−φ)/((1−φ(1−K))√F), where K is the steady-state Kalman gain and
F the innovation variance. If μ∞ < k, the post-transient CUSUM has
negative drift and P(alarm in L further obs) ≤ (L+1)·exp(−2(k−μ∞)h): the
detector fires during the adaptation transient or, with exponentially
small probability, never. With φ = 0.95 and k = 0.5, μ∞ > k would need
shifts of order 10σ — the innovation CUSUM is *structurally* in the
fast-or-never regime. Verification: the Monte Carlo innovation path
matches μ_t within MC error; the bound is never violated; at the actual
calibrated thresholds, μ∞ sorts every observed detection rate — δ ≤ 1σ
gives bound ≤ 0.7% (observed ≈ FAR); δ = 3σ is a knife-edge (μ∞ =
0.43–0.48 vs k = 0.5) matching the observed 0.55–0.67. The raw CUSUM's
sustained drift Δ = δ/σ_Y gives Wald delays 68/84/110 across SNRs versus
observed medians 58/75/91, and explains its partial 1σ detection (0.30)
without any fitting: Δ = 0.577 barely exceeds k, so the Wald delay (1334)
dwarfs the 250-observation horizon.

*[EDIT: this is the paper's intellectual core — consider promoting the
theorem statements to formal Proposition environments; full derivations
in experiments/THEORY.md.]*

## 5. Second moments: the latent layer's home turf

At ×1.5 observation-noise scale — invisible to the eye and to every
raw-data method (≤ 0.09, i.e. chance) — the composite detects 0.82 / 0.87
/ 0.91 across SNRs at T = 500, scaling with runway: 0.11 at T = 200, 0.99
at T = 2000. At ×3, detection is at ceiling even at T = 200 (1.00, median
delay 26 of the ~100 post-break observations, best benchmark 0.17).
Attribution: the e²-based variance-pressure features drive these alarms.
On real data the same features drive every crisis alarm (§9).

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
chatter, binds. Bounded-memory statistics are the identified fix
*[EDIT: keep as future work]*.

## 8. Robustness

**8.1 Sample size.** §5 numbers; short-T caveat: heavy-tailed null maxima
make quantile thresholds noisy, and at T = 200 the two baseline CUSUMs
calibrate hot (8.8–9.4% vs the 5% target). Quote empirical FARs alongside
any short-sample power claim.

**8.2 Misspecification.** t₅ observation noise: rankings preserved; raw
CUSUM unhurt on levels (0.99); the composite's subtle-variance case
collapses (×1.5: 0.87 → 0.16) — repaired in §8.3. Nonlinear tanh drift
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
p = 0.15).

**Real-time vintages (ALFRED).** COVID: robust — real-time alarm at the
2020-04 vintage (data month 2020-03), one month earlier than on revised
data and ~2 months before the NBER announcement. GFC: downgraded — the
crossing is at the same data month (2008-09) but only in the 2008-12
vintage; real-time knowledge is coincident with the NBER announcement
(2008-12-01), though still 4 months ahead of raw CUSUM's real-time alarm
(2009-04). *[EDIT: this self-correction is a feature of the paper —
consider a short subsection titled "What real-time data changes".]*

**GDP (GDPC1, quarterly).** GFC and COVID caught (variance_pressure,
2008Q4 and 2020Q2). The registered Great Moderation event (1984Q1) is an
honest miss: causal rolling monitoring detects the quieting only in
1992Q2 (composite) / 1997Q1 (tail shortfall), because 60-quarter training
windows contain 1970s volatility until the early 1990s. Retrospective
full-sample break dates are not reproducible by honest monitoring.

**10-year Treasury yield changes (GS10).** Volcker regime caught 4
months after the registered 1979-10 event (variance_pressure; raw and
innovation CUSUM the same month); the post-Volcker disinflation
quietings are flagged (1989–1996, shortfall/variance_quiet); the 2008
ZLB event is missed within 12 months.

**Sensitivity.** FAR = 10% behaves as expected. Training window 180
months breaks the composite's bootstrap calibration on nonstationary
real data (14 alarms / 20 windows) while the single-statistic detectors
stay sane — training windows must be short enough to be locally
stationary (120 months worked).

## 10. Discussion *[EDIT — draft claims to build prose around:]*

- The latent-state diagnostics layer is not a better level-shift
  detector; it is a *different instrument*, reading second moments and
  dynamics that raw-data detectors are blind to. The division of labor
  is now both measured (grids) and derived (fast-or-never).
- Practical recipe: run raw CUSUM for levels, the exceedance CUSUM for
  scale changes under distributional uncertainty, the composite for
  breadth; calibrate everything on matched nulls at a common FAR and
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

## Reproducibility note *[EDIT — for the paper's appendix]*

`make all` regenerates every table and figure from pinned seeds
(Python 3.14, statsmodels/hmmlearn; `make fred` / `make realdata` /
`make realtime` for the data applications, snapshots under `data/`).
77 tests include bit-identical no-lookahead checks for every feature and
detector, DGP ground-truth checks, and calibration-parity checks. All
post-hoc design changes and pre-registered hypotheses (including the
three falsified ones and the two failed robust-feature designs) are in
`experiments/CHANGELOG.md`; full experiment narratives in
`experiments/FINDINGS.md`; theory in `experiments/THEORY.md`.

## Key numbers table (for the editor — not for the paper)

| Claim | Number | Source |
|---|---|---|
| Raw CUSUM level 3σ detect, all SNRs | 0.97–0.99 | grid_v1 |
| Innovation CUSUM delay vs raw, 3σ | 24–53 vs 58–91 obs | exp02 |
| Innovation CUSUM detect at 3σ | 0.55–0.67 | exp02/grid_v1 |
| μ∞ at 3σ, SNR 0.5 (knife-edge vs k=0.5) | 0.469 | exp06 |
| Composite variance ×1.5, T=500 | 0.82–0.91 (others ≤0.09) | grid_v1 |
| Variance ×1.5 across T = 200/500/2000 | 0.11 / 0.87 / 0.99 | grid_v2_T |
| t₅ collapse and repair (×1.5) | 0.16 → 0.75 (tail_cusum) | grid_v2_misspec, grid_v3c |
| Quieting ×⅔ (only tail_cusum) | 0.41 / 0.33 | grid_v3c |
| Persistence-down, best anywhere | 0.33 (SNR 2.0) | grid_v1 |
| Multi-break: raw second-event recall | 0.00 | exp04 |
| Composite level→var second event | 0.60 (F1 0.63) | exp04 |
| INDPRO permutation p (composite) | 0.008 | rd_eval |
| GFC real-time | data month 2008-09, known 2008-12 | rd_realtime |
| COVID real-time | data 2020-03, ~2 mo before NBER | rd_realtime |
