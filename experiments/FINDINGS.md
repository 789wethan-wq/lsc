# Running findings log

Honest summaries of each experiment, including negative results
(SPEC §11). Numbers are 500-replication Monte Carlo with all detectors
calibrated to a 5%-per-500-obs false-alarm rate on matched nulls;
MC-SEs in the results parquets.

## exp01 (v2) — calibrated detection comparison, AR(1) + local-level arenas

Files: `paper_assets/exp01_results.*`, `exp01_far_calibration.csv`,
`exp01_frontier.png`; v1 (before the state-baseline feature) preserved
as `exp01_v1_*`.

**1. Level breaks, AR(1) arena (spec-SNR ≈ 5, i.e. low obs noise):
latent-space detection matches but does not beat raw-Y CUSUM.**
lsc_state_cusum vs raw_cusum: detect 0.96 vs 0.98 (3σ), 0.21 vs 0.28
(1σ); median delay 122 vs 98 (3σ). Both have slightly-above-target
empirical FAR (8.6% / 7.8%) from heavy-tailed threshold calibration
noise, so the head-to-head is fair. Interpretation: with nearly
noiseless observations the filtered state is essentially Y, so no edge
is available. This motivated exp02's SNR sweep (hypothesis logged in
CHANGELOG before running).

**2. The innovation CUSUM is "fast or never" on level breaks.**
Median delay 23 obs at 3σ (vs 98 for raw) but detect rate 0.67 vs 0.98:
the filter adapts to a persistent shift, innovations carry only
(1−φ)·δ per step, below the CUSUM drift allowance, so evidence stops
accumulating. Known trade-off, now quantified at matched FAR.

**3. Variance breaks: the diagnostics layer dominates every benchmark.**
Composite detect 0.98 with median delay 18 obs (AR(1) arena) and 0.98 /
delay 11 (local-level); best benchmark: arima_cusum 0.34 (local-level),
raw_cusum 0.04. Driven by the variance_pressure (CUSUM of e²−1) and
instability features on the filtered path.

**4. The composite pays a breadth tax on level breaks.** Max-over-8-
standardized-features costs power vs the single best feature (0.69 vs
0.96 at 3σ) — expected multiple-testing cost; adaptive weighting is a
v2 direction.

**5. Local-level (random-walk state) arena: no method detects level
breaks above FAR** — the break is statistically absorbed into the
random walk (see CHANGELOG entry 1). Variance breaks remain detectable
(composite 0.98). plain_hmm cannot be FAR-calibrated on nonstationary
data at all: its filtered regime probability saturates to machine-1.0
under the null (33% empirical FAR at the maximal threshold) — reported
as a benchmark failure mode, excluded from delay comparisons there.

## exp02 — SNR sweep: hypothesis falsified

Files: `paper_assets/exp02_results.*`, `exp02_far_calibration.csv`.
Hypothesis (pre-registered in CHANGELOG): lsc_state_cusum's edge over
raw_cusum at matched FAR grows as spec-SNR falls from 2.0 to 0.1.

**Outcome: falsified — there is no SNR regime where latent-space
detection beats raw-Y CUSUM on level-shift detect rate.** At 3σ
raw_cusum detects 0.97–0.99 across SNR ∈ {0.1, 0.5, 2.0};
lsc_state_cusum rises with SNR (0.19 → 0.75 → 0.94) but never
overtakes. Two mechanisms: (a) at low SNR, Y is nearly white around its
mean — exactly the regime where the classical CUSUM is near-optimal —
while the state CUSUM's training baseline is estimated from a
persistent filtered series (tiny effective sample size) and
standardization by the small filtered-state sd amplifies that error
into its null distribution; (b) at high SNR the filtered state is
essentially Y, so no information advantage exists. FAR calibration was
tight at low SNR (3.4–5.0%) and modestly loose for both baseline-CUSUM
detectors at SNR 2.0 (7.8% / 8.2% — same direction, fair head-to-head).

**Consistent positive:** the innovation CUSUM remains the fastest
detector conditional on firing at every SNR (median delay 24–53 obs at
3σ vs raw's 58–91), at the cost of detect rate (0.55–0.67 vs ~0.99) —
the "fast-or-never" profile quantified across the noise spectrum.

**Level-shift question is settled:** the latent layer's contribution
for first-moment breaks is speed, not reliability. The layer's power
case rests on higher-moment/dynamics changes → exp03.

## exp03 — pure dynamics (persistence) breaks: nobody detects them

Files: `paper_assets/exp03_results.*`, `exp03_far_calibration.csv`.
Hypothesis (pre-registered): raw_cusum blind, arima_cusum partially
sighted, LSC diagnostics clearly above FAR.

**Outcome: wrong in both directions — at spec-SNR 0.5 and these
magnitudes (φ 0.95→0.995 and 0.95→0.80 with marginals preserved), no
tested method detects the change above FAR.**

- persistence-up: LSC 0.03–0.07, ARIMA 0.05 — but raw_cusum reached
  0.16: freezing the dynamics (φ≈1, q≈0) pins the state at its
  break-time draw, which *conditionally on the path* acts like a small
  never-reverting level shift — raw CUSUM's exact target. "Marginals
  preserved" holds unconditionally, not path-wise.
- persistence-down: every method at 0.01–0.03, i.e. BELOW null alarm
  rates. Verified mechanism: faster mean reversion makes long
  excursions rarer, so all excursion-based statistics are actively
  suppressed (post-break filtered-state sd 0.443 vs null 0.478); the
  innovation-variance rise (e²→1.126) stays below the variance-CUSUM
  drift allowance (k=0.25); the whole change compresses to a Y lag-1
  autocorrelation shift of 0.317→0.267 — information-theoretically thin.

Detecting dynamics-quieting needs dedicated two-sided statistics
(rolling AR coefficient with calibrated two-sided alarm, "quietness"
CUSUM of (1−e²)); logged as a v2 method direction.

## grid_v1 — full calibrated grid (3 SNRs × 9 scenarios × 4 methods, T=500)

Files: `paper_assets/grid_v1_*` (parquet/CSV, LaTeX tables, frontier
plots). Run with the current composite (11 features, per-t
standardization) — SUPERSEDES the composite rows of exp01/exp02, which
predate the standardization fix.

- **Subtle variance breaks are the composite's signature case:** at
  ×1.5 observation-noise scale, composite detects 0.82 / 0.87 / 0.91
  (SNR 0.1/0.5/2.0) while every other method sits at 0.04–0.09 — i.e.,
  at FAR. At ×3 the composite is 0.98–0.99 everywhere.
- **Level/ramp breaks: raw-Y CUSUM dominates at all SNRs**
  (0.93–0.99 at 3σ), confirming exp02 across ramp speeds; the best
  latent challenger is the state CUSUM at high SNR (0.90–0.94).
- **Quietness features vindicated at high SNR:** persistence-down at
  SNR 2.0 is detected by the composite at 0.33 — the ONLY above-FAR
  persistence detection by any method anywhere in the grid (everything
  else ≤ FAR, most suppressed below it). At lower SNR the dynamics
  signal in the innovations is too thin, consistent with exp03b.
- FAR table: `grid_v1_far_calibration.csv` (3.4–8.2%; the two
  baseline-CUSUM detectors are the heavy-tailed offenders at SNR 2.0,
  same direction).

## m6 — FRED INDPRO application (illustrative)

Files: `paper_assets/m6_fred_*`. Monthly INDPRO growth 1948–2026,
rolling 120-month training / 60-month monitoring, thresholds calibrated
per segment by parametric bootstrap from the fitted AR(1)-SSM at 5%
FAR per window. NBER peaks are reference events, not ground truth.

At this deliberately strict FAR the detectors alarm only on major
dislocations — which is the calibrated behavior, not a failure: LSC
composite fired 2008-09 (9 months after the 2007-12 NBER peak, 3
months BEFORE both raw CUSUM and the NBER committee's own announcement
of that peak), 2020-04 (2 months after the COVID peak; raw CUSUM missed
it in-window), 1990-12 (5 months post-peak; only the composite fired),
and 1969-08 (4 months before the 1969-12 peak — early warning or false
alarm, reported as-is). Milder recessions produced no alarms at 5%
FAR / 60 months. The real-data detection profile mirrors the
simulations: the latent composite earns its keep on volatility-type
dislocations and speed.

## grid_v2_T — sample-size sweep (T ∈ {200, 2000}, SNR 0.5)

Files: `paper_assets/grid_v2_T_*`. Hypotheses pre-registered in
CHANGELOG (2026-07-11). T=500 reference values are grid_v1's.

- **Composite's subtle-variance edge scales with T — confirmed.**
  Variance ×1.5 detect: 0.11 (T=200) → 0.87 (500) → 0.99 (2000);
  variance ×3 is at ceiling even at T=200 (1.00, median delay 26 of the
  ~100 post-break obs, while the best benchmark sits at 0.17). Variance
  evidence accumulates linearly in e², so runway is what it needs.
- **Level 3σ: raw CUSUM stays on top at T=200 (0.80 vs composite
  0.57); at T=2000 everyone saturates** (0.98–1.00) and the latent
  state CUSUM (1.00, but median delay 169 vs raw's 90) no longer trails
  on detect rate — "slow but sure" methods converge given runway.
- **persistence_down at T=2000: composite 0.17 vs FAR 0.046** — above
  FAR (as hypothesized) but far from powerful: the dynamics signal
  grows with T, yet even 1000 post-break obs at SNR 0.5 yield only
  ~3.7× FAR. persistence_up at T=2000 is detected best by raw_cusum
  (0.35) and state_cusum (0.28) via the conditional-level-freeze
  effect first seen in exp03 — again NOT a genuine dynamics detection.
- **Short-T calibration health:** at T=200 the two heavy-tailed CUSUM
  detectors run hot (empirical FAR 8.8–9.4% vs 5% target) — the
  order-statistic threshold noise documented in exp01 worsens as the
  null max distribution shortens; at T=2000 calibration is tight
  (2.4–5.2%). Short-T claims should quote these FARs alongside power.

## grid_v2_misspec — t₅ noise and nonlinear drift arenas (T=500, SNR 0.5)

Files: `paper_assets/grid_v2_misspec_*`. Calibration uses the matched
misspecified null, so FAR stays honest (4.8–8.0%); the question is
power. Hypotheses pre-registered in CHANGELOG (2026-07-11).

- **t₅ observation noise: rankings preserved, and the subtle-variance
  edge is the main casualty.** Level 3σ: raw_cusum 0.99 (unhurt — it
  never trusted the Gaussian model), all Kalman-based methods lose
  power (state 0.70, composite 0.51). Variance ×3: composite 0.97,
  still no contest. But variance ×1.5 collapses to 0.16 (from 0.87
  Gaussian): heavy-tailed e² inflates the null variance-CUSUM
  distribution, so a 1.5× scale rise no longer clears the calibrated
  bar. The composite's signature case needs either tail-robust
  (Huberized) innovation features or a bigger break under fat tails.
- **Nonlinear tanh drift (mildly bimodal state): level detection
  collapses for EVERYONE** (raw 0.15, LSC 0.04–0.13 at 3σ): the
  bimodal state's own regime-hopping inflates every level-detector's
  null threshold — an echo of the local-level degeneracy, now caused
  by nonlinearity instead of nonstationarity. Meanwhile the
  composite's variance detection is fully intact (×1.5: 0.97, ×3:
  0.97) — second-moment features don't care that the mean dynamics are
  misspecified. This is the cleanest illustration yet that the two
  detection families read disjoint information channels.

## exp04 — multi-break paths, event-level F1 (T=500, SNR 0.5)

Files: `paper_assets/exp04_*`. Re-arm protocol (drain below ½
threshold + 20-obs refractory, identical for all methods), one-to-one
greedy alarm–break matching within 100 obs. Empirical FAR 4.8–6.2%;
the protocol adds almost no null chatter (≤1.2% of null paths produce
a second alarm). Hypotheses pre-registered in CHANGELOG (2026-07-11).

- **Raw CUSUM is structurally one-shot — confirmed.** Second-event
  recall 0.00 in every scenario (vs 0.74 on a first 3σ level break):
  the fixed-baseline statistic saturates and cannot drain.
- **But CUSUM-family LSC detectors are also effectively one-shot at
  this event spacing (150 obs).** On level→level(−) every method's
  second-event recall is ≤ 0.05: first-event delays of ~40–65 obs plus
  slow post-adaptation drain leave detectors disarmed (or the matching
  window closed) when the second event arrives.
- **The composite is the exception when the events live in different
  channels:** level→variance recall on the second event 0.60 (others
  ≤ 0.22), best F1 0.63 — its variance features were never saturated
  by the level event. Conversely var-up→var-down: the vol-up is
  caught almost surely (0.99, delay 22) but the quieting-back never is
  (0.00) — the variance CUSUM saturates upward and blocks re-arm.
  Multi-break detection needs statistics with bounded memory (e.g.
  windowed variants) rather than a smarter re-arm rule; logged as a v2
  direction, not patched post hoc.

## exp05 — clip-based robust variance features: falsified (worse everywhere)

Files: `paper_assets/grid_v3_robust_*`. Hypotheses pre-registered in
CHANGELOG; all three wrong.

The Huberized features (innovations clipped at 2.5·MAD before
squaring) were meant to repair the t₅ collapse of the subtle-variance
case. Result: composite_robust is strictly worse than the standard
composite on every variance scenario in BOTH arenas — under t₅,
variance ×1.5 fell 0.16 → 0.03 and ×3 0.97 → 0.46; under Gaussian
noise ×1.5 fell 0.87 → 0.06 and ×3 0.99 → 0.82 (median delay 26 →
173). Level and persistence rows were unchanged as expected (shared
features). FAR calibration was clean (4.6–6.2%), so this is a pure
power loss.

Post-mortem: a variance rise manifests almost entirely in the tail of
e². Clipping bounds the null summand by removing exactly the
observations that carry the signal — it buys a thin-tailed null by
destroying the alternative. The general lesson: tail-robustness for
scale detection must come from statistics that are bounded per
observation yet still driven by tail EVENTS (indicators/ranks), not
from truncating tail magnitudes. That design is exp05b/c.

## exp05b/c — exceedance-indicator CUSUM: the t₅ repair that works (standalone)

Files: `paper_assets/grid_v3b_exceedance_*`, `grid_v3c_tail_*`.
Hypotheses and the non-eval-seed k-selection procedure pre-registered
per step in CHANGELOG.

The exceedance features (one-sided CUSUMs of the INDICATOR
1{|e| > training q̂90} minus its training rate) have bounded summands
under any innovation distribution, yet a ×1.5 scale rise moves the
exceedance probability 0.10 → ~0.25 even under t₅. Two-stage outcome:

- **exp05b (inside the composite, k=0.02): falsified at the composite
  layer.** The raw feature separates near-perfectly (93–100% of break
  paths above the null 95th percentile in both arenas), but the
  max-over-features composite rewards break-to-null-IQR ratio, and a
  bounded-increment CUSUM with a barely-negative null drift wanders
  enough to compress its standardized z (~15) below the composite's
  threshold (~28, set by break_pressure/instability null tails).
- **exp05c (standalone detector `lsc_tail_cusum` = max of exceedance
  k=0.05 and shortfall k=0.02 arms):** variance ×1.5 detect 0.87
  Gaussian / **0.75 t₅** (composite: 0.87 / 0.16) — the heavy-tail
  hole is repaired, 3–5pp under the pre-registered bars (reported
  as-is); ×3 at 0.99–1.00 with median delay ~37 under BOTH
  distributions; and subtle QUIETING ×0.67 detected at 0.41 / 0.33
  while every other method sits at FAR — the first successful
  quieting detection in the project. FAR clean (3.6–6.2%). The
  composite-embedded variant improved (0.58 / 0.21) but stays well
  below the standalone: the z-compression is structural.

Recommendation: `lsc_tail_cusum` is the second-moment detector of
choice under distributional uncertainty; the 11-feature composite
remains the breadth instrument (best on Gaussian subtle variance and
the only method with any persistence sensitivity); the clip-based
features are a documented negative result and are in no recommended
composite.

## exp06 — fast-or-never formalized and verified (THEORY.md)

Files: `experiments/THEORY.md`, `lsc/theory.py`,
`paper_assets/exp06_*`. Predictions pre-registered in CHANGELOG.

Proposition 1: after a state level shift δ, the standardized
innovations acquire a deterministic mean path decaying geometrically
(rate ρ = φ(1−K)) to μ_∞ = δ(1−φ)/((1−φ(1−K))√F). Proposition 2: if
μ_∞ < k, post-transient detection probability ≤ (L+1)e^{−2(k−μ_∞)h} —
"fast or never" is a theorem, not a heuristic. Proposition 3: raw
CUSUM's sustained drift Δ = δ/σ_Y gives certain detection with Wald
delay ≈ h/(Δ−k) when Δ > k. With φ=0.95 and k=0.5, μ_∞ > k would
require ~10σ_ref shifts: the innovation CUSUM is STRUCTURALLY in the
fast-or-never regime at any realistic magnitude.

Verification (1000 reps): the MC innovation mean path matches μ_t
(max dev 0.079, MC SE 0.032); the reduced simulation (iid normal +
μ_t) reproduces full-filter detection curves within MC error; the
bound is never violated. Against grid_v1's actual calibrated
thresholds: δ≤1σ ⇒ bound ≤ 0.7% and observed detect ≈ FAR; δ=3σ is a
knife-edge (μ_∞ 0.43–0.48 vs k=0.5) matching the observed 0.55–0.67;
raw CUSUM Wald delays 68/84/110 vs observed medians 58/75/91, and its
partial 1σ detect (0.30) is explained by drift 0.577 barely above k
(Wald delay 1334 ≫ 250-obs horizon). The whole exp01–exp02 level-break
phenomenology now follows from two closed-form quantities (μ_∞, Δ).

## m6x — real-data extension: attribution, new series, sensitivity, permutation tests, real-time vintages

Files: `paper_assets/rd_*`, snapshots in `data/` (2026-07-11), design
registered in CHANGELOG. All illustrative (SPEC §4.5/§8).

**Attribution confirms the simulation profile on real data.** Every
composite alarm on every series is attributed to a SECOND-MOMENT
feature: variance_pressure for GFC and COVID (INDPRO 2008-09/2020-04,
GDP 2008-10/2020-04, GS10 Volcker 1980-02), variance_quiet for the
1990-12 and 1969-08 INDPRO alarms and the post-Volcker bond-market
quietings (GS10 1990-11, 1996-11, 2019-05). The latent layer's
real-data value is exactly where the simulations put it.

**Event association is significant for the composite on INDPRO:**
3/9 NBER peaks hit within 12 months, 1 stray alarm vs 0.7 expected at
the FAR budget, permutation p = 0.008 (innovation CUSUM p = 0.018;
raw CUSUM 1 hit, p = 0.15). The stray (1969-08, variance_quiet) is
within the false-alarm budget — reported as such rather than claimed
as early warning.

**GS10 (yield changes):** the composite alarms 4 months after the
registered Volcker event (1980-02, variance_pressure; raw and
innovation CUSUM the same month) and flags the well-known
disinflation-era quietings; the 2008-12 ZLB event is missed within 12
months. Several "strays" (1968-05, 1973-05 oil-shock era) are
plausible vol events outside the registered 2-event list — the
registered list was deliberately minimal, so they count against us
(permutation p = 0.22).

**GDP / Great Moderation: honest miss.** No detector alarms near the
registered 1984Q1 event. The quieting IS eventually detected —
composite variance_quiet 1992Q2, tail shortfall 1997Q1 — consistent
with a rolling design whose 60-quarter training windows still contain
1970s volatility until the early 1990s. Causal, FAR-calibrated
detection of the Great Moderation lags the retrospective full-sample
break date by ~8–13 years; that is a property of honest monitoring,
reported as such.

**Sensitivity:** the FAR=10% variant behaves as expected (more
alarms, association attenuates, raw p = 0.019). The (180, 36) window
variant exposes a REAL caveat: the composite becomes chatty (14
alarms/20 windows, 12 stray, p = 0.55) while raw/innovation/tail stay
sane — the per-segment bootstrap null is fit on 15 years of
nonstationary real data and no longer matches the monitored window.
Composite calibration on real data needs training windows short
enough to be locally stationary (120 months worked; 180 did not).

**Real-time (ALFRED vintage) check of the headline claims:** COVID is
fully robust — real-time composite alarm at the 2020-04 vintage (data
month 2020-03), one month EARLIER than on revised data and ~2 months
before the NBER announcement (2020-06-08). The GFC claim must be
downgraded: the crossing occurs at the same data month (2008-09) but
only in the 2008-12 vintage — earlier vintages' initially-published
data don't cross — so real-time knowledge arrives ~2008-12,
coincident with the NBER announcement (2008-12-01), not 3 months
before it. It remains 4 months ahead of raw CUSUM's real-time alarm
(2009-04). Paper wording: "same data month as on revised data;
knowledge date coincident with the NBER announcement; 4 months ahead
of the raw-data benchmark in real time."

**tail_cusum on real data:** late or off-reference on INDPRO NBER
peaks (0 hits, p = 1.0) — its alarms (1975-09, 2009-08, 2021-03
exceedance-up; 1992-04, 2017-04 shortfall-down) read as vol-regime
events, not business-cycle peaks. The instrument works, but NBER
peaks are the wrong reference class for it; a vol-regime reference
set would be needed to score it fairly on real data.

## Overall picture (exp01–exp04, grid_v1–v2, FRED)

At matched, null-calibrated FAR: (1) for first-moment (level/ramp)
shifts, raw-Y CUSUM is the reliability king at every SNR and sample
size up to the point where everything saturates (T=2000); the latent
innovation CUSUM is the speed king conditional on firing
(fast-or-never). (2) For second-moment breaks the LSC diagnostics layer
dominates all benchmarks — decisively so for subtle changes (variance
×1.5: 0.82–0.91 vs ≤0.09 at T=500, 0.99 at T=2000, at ceiling for ×3
even at T=200) where every Y-space method is blind. This edge survives every robustness probe once the exceedance repair
is in: heavy tails collapse the e²-based composite on ×1.5 (0.87 →
0.16 under t₅) but the standalone exceedance-indicator CUSUM
(lsc_tail_cusum, exp05c) restores it (0.75 t₅ / 0.87 Gaussian), adds
the first subtle-quieting detection (×0.67: 0.33–0.41 vs FAR for
everyone else), and holds ×3 at ~1.0 under both distributions; under
nonlinear drift — which destroys level detection for every method —
variance detection is untouched (0.97). (3) Pure-dynamics
(persistence) changes are near the information floor: only the
composite's quietness/autocorrelation features detect them above FAR
(0.33 at SNR 2.0; 0.17 at T=2000, SNR 0.5), and apparent detections by
level-type methods are the conditional-level-freeze artifact. Quieting
changes suppress all excursion statistics below null levels. (4)
Multi-break paths: every CUSUM-family detector is effectively one-shot
at 150-obs event spacing (raw CUSUM structurally so — second-event
recall 0.00); the composite alone catches a second event when it lives
in a different channel than the first (level→variance: 0.60). Bounded-
memory statistics are the v2 fix. (5) The FRED application reproduces
the simulation profile on real data (fastest on GFC/COVID, ahead of
the NBER announcement for 2007-12). (6) Methodological byproducts:
plain-HMM regime alarms cannot be FAR-calibrated on nonstationary
data; heavy-tailed null maxima make quantile thresholds noisy at
500-rep budgets (worst at short T: 8.8–9.4% empirical FAR at T=200);
composite features must be standardized per-time-point, not pooled;
drain-based re-arm rules interact badly with saturating statistics.
The calibrated-FAR parity protocol itself exposed all of the above —
consistent with SPEC §11's fallback contribution.
