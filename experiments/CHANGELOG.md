# Experiment changelog (SPEC §11)

All post-hoc changes to experiment grids, with rationale. Entries are
append-only.

## 2026-07-10 — exp01: primary arena changed from local-level to AR(1) state

**What changed.** The first (30-rep smoke) version of exp01 used the
local-level DGP (random-walk latent state) as the only arena. After
seeing smoke results, the primary arena was changed to a mean-reverting
AR(1) latent state (phi=0.95, SNR 0.5); local-level was kept as a
secondary, documented hard case.

**Why (and why this is not metric shopping).** The smoke run showed
detect rates ≈ FAR for *every* method on local-level level breaks,
including at 3σ. Diagnosis: under a random-walk state model, a level
jump is statistically indistinguishable from an ordinary random-walk
innovation beyond the first observation — the well-specified Kalman
filter absorbs the break within a few steps, so no causal method has
sustained power; likewise raw-Y CUSUM is undefined-in-spirit on a
nonstationary path (its calibrated threshold was ~2300). The arena
cannot rank methods because the detection problem is nearly
unidentifiable there. The AR(1) arena makes the break identifiable
(persistent shift relative to mean-reverting dynamics) and is *more*
favorable to the raw-Y CUSUM benchmark (Y is stationary pre-break), so
the change strengthens rather than weakens the comparison. Local-level
results are still computed and reported.

**Also changed with rationale.**
- plain_hmm benchmark score changed from P(non-dominant regime) to its
  log-odds: the raw probability saturates at 1.0, so no threshold could
  achieve the 5% FAR target (the calibrated threshold hit the maximum
  of the score's support). Monotone-equivalent otherwise.
- Smoke scale (30–40 reps) results were used ONLY for these design
  decisions; all reported numbers come from fresh 500-rep runs with the
  seed layout documented in the experiment header.

## 2026-07-10 — exp01 v2: state-baseline CUSUM feature added after seeing v1 results

**What the v1 500-rep run showed.** At calibrated FAR, raw-Y CUSUM beat
the LSC innovation CUSUM on detect rate for level breaks in the AR(1)
arena (3σ: 0.98 vs 0.67), while LSC won on speed-when-detected (median
delay 23 vs 98) and on variance breaks. Mechanism identified: the
filter adapts to a persistent state shift, so innovations retain only
(1−φ)·δ ≈ 0.11σ of signal per step — below the CUSUM drift allowance
k=0.5 — and the statistic stops accumulating ("fast or never"). The raw
CUSUM tests against a fixed training baseline and accumulates the full
shift indefinitely ("slow but sure"). v1 results are preserved in
`paper_assets/exp01_v1_results.*`.

**Change.** Added two features to the diagnostics layer and the
composite: (a) `state_shift_pressure` — Page CUSUM of the filtered
state standardized by its training-prefix moments, i.e., the direct
latent-space counterpart of the raw-Y benchmark (full shift per step,
observation noise filtered out — a priori higher SNR than raw CUSUM);
(b) `variance_pressure` — CUSUM of eₜ²−1, which existed in features.py
but had been left out of the composite by oversight. Also added a
standalone `lsc_state_cusum` detector for a clean single-statistic
head-to-head against `raw_cusum`, and the ARIMA+CUSUM benchmark.

**Integrity note.** This is a method addition motivated by an
understood mechanism, not threshold tuning: all detectors, including
the new ones, are calibrated by the same null-quantile procedure at the
same 5% FAR and evaluated on the same fresh draws. The v1-vs-v2
comparison is itself reported.

## 2026-07-10 — exp02 added: SNR sweep (hypothesis stated before running)

exp01's AR(1) arena had spec-SNR (stationary state var / obs var) ≈ 5.1
— nearly noiseless observations, the regime where latent filtering buys
the least; there the latent state CUSUM only matched raw-Y CUSUM.
Mechanism-based hypothesis, registered here BEFORE exp02 was run: at
low SNR (0.1, 0.5) the filter removes a growing share of observation
noise, so lsc_state_cusum should beat raw_cusum on detect rate and
delay at matched FAR, with the gap widening as SNR falls. Break sizes
are in stationary-state-sd units so difficulty is comparable across
SNR. Same harness/seeds/reps as exp01.

## 2026-07-10 — exp02 outcome recorded; exp03 added (hypothesis stated before running)

**exp02 outcome: hypothesis falsified.** raw_cusum has the best detect
rate on level breaks at every SNR in {0.1, 0.5, 2.0} (0.97–0.99 at 3σ);
lsc_state_cusum never overtakes it (0.19→0.94 as SNR rises). The latent
innovation CUSUM is consistently the fastest conditional on detection
(median delay 24–53 vs raw's 58–91 at 3σ). Recorded in FINDINGS.md; no
grid changes made to chase a win on level breaks.

**exp03 hypothesis (registered now).** For a pure dynamics change —
AR(1) persistence jumps 0.95→0.995 or drops 0.95→0.80 at t=250, with
state-noise variance rescaled so the stationary mean AND variance of Y
are unchanged — Y-space level detectors (raw_cusum) should be blind
(detect ≈ FAR), ARIMA residual CUSUM partially sighted (its fixed AR
fit mispredicts post-break), and the LSC diagnostics layer (persistence,
instability, innovation-based features on the filtered path) should
detect well above FAR. New DGP break kind 'persistence' added with
ground-truth tests (marginals preserved, autocorrelation shifts).

## 2026-07-10 — exp03 outcome recorded

Both registered predictions were wrong in instructive ways. (1) No
method — LSC or benchmark — detects a persistence change above FAR at
spec-SNR 0.5 (up: LSC 0.03–0.07, ARIMA 0.05; down: everyone 0.01–0.03).
(2) raw_cusum was NOT blind to persistence-up (0.16): freezing the
dynamics pins the state at its break-time value, which conditionally
resembles a small never-reverting level shift. (3) persistence-down
actively SUPPRESSES every excursion-based statistic below its null
level (verified: post-break filtered-state sd 0.443 vs null 0.478;
innovation e² rises only to 1.126, below the variance-CUSUM allowance
k=0.25; Y lag-1 autocorrelation moves just 0.317→0.267). Detecting
"the series got quieter/less persistent" requires two-sided dynamics
statistics none of the tested detectors possess — logged as a v2
method direction, not patched post hoc.

## 2026-07-10 — quietness/dynamics features added; exp03b (hypothesis first)

Motivated by exp03's suppression mechanism (not by tuning on its eval
draws): added `variance_quiet` (one-sided CUSUM of 1−e², k=0.05),
`variance_pressure_slow` (k=0.05 variant for small persistent variance
rises like exp03's e²≈1.13), and `innovation_ac` (rolling lag-1
autocorrelation of innovations — white under a correct filter, serially
correlated under any dynamics change; two-sided in the composite).
Composite now has 11 features. Hypothesis for exp03b, registered before
running: the upgraded composite detects persistence breaks above FAR in
both directions (up via variance_quiet + innovation_ac, down via
variance_pressure_slow + innovation_ac), at some breadth-tax cost to
its level-break power (to be re-measured in the M5 grid).

## 2026-07-10 — composite standardization fixed: per-time-point null scales

exp03b (preserved as exp03b_pooledscale_*) showed the new quietness
features changed nothing: composite threshold was bit-identical to v1.
Diagnosis (per-feature z inspection on null vs break draws): feature
scales were pooled over ALL time points, but CUSUM-type features grow
within a path, so their late-time null values dominated the pooled
scale; the calibrated composite threshold sat at ~25 while stationary
features max out near z≈3 under breaks — the composite was blunted for
every feature, in every experiment so far. Fix: standardize each
feature at each time t by the median/IQR of the null-replication
distribution AT THAT t (self-normalization), with a floor at 10% of the
global scale. This is a design-flaw repair to the standardization, not
threshold tuning: calibration still sets the final threshold on the
same nulls at the same FAR. exp03b rerun after the fix; the M5 grid
uses the fixed composite throughout.

## 2026-07-10 — exp03b (rerun with fixed standardization) outcome

Quietness hypothesis falsified again: persistence_up 0.04,
persistence_down 0.01 — even purpose-built two-sided dynamics features
cannot see a φ 0.95→0.995/0.80 change at spec-SNR 0.5 with 250
post-break observations; the underlying signals (innovation lag-1
autocorrelation ±0.02, e² +0.13) are below any calibrated threshold's
reach. Conclusion stands on information grounds, not implementation.
The new features and per-t standardization are retained (principled,
FAR-clean: 4.8%); composite on level 3σ: 0.53 (breadth tax vs 0.75
single-feature state CUSUM), variance ×3: 0.99 / delay 26.

## 2026-07-11 — v2 robustness round registered before running: T sweep, misspecification arenas, multi-break F1

Three extensions to harden the grid_v1 conclusions, hypotheses stated
here BEFORE any of them is run. Same harness, seed layout, and 5% FAR
target throughout; n_reps = 500.

**grid_v2_T (T ∈ {200, 2000}, SNR 0.5, train_frac 0.25).** Hypotheses:
(a) the composite's subtle-variance edge (×1.5) GROWS with T — variance
evidence accumulates in e² at a constant rate, so detection should be
near-ceiling at T=2000 and clearly degraded at T=200 (only 75
post-break obs, 50 training obs); (b) raw CUSUM's level-break dominance
is unchanged at both T; (c) persistence_down (composite quietness
features) becomes detectable above FAR at T=2000 even at SNR 0.5 — the
exp03b failure was information-budget-limited, not mechanism-limited;
(d) at T=200 the composite is hurt MORE than single-statistic methods
(per-t null scales estimated from 50-obs training and 50 scale reps are
noisier, and the breadth tax bites hardest when evidence is scarce).

**grid_v2_misspec (T=500, SNR 0.5; arenas: t₅ observation noise;
nonlinear tanh state drift, drift_coef 0.1).** Calibration uses the
matched (equally misspecified-for-the-model) null, so empirical FAR
should stay ≈5% by construction for all methods — the question is
power. Hypotheses: (a) under t₅ noise every method loses power on level
breaks (heavier-tailed null maxima push thresholds up), but the
RANKING is preserved (raw CUSUM first on level, composite first on
variance ×1.5, which should survive well above FAR since e²-based
features respond to scale, not shape); (b) under nonlinear drift the
Kalman-based methods (state CUSUM, innovation CUSUM, composite) degrade
relative to raw CUSUM on level breaks because the fitted linear AR(1)
is wrong, while variance detection is roughly unaffected.

**exp04_multibreak (T=500, SNR 0.5; two breaks per path at fracs
0.4/0.7; event-level F1 with ±window matching).** New protocol code:
detectors gain a re-arm rule (after an alarm, re-arm once the score
drains below rearm_frac·threshold and a refractory has passed) applied
IDENTICALLY to every method; alarms and breaks are matched one-to-one
greedily within a 100-obs window; precision/recall/F1 per replication.
Hypotheses: (a) raw CUSUM has high recall on the first level break but
LOW recall on the second event because its statistic tests against the
fixed training baseline and stays saturated after break 1 (it never
drains → cannot re-arm); the filter-based LSC detectors adapt and
re-arm, so they win on the second event; (b) on level→variance paths
only the composite detects both events; (c) re-arming costs precision
for everyone (extra alarms), quantified at matched FAR.

## 2026-07-11 — v2 robustness round outcomes recorded

All three runs completed at 500 reps; full numbers in FINDINGS.md. No
grid or method changes were made after seeing results.

**grid_v2_T:** (a) confirmed — composite variance ×1.5 goes 0.11 →
0.87 → 0.99 across T ∈ {200, 500, 2000}; (b) held at T=200 (raw 0.80
vs composite 0.57 on level 3σ) and vacuous at T=2000 (all methods
0.98–1.00); (c) partially confirmed — persistence_down at T=2000
reaches 0.17 vs 4.6% FAR, above FAR but weak; (d) confirmed — at T=200
the composite is the most degraded method AND the hottest calibration
(9.4% empirical FAR from heavy-tail threshold noise at short T).

**grid_v2_misspec:** (a) partially confirmed — under t₅ noise rankings
are preserved and raw_cusum is unhurt (0.99 level 3σ), but the
composite's variance ×1.5 edge collapses to 0.16 (×3 intact, 0.97);
the pre-registered guess that the subtle-variance case "should survive
well above FAR" was too optimistic — heavy tails inflate the null e²
CUSUM directly. (b) wrong in an instructive way — nonlinear drift does
not merely degrade the Kalman methods relative to raw CUSUM: it
destroys level detection for EVERY method (raw 0.15 at 3σ) by
inflating null thresholds via the state's own bimodal hopping, while
composite variance detection is fully intact (0.97 at ×1.5, better
than the Gaussian arena). Tail-robust innovation features logged as
the highest-value v2 method item; no post-hoc patching done.

**exp04:** (a) half-confirmed — raw CUSUM is structurally one-shot
(second-event recall 0.00 everywhere) as predicted, but the LSC
detectors also rarely re-arm in time at 150-obs spacing (level→level
second-event recall ≤ 0.05): first-alarm delay plus slow CUSUM drain
exceeds the event gap. (b) confirmed — level→variance is caught by the
composite alone (second-event recall 0.60, F1 0.63). (c) wrong in a
good way — re-arming costs almost nothing (≤1.2% of null paths give a
second alarm); the binding constraint is saturation, not chatter.
Bounded-memory (windowed) statistics logged as the v2 fix.

## 2026-07-11 — registered before running: robust variance features (exp05) and theory verification (exp06)

**exp05 — Huberized second-moment features.** Motivated by the
grid_v2_misspec t₅ failure (variance ×1.5: 0.87 Gaussian → 0.16 t₅),
mechanism understood: e² is heavy-tailed under t₅, so the null CUSUM of
(e²−1) has occasional huge jumps that inflate the calibrated threshold.
Fix: clip. New features `variance_pressure_robust` /
`variance_quiet_robust` — innovations rescaled by their training-prefix
robust scale (1.4826·MAD), clipped at c=2.5, squared, centered and
standardized by the TRAINING-PREFIX mean/sd of the clipped square
(causal), then one-sided CUSUMs (k=0.05 each way). Clipping bounds the
summand, so the null max is thin-tailed regardless of the innovation
distribution; training-moment centering adapts to whatever null
distribution the arena has. The existing composite is FROZEN as the
11-feature set (new `COMPOSITE_V1` include-list, default of
make_composite_detector — all previous results stay bit-reproducible);
a new method `lsc_composite_robust` swaps the three e²-based features
for the two robust ones. Hypotheses, registered before running
grid_v3_robust.yaml (Gaussian + t₅ arenas × {level_3s, variance_x1.5,
variance_x3, persistence_down} × {composite, composite_robust,
raw_cusum}): (a) under t₅, composite_robust recovers variance ×1.5 to
≥ 0.5 detect (from composite's 0.16); (b) under Gaussian noise the
robustness tax on variance scenarios is ≤ ~10pp; (c) level and
persistence rows are roughly unchanged (those features are shared).

**exp06 — fast-or-never made precise (theory + numerical check).**
Claim to verify: for a steady-state Kalman filter (gain K, innovation
sd √F) on the AR(1)+noise model, a state level shift δ at t₀ adds a
deterministic mean path to the standardized innovations,
μ_t = μ_∞ + (δ−μ_∞·√F)·ρ^{t−t₀}/√F with ρ = φ(1−K), decaying
geometrically to μ_∞ = δ(1−φ) / ((1−φ(1−K))·√F). If μ_∞ < k (the CUSUM
drift allowance), the post-transient CUSUM has negative drift and a
finite-horizon Lundberg/union bound gives P(alarm in remaining L obs)
≤ L·exp(−2(k−μ_∞)h) — "never"; detection can only come from the
transient — "fast". For raw CUSUM the standardized shift Δ = δ/σ_Y
persists forever, so if Δ > k detection is certain with Wald delay
≈ h/(Δ−k). Numerical predictions computed from the formulas BEFORE
running exp06 (SNR 0.5 arena, k=0.5): K=0.165, √F=1.094, μ_∞(3σ) =
0.469 — a knife-edge 0.031 below k, consistent with the observed
partial detect rates 0.55–0.67; μ_∞(1σ) = 0.156 → bound ≈ 7e−5 per 250
obs at h≈22 → never (observed ≈ FAR); raw Wald delay at 3σ with
h≈103 ≈ 84 obs (observed median 82–91). exp06 checks (i) the mean
path against full-filter MC, (ii) the reduced CUSUM (z+μ_t) against
full MC detect rates over an h grid vs the bound, (iii) the μ_∞ table
against grid_v1 innovation-CUSUM detect rates. Derivations in
experiments/THEORY.md.

## 2026-07-11 — exp05 outcome: FALSIFIED; exp05b (exceedance CUSUM) registered before running

**exp05 outcome.** The clipped features are worse everywhere, not
better: under t₅, variance ×1.5 went 0.16 (composite) → 0.03
(composite_robust) and ×3 0.97 → 0.46; even under Gaussian noise the
robust composite collapsed on its target case (×1.5: 0.87 → 0.06; ×3
0.99 → 0.82 with median delay 26 → 173). All three pre-registered
hypotheses wrong. Mechanism (understood after the fact): a variance
rise manifests almost entirely in the TAIL of e² — the clip at
2.5·MAD removes the same observations that carry the signal, so
clipping buys a thin-tailed null by destroying the alternative. The
clip-based features remain in the codebase as a documented negative
result but in no recommended composite.

**exp05b — exceedance-rate features, registered now.** The right
robust statistic must have a bounded null summand while keeping tail
SENSITIVITY: use the exceedance INDICATOR, not the exceedance
magnitude. New features `tail_exceedance` / `tail_shortfall`:
one-sided CUSUMs of (1{|e_t| > q̂90} − p̂ − k) and (p̂ − 1{|e_t| > q̂90}
− k), with q̂90 the training-prefix 90% quantile of |e|, p̂ the
training exceedance rate (≈0.10 by construction), k = 0.02. The
summand is bounded in [−1, 1] under ANY innovation distribution
(thin-tailed null maxima), while a ×1.5 scale rise moves the
exceedance probability 0.10 → ≈0.27 (Gaussian) / ≈0.24 (t₅) — drift
≈ 0.12–0.15 per step against summand sd ≈ 0.3–0.35.
`COMPOSITE_ROBUST2` = COMPOSITE_V1 with the three e²-based variance
features replaced by the two exceedance features. Hypotheses for
grid_v3b (same arenas/scenarios/methods pattern as exp05): (a) under
t₅, composite_robust2 detects variance ×1.5 ≥ 0.5 (composite: 0.16);
(b) under Gaussian noise its variance ×1.5 power is within ~15pp of
the composite's 0.87; (c) level/persistence rows unchanged (shared
features).

## 2026-07-11 — exp05b outcome: falsified AT THE COMPOSITE LAYER; exp05c registered before running

**exp05b outcome.** composite_robust2 sat at FAR on variance ×1.5 in
both arenas (0.06 Gaussian / 0.03 t₅). Diagnosis (on calibration-range
and 400000+ seeds, NOT evaluation seeds): the exceedance FEATURE
separates almost perfectly — its raw CUSUM max under a ×1.5 break
exceeds the null 95th percentile on 93–100% of paths, in both arenas —
but the composite kills it: with allowance k=0.02 the null exceedance
CUSUM wanders (drift −0.02), inflating its per-time-point null IQR, so
its standardized break z tops out around ~15 while the composite's
calibrated threshold (~28) is set by break_pressure/instability null
tails. A max-over-features composite rewards break-to-null-IQR RATIO,
and the bounded-increment exceedance CUSUM cannot reach the ratios the
unbounded e² CUSUM achieves under Gaussian nulls.

**exp05c — registered now.** Two changes, both disclosed: (i) the
allowance for `tail_exceedance` is raised to k=0.05, chosen by a
separation diagnostic (null95 vs break median of the raw feature) run
ONLY on non-evaluation seeds (nulls 100000+, breaks 400000+; k grid
{0.02, 0.05, 0.10, 0.15}); `tail_shortfall` stays at k=0.02 (the
quieting drift is small, ~0.056/step at ×⅔ scale, and cannot afford a
large allowance). (ii) The exceedance statistic is ALSO exposed as a
standalone calibrated detector `lsc_tail_cusum` (score = max of the
two arms), mirroring the exp01-v2 precedent of lsc_state_cusum: the
statistic is strong but the composite's standardization layer
compresses it, so it gets a clean single-statistic head-to-head at
matched FAR. Scenario set gains variance_x0.67 (subtle quieting, the
shortfall arm's target). Hypotheses for grid_v3c: (a) standalone
lsc_tail_cusum detects variance ×1.5 at ≥ 0.9 Gaussian / ≥ 0.8 t₅ —
i.e. the t₅ repair the clip approach failed to deliver; (b)
composite_robust2 (with k=0.05) improves over exp05b but may remain
well below the standalone (z-compression is structural); (c)
variance_x0.67 quieting is detected above FAR by lsc_tail_cusum
(predict ≥ 0.25) in both arenas; (d) raw_cusum stays at FAR on all
variance rows.

## 2026-07-11 — exp05c outcome recorded

(a) Substantially confirmed, slightly under the registered bars:
lsc_tail_cusum detects variance ×1.5 at 0.87 Gaussian (bar: ≥0.9) and
0.75 t₅ (bar: ≥0.8) — the t₅ repair is real (composite: 0.16) but both
numbers miss the registered thresholds by 3–5pp, reported as such.
(b) Confirmed: composite_robust2 rose to 0.58 Gaussian / 0.21 t₅ —
better than exp05b's 0.06/0.03, still far below the standalone
(z-compression is structural, as registered). (c) Confirmed: subtle
quieting ×0.67 detected at 0.41 / 0.33 (every other method 0.02–0.04)
— the first successful quieting detection in the project. (d)
Confirmed: raw_cusum at FAR on every variance row. FAR calibration
3.6–6.2%. lsc_tail_cusum also incidentally fires on level 3σ breaks at
0.30–0.37 (transient innovation exceedances) and hits 0.99–1.00 on ×3
with delay ~37 under BOTH distributions. Recommendation recorded in
FINDINGS: lsc_tail_cusum is the second-moment detector of choice under
distributional uncertainty; COMPOSITE_V1 remains the breadth
instrument; the clip-based features (exp05) and the composite-embedded
exceedance variant (exp05b) are documented negative results. No
further tuning after these results.

## 2026-07-11 — M0 (varbench addendum): claim-adoption decision rule PRE-REGISTERED before any grid_v4 cell runs

Registered per `SPEC_addendum_varbench.md` §1, before implementing or
running the new variance benchmarks. The paper will adopt whichever
claim the results select; no post-hoc reinterpretation.

**New detectors (whitening ladder, M1).** `raw_var_cusum` — Page CUSUM
of z_t²−1 with z_t = (Y_t − ȳ_train)/σ̂_train frozen from the training
prefix; up-arm allowances k = 0.25 and k = 0.05 (mirroring the latent
variance_pressure / variance_pressure_slow features exactly), down-arm
(quieting) CUSUM of 1−z_t² with k = 0.05; score = max over the three
arms; no per-time-point standardization (standalone detector, same
treatment as lsc_tail_cusum). `arima_var_cusum` — the identical
statistic on the standardized one-step residuals of the existing ARIMA
benchmark's training-prefix-fitted, frozen model. Ladder: raw →
ARIMA-whitened → Kalman-whitened (existing e²-based CUSUMs), same
statistic, same allowances, same calibration routine, three
information sets.

**Decision rule.** Let D_raw = raw_var_cusum detection rate at
variance ×1.5, T = 500, per SNR ∈ {0.1, 0.5, 2.0}; D_comp = the
composite's published 0.82 / 0.87 / 0.91 (grid_v1, identical seeds);
FAR target 5%.

- **Outcome A (strong claim):** D_raw within 5 pp of FAR at every SNR
  → abstract/intro upgrade to "raw-data detectors, including a
  variance CUSUM given identical calibration, sit at chance."
- **Outcome B (prewhitening claim):** D_raw within 10 pp of D_comp at
  every SNR → reframe: the advantage is *prewhitening under
  autocorrelation*, not latency per se; §5 and §11 rewritten
  accordingly (the fast-or-never side is untouched). Honest-outcome
  clause: if the latent state layer is unnecessary for second moments,
  the paper says so plainly — "for whitening, not for state
  estimation" — and does not soften it.
- **Outcome C (mixed):** anything else → report the full ladder, keep
  the scoped language, add a paragraph explaining the SNR-dependence.

All three outcomes are publishable; which fired will be logged here
with the numbers.

**Predictions registered alongside (not gating).** (i) Under t₅ noise
raw_var_cusum's ×1.5 power should collapse like the composite's did
(0.87 → 0.16) — z² has the same tail sensitivity; if so, the
exceedance repair story extends to the raw side. (ii) The variance
detectors should NOT detect level shifts above FAR (disjoint-channels
table). (iii) Real-data reruns (M3) may weaken the real-data
uniqueness claim — that is the point of running them; if raw_var_cusum
catches the same crises, §10's claim becomes about the
simulation-calibrated subtlety threshold (×1.5 invisibility), not
real-data uniqueness.

## 2026-07-11 — real-data extension registered before running (m6x)

All real-data results remain ILLUSTRATIVE (SPEC §4.5/§8); this entry
registers the design, not power hypotheses. Components:

1. **Data snapshots.** Date-stamped CSVs committed under `data/`;
   loaders prefer the snapshot (live download only on request) so
   published numbers survive FRED revisions.
2. **Generalized engine** (`experiments/real_data.py`), replacing
   nothing — `m6_fred.py` is kept untouched. Detector set now includes
   `lsc_tail_cusum`; composite alarms are ATTRIBUTED (which feature's
   z crossed at the alarm), tail alarms attributed to the up/down arm.
3. **New series, chosen for the method's signature cases before
   looking at their results:** (a) quarterly real GDP growth (GDPC1) —
   the 1984:Q1 Great Moderation is the canonical QUIETING event
   (McConnell–Perez-Quiros 2000); (b) monthly 10-year Treasury yield
   changes (GS10) — the 1979–82 Volcker episode is the canonical
   volatility-UP regime, the post-2008 ZLB a quieting. Reference
   events fixed in the script header.
4. **Sensitivity variants** for INDPRO: (train, monitor) = (180, 36)
   and FAR = 10%, alarms compared to the baseline set.
5. **Evaluation** (`real_data_eval.py`): per series/method, hits
   within 12 months (4 quarters) after reference events, expected
   vs observed non-event alarms at the calibrated FAR, and a
   permutation p-value (alarm months resampled uniformly over
   monitored months, 20k draws) for the event-association.
6. **Real-time vintage check** (`realtime_check.py`, ALFRED
   `alfredgraph.csv?vintage_date=`): month-by-month decisions for the
   GFC (2008-01..2009-06) and COVID (2020-01..2020-12) episodes,
   training on the 120 months ending at the (later-declared) NBER
   peak, recalibrated per vintage. The revised-data timing claim
   ("composite alarmed 2008-09") counts as robust only if the
   real-time alarm month matches within ±1 month; otherwise the paper
   claim is downgraded to revised-data-only.

## 2026-07-12 — varbench outcome: **Outcome C fired** (M4 resolution)

The pre-registered decision rule (M0 entry above) is now resolved
against the completed grid_v4_varbench run (500 reps, identical seed
blocks, empirical FARs 4.2–6.8%).

**Numbers (raw_var_cusum, variance ×1.5, T = 500).** D_raw = 0.996 /
0.560 / 0.102 at SNR 0.1 / 0.5 / 2.0; empirical FAR ≈ 6%; the composite
D_comp = 0.82 / 0.87 / 0.91.

- **Outcome A** (D_raw within 5 pp of FAR at *every* SNR) — FALSIFIED:
  0.996 and 0.560 are far above FAR (only the SNR-2.0 cell, 0.102 vs
  6.0%, is near chance).
- **Outcome B** (D_raw within 10 pp of D_comp at *every* SNR) —
  FALSIFIED: off by 18 / 31 / 81 pp.
- **Outcome C (mixed) — FIRED.** The raw variance CUSUM is strongly
  SNR-dependent: it *beats* the composite when observation noise
  dominates (0.996 at SNR 0.1) and collapses to chance as the latent
  signal grows (0.102 at SNR 2.0). Mechanism: as SNR rises the latent
  state's variance dominates Var(Y), so a ×1.5 change in the shrinking
  noise component is masked by state-driven autocorrelation.

**Sharpening sub-finding (whitening rung).** The middle rung
`arima_var_cusum` on the same ×1.5 break is essentially flat across SNR
(0.90 / 0.94 / 0.87) and tracks the composite step for step —
prewhitening recovers the full second-moment advantage, and the latent
*state estimate* adds little the ARIMA residuals do not. Reported per
the honest-outcome clause (SPEC §8): for second moments the edge is
"whitening, not state estimation." Under t₅ the composite collapses
(×1.5: 0.16) while the raw (0.43) and ARIMA (0.74) variance rungs hold —
the plain z²/e² max-over-arms statistic keeps the tail signal the
per-time composite standardization discards.

**Real-data resolution (M3).** raw_var_cusum was added to the INDPRO /
GDP / GS10 monitoring and (because it alarmed on revised INDPRO at
2008-09) to the ALFRED vintage protocol. It catches every headline
crisis — GFC (INDPRO 2008-09, GDP 2009Q2), COVID (GDP 2020Q2), Volcker
(GS10 1980-02) — and its real-time vintage timing is *identical* to the
composite for both GFC (2008-12 vintage, data month 2008-09) and COVID
(2020-04 vintage, 2020-03). So the real-data *uniqueness* claim is
downgraded per the M0 spirit: the crises are not uniquely detected by
the latent layer. What survives is (i) the composite's clean association
profile — significant, low-stray (INDPRO p = 0.007 vs raw_var p = 0.56,
which strays on quieting alarms and washes out; GS10 raw_var 9 alarms /
8 stray) — and (ii) the simulation-calibrated ×1.5 subtlety threshold at
high SNR, where only the whitened rungs see the break. The 180-month
training window breaks *both* variance detectors (composite and raw_var
each 14 alarms), isolating the sensitivity to the second-moment
statistic rather than the composite machinery.

**Paper patch (M4).** PAPER_DRAFT.md updated: abstract and §1
contribution 2 reframed to the prewhitening finding; §5 rewritten as the
whitening ladder with the raw/ARIMA/latent table and a T-sweep note;
§8.2 gains the t₅ raw/ARIMA rungs; §9 gains raw_var_cusum rows, the
apples-to-apples vintage comparison, and the downgraded uniqueness
language; §10 recipe becomes "whiten, then run the variance CUSUM." All
editorial `[EDIT]` brackets resolved; propositions numbered; test count
77 → 84.

## 2026-07-12 — full `make all` reproduction gate + stale-artifact correction

Ran the entire deterministic suite from the committed code (`make all`,
~2 h, green, 0 errors) as the clean-clone reproducibility gate. Findings,
with the adjudication trail:

**Result.** 7 of 12 experiment groups regenerate BYTE-IDENTICAL, including
every paper-quoted source: grid_v1, grid_v2 (T + misspec), grid_v3_robust,
grid_v3c, grid_v4, exp04, exp06. grid_v4_varbench was separately confirmed
bit-identical on re-run from pinned seeds (all 60 rows, thresholds, FARs).

**5 groups had STALE committed artifacts** — parquets generated by an
earlier (pre-git) version of the composite code and never regenerated
before the initial commit bundled them with the newer code:
- exp01 / exp02 / exp03: only `lsc_composite` rows move (≤0.036); the
  single-statistic detectors and exp02's cited `lsc_state_cusum` numbers
  are unchanged.
- grid_v3b_exceedance: only `lsc_composite_robust2` moves (both arenas;
  ×1.5 SNR 0.5 0.058→0.582, ×1.5 t₅ 0.034→0.208, ×3 t₅ 0.506→0.972); the
  default `lsc_composite` rows in the same grid are byte-identical.
- m2_param_recovery: `markov_switching` rows differ at ~1e-13 (float /
  BLAS-threading epsilon, not the RNG issue; rounds identical).

**Git archaeology.** No commit in this repo altered the composite score:
`features.py` and `alarms.py` (the exceedance features and the per-t
null-standardization) are byte-identical to the initial commit; M1's only
touch to `detectors.py` ADDED the two variance-detector factories and left
`make_composite_detector` untouched. The change that moved robust2
therefore predates `git init` (the pre-git exp05 "pooled scale → per-t
standardization" fix logged in §8.4), and the stale parquet slipped in
because no test pins grid_v3b's output.

**Hand adjudication (not a re-run).** On one ar1_snr0.5 path we recomputed
the `tail_exceedance` feature's per-t null center/IQR manually and matched
the pipeline exactly (max |Δ| = 0.0), hand-standardized its score at
several t (e.g. t=400: (14.317−0.805)/1.520 = 8.89), and confirmed the
max-over-features composite equals `det(Y)` at every post-train t
(max |Δ| = 0.0), with `tail_exceedance` the dominating feature post-break
(z 2.1 → 8.9 → 19.2). Current code (0.582) is arithmetically correct; the
committed 0.058 was the pre-fix pooled-scale output.

**Actions.** (1) Committed the regenerated artifacts so `make all` is
idempotent and every number is backed by the current code. No
paper-QUOTED number changes. (2) Corrected §8.3(ii): the exceedance
indicator is *diluted, not dead*, inside the composite (0.58 / 0.21, even
edging the e²-composite under t₅); its per-t score reaches z ≈ 19, so the
old "bounded increments can't reach the ratios" mechanism was an artifact
of the pre-§8.4 pooled-scale build. The standalone recommendation (iii)
stands — now justified by FAR-budget dilution across ~10 features, not by
the composite crushing the feature.

## 2026-07-13 — M0 (R1 referee-hardening): M1/M2 decision rules PRE-REGISTERED before any R1 artifact

Registered per `SPEC_latent_state_change.md` R1 round §M0, before
generating any R1 artifact (no exp07 file, no grid_v5 config, no
grid_v5/v6 parquet exists at commit time — verified: `git status` clean,
`ls experiments/exp07* configs/grid_v5* configs/grid_v6*` empty). The
paper adopts whichever branch the results select; no post-hoc
reinterpretation.

**M1 rule — ARMA(1,1) equivalence [GATE].** Let ρ̄ = median Pearson
correlation between the ARIMA standardized-residual series and the
Kalman standardized-innovation series over the post-burn-in monitoring
region, across null (no-break) paths, at each SNR ∈ {0.1, 0.5, 2.0},
T = 500, ≥200 paths.

- **A1 — Equivalent (expected): ρ̄ ≥ 0.95.** The two rungs are the same
  filter up to estimation error. §5 is reframed as a two-rung ladder
  (raw vs. whitened) with the ARMA(1,1) reduced-form equivalence stated
  as theory and the grid reported as numerical confirmation. Practical
  recipe (§10) sharpens to "ARMA whitening suffices; the state-space
  layer is not required for second-moment monitoring."
- **A2 — Not equivalent: ρ̄ < 0.95. STOP.** For an AR(1)+white-noise
  DGP the steady-state Kalman innovations and the ARMA(1,1) innovations
  are the same linear innovations of the same Gaussian process; a large
  gap implies an estimator bug (candidate causes: ARIMA order
  misselection — especially AIC choosing a differencing order —
  non-frozen parameters, standardization mismatch, burn-in
  contamination). Diagnose the estimator before running anything else.
  Do NOT proceed to M2–M4.

Regression guard: `test_arma_kalman_equivalence` asserts ρ̄ ≥ 0.95 with
TRUE (not estimated) parameters, where agreement should be near-exact.

**M2 rule — q-break (state-innovation variance) ladder.** Let R_q =
ladder detection rates (raw / ARIMA-whitened / Kalman-whitened) for a
state-innovation-variance break, and R_r = the published
observation-noise (r) results at ×1.5, T = 500: raw 1.00 / 0.56 / 0.10,
whitened ≈ 0.90 / 0.94 / 0.87 across SNR 0.1 / 0.5 / 2.0. Convention:
the q-break scales the state-innovation **SD** by the same `vol_mult`
the r-break scales the observation-noise SD by (verified: the existing
`variance` break multiplies obs-noise std, per `obs_noise_scale_path`
and the `BreakSpec` docstring), so "×1.5" means SD×1.5 in BOTH channels
— the only setting under which the two-channel comparison is
meaningful.

- **B1 — Ordering survives:** raw remains strongly SNR-dependent and
  whitened remains approximately flat → the §5 claim generalizes; report
  both break channels; paper materially stronger.
- **B2 — Ordering inverts or flattens:** the "prewhitening beats raw"
  result is specific to white-component (r) breaks → §5's claim is
  scoped to r-breaks, and the paper MUST say so, because every
  real-data motivating event (Great Moderation, crisis volatility) is a
  q-break. §9–§10 framing revised accordingly (honest-outcome clause).
- **B3 — Mixed / SNR-dependent in a new way:** report the full
  two-channel ladder and characterize.

All M1/M2 outcomes are publishable. The fired branch is logged with
numbers when M1/M2 resolve.

## 2026-07-13 — M1 [GATE] RESOLVED: **A1 fires** — the ARIMA and Kalman rungs are the same filter

`experiments/exp07_arma_equivalence.py` (≥200 null paths, T=500, SNR ∈
{0.1, 0.5, 2.0}); theory + machine-precision cross-checks in
`experiments/THEORY.md` (§"ARMA(1,1) equivalence") and
`lsc.theory.arma11_representation`; regression guards
`test_arma11_riccati_identities` + `test_arma_kalman_equivalence`.

**Structural theory (machine precision).** The observable Y has an exact
ARMA(1,1) reduced form with AR parameter φ, MA parameter θ, innovation
variance σ_ε². Two identities hold to < 1e-12: σ_ε² = F (the Riccati
innovation variance of Proposition 1) and θ = ρ = φ(1−K) (the
innovation-mean decay rate). Hence the steady-state Kalman innovations
and the ARMA(1,1) innovations are the same linear innovations of the
same Gaussian process.

**Empirical (deliverable `paper_assets/arma_equivalence.csv`).**
- TRUE parameters: median Pearson ρ = 1.000000 at every SNR; max|Δ|
  between the (independent) statsmodels ARMA filter and the hand-written
  steady-state Kalman recursion ≈ 1e-9 (1.5e-7 at SNR 0.1). Near-exact,
  as predicted — the wedge is entirely estimation error.
- ESTIMATED parameters (ladder's real operating condition): ρ̄ = 0.9914
  (medians 0.9947 / 0.9914 / 0.9880 at SNR 0.1 / 0.5 / 2.0), ≥ 0.95 ⇒
  **A1**. Forcing the ARIMA order to the true (1,0,1) tightens this to
  ρ̄ = 0.9995 (0.9992 / 0.9995 / 0.9996), so AIC order-selection is the
  sole material wedge.

**Order-selection finding (SPEC M1 asked for this explicitly).** AIC over
the benchmark grid selects the true (1,0,1) on only 16.5% / 9% / 5.5% of
paths; it prefers (1,0,0) at SNR 0.1 (36.5%) and the differencing order
(0,1,1) at SNR 0.5 / 2.0 (63.5% / 71%). This is a near-unit-root artifact
(φ = 0.95), NOT a code bug: at φ≈1 both IMA(1,1) and AR(1) approximate
the ARMA(1,1) closely enough to keep ρ̄ ≥ 0.95, and the AIC-selected
model is a legitimate whitener. Not changing the published varbench
ARIMA rung (AIC selection is the benchmark's design and the change would
ripple into grid_v4); reported as-built, with the forced-(1,0,1) column
alongside and a scope note that mis-differencing would bite at small φ
(picked up in M3).

**Paper action (M6).** §5 reframed as a **two-rung ladder** (raw vs.
whitened); ARMA(1,1) equivalence stated as theory (Appendix B), grid as
numerical confirmation; §10 recipe sharpened to "ARMA whitening
suffices; the state-space layer is not required for second-moment
monitoring." GATE PASSED — proceeding to M2.

## 2026-07-13 — M2 RESOLVED: **B2 fires** — prewhitening's advantage is specific to observation-noise (r) breaks

`configs/grid_v5_qbreak.yaml` (state-innovation q-break, SD-scaled to
match the r-break convention; 500 reps, T=500, identical seed blocks);
new DGP kind `state_var` in `lsc/dgp/breaks.py` +
`state_noise_scale_path`, wired into AR1StateDGP and LocalLevelDGP
(tests `test_state_var_break_scales_state_innovation_sd`,
`test_state_var_null_matched_and_reproducible`). Assembler
`experiments/qbreak_ladder.py` → two-channel `ladder_table.csv`
(`break_channel` r|q). q-break FARs 3.6–6.8% (calibrated).

**The ladder ordering INVERTS across channels** (detect rate at the
discriminating ×1.5 break, T=500, SNR 0.1/0.5/2.0):

| chan | rung  | 0.1  | 0.5  | 2.0  |
|------|-------|------|------|------|
| r    | raw   | 1.00 | 0.56 | 0.10 |  (falls with SNR)
| r    | ARIMA | 0.90 | 0.94 | 0.87 |  (flat — WINS at high SNR)
| q    | raw   | 0.09 | 0.21 | 0.23 |  (RISES with SNR — WINS everywhere)
| q    | ARIMA | 0.03 | 0.10 | 0.16 |  (below raw at every SNR)

For q-breaks the raw variance CUSUM ≥ the whitened rung at every SNR
(and the composite): ×1.5 raw beats ARIMA 0.09/0.21/0.23 vs
0.03/0.10/0.16; ×3 raw 0.72/0.96/0.96 vs ARIMA 0.26/0.79/1.00 (ARIMA
only catches up at the ×3 SNR-2.0 ceiling). And the raw rung's
SNR-dependence FLIPS sign: r-break raw falls 1.00→0.10, q-break raw
rises 0.09→0.23.

**Mechanism (clean and symmetric).** An r-break lives in the WHITE
component of Y; state autocorrelation masks it in raw z², and whitening
removes exactly that autocorrelation → whitening wins, and raw fades as
SNR rises and the white component shrinks. A q-break lives in the STATE
(autocorrelated) component; it inflates the marginal variance of Y — which
dominates at high SNR — so raw z² sees it directly, while whitening
strips out the state-carried signal → raw wins, and raw strengthens as
SNR rises. A q-break also shifts the ARMA(1,1) MA parameter θ (verified:
θ 0.793→0.710 at ×1.5, SNR 0.5), changing the autocorrelation structure,
whereas an r-break changes only the marginal variance.

**Decision: B2** (pre-registered). "Prewhitening beats raw" is specific
to observation-noise breaks. **Honest-outcome clause invoked**: §5 and
§10 are scoped to r-breaks; the paper states plainly that for the
state-innovation (q) breaks that motivate the empirical section — Great
Moderation, crisis volatility are q-like shifts — a raw variance CUSUM is
at least as good as whitening at every SNR. This also *explains* the §9
real-data result (raw_var_cusum's crisis timing is identical to the
composite's): real crises are q-channel breaks, exactly where B2 says raw
matches whitened. Abstract revised (B2 changes the headline). Quieting
(×⅔ i.e. q reduced) is undetectable by every rung (≤0.07, at FAR) — a
low-q state contributes too little to Y to register its own reduction.
Under t₅ the q-break ×3 ranking holds with raw most robust (raw 0.93,
ARIMA 0.66, composite 0.48, tail 0.79). All numbers in
`paper_assets/grid_v5_qbreak_results.parquet` + `ladder_table.csv`.

## 2026-07-13 — M3: φ sweep confirms μ∞ sorts detection — with a boundary condition

`configs/grid_v6_phisweep.yaml` (level 1σ/3σ × φ∈{0.5,0.8,0.95,0.99} ×
SNR∈{0.1,0.5,2.0}, T=500, 500 reps; SNR held fixed across φ by q =
SNR·(1−φ²)); analysis `experiments/phisweep_analyze.py`. Deliverables
`paper_assets/grid_v6_phisweep_muinf.csv`, `grid_v6_muinf_scatter.png`
(headline theory-verification figure).

**μ∞ sorts detection: Spearman(μ∞, innovation-CUSUM detect) = 0.942**
across all 24 cells. Because μ∞ = δ(1−φ)/((1−φ(1−K))√F) is increasing in
(1−φ), the innovation CUSUM ESCAPES the fast-or-never regime at low φ:
3σ detection is 0.98/1.00 at φ=0.5 and 0.93/0.97 at φ=0.8 (μ∞ ≥ 0.69,
FAST) but collapses to 0.65/0.55 at φ=0.95 and 0.63/0.30 at φ=0.99
(μ∞ ≤ 0.48, "never"). Fast-regime cells (μ∞ ≥ k=0.5) detect 0.83–1.00;
never-regime cells (μ∞ < 0.5) detect 0.07–0.67. The crossover sits
between φ=0.8 and φ=0.95, exactly at the predicted k=0.5 boundary.

**Boundary condition for the paper.** Proposition 1's fast-or-never is
therefore not an unqualified claim: filtering fails for LEVEL detection
only when the state is PERSISTENT (φ ≳ 0.9) — which is the empirically
relevant case (φ=0.95 is the paper's baseline). At low persistence the
innovation CUSUM is competitive with raw. raw_cusum, by contrast, detects
3σ levels at 0.96–1.00 across ALL φ (φ-robust, slow-but-sure), so the
φ-dependence is specific to the filtered detector — a direct corroboration
of the mechanism.

**Honest caveat (the one off-trend cell).** At the near-unit-root corner
φ=0.99, SNR 0.1, 3σ, detection is 0.63 despite μ∞=0.21 (deep "never") —
above the μ∞ trend. Mechanism: at φ→1 the adaptation transient (decay
rate ρ=φ(1−K)) is so long that its accumulated mass triggers the CUSUM
during the transient even though the asymptotic drift is negligible —
i.e. the "fast" branch of fast-or-never is itself stronger when the
transient is long. So μ∞ sorts the ASYMPTOTIC regime cleanly (Spearman
0.94) but finite-sample detection at very high φ also carries a transient
contribution; reported as such, not smoothed over. Proposition 1 holds;
the caveat is that total detection = transient mass + post-transient
tail, and μ∞ governs only the second.

## 2026-07-13 — M7 PRE-REGISTERED: φ × q-break cross-grid (amplification test), before any grid_v8 cell runs

Registered before implementing `configs/grid_v8_phiqbreak.yaml` (verified:
no grid_v8 config/parquet at commit time). Extends M2 (q-break) × M3 (φ)
into a genuine cross: hold the shock variance q and obs variance r FIXED
and sweep φ, so the state's stationary variance q/(1−φ²) — and hence the
signal-to-noise ratio SNR(φ) = (q/r)/(1−φ²) — rises as the **1/(1−φ²)
amplification factor**. Anchor q = 0.04875, r = 1.0 (so φ = 0.95 gives
SNR 0.5, matching the body arena); φ ∈ {0.1, 0.5, 0.7, 0.85, 0.95, 0.99}
gives induced SNR {0.049, 0.065, 0.096, 0.176, 0.5, 2.45}. q-break ×1.5
and ×3; rungs raw_var_cusum / arima_var_cusum / lsc_composite; 500 reps,
T = 500, identical seed blocks.

**Pre-registered prediction (falsifiable).** The raw rung's *advantage*
on q-breaks — Δ(φ) = detect(raw_var) − detect(arima_var) (and vs
composite) — is (i) monotonically increasing in φ and (ii) → 0 as φ → 0,
(iii) tracking the amplification 1/(1−φ²) (equivalently the induced SNR).
Mechanism: at φ → 0 the observable is white (AR(0)), so whitening is a
no-op and the q-break barely moves Y's variance → raw ≈ whitened ≈ FAR,
Δ ≈ 0; as φ → 1 the 1/(1−φ²) amplification inflates the state's share of
Y's variance, which a raw z² statistic sees directly while whitening
strips it out → Δ large. Secondary consistency check: Δ at each cell
should match the M2 SNR-sweep (grid_v5, fixed φ = 0.95) at the *same
induced SNR* — i.e. the effect operates through amplified SNR, not φ per
se. **Falsifiers:** Δ(φ) non-monotone; Δ(0.1) materially > 0; Δ present
without autocorrelation; Δ inconsistent with the M2 SNR-sweep at matched
SNR. Outcome logged with numbers when resolved; all outcomes publishable.

## 2026-07-13 — M7 RESOLVED: prediction CONFIRMED IN PART, FALSIFIED IN PART (honest mixed)

`configs/grid_v8_phiqbreak.yaml` (fixed q=0.04875, r=1; φ ∈ {0.1, 0.5,
0.7, 0.85, 0.95, 0.99}; induced SNR 0.049→2.45); analysis
`experiments/phiqbreak_analyze.py` → `grid_v8_phiqbreak_summary.csv`,
`grid_v8_phiq_amplification.png`. Raw advantage Δ(φ) = detect(raw_var) −
detect(arima_var):

| φ (SNR)     | amp 1/(1−φ²) | Δ ×1.5 | Δ ×3 |
|-------------|--------------|--------|------|
| 0.10 (0.05) | 1.01         | 0.00   | 0.34 |
| 0.50 (0.07) | 1.33         | 0.02   | 0.53 |
| 0.70 (0.10) | 1.96         | 0.04   | 0.53 |
| 0.85 (0.18) | 3.60         | 0.10   | 0.20 |
| 0.95 (0.50) | 10.3         | 0.11   | 0.17 |
| 0.99 (2.45) | 50.3         | 0.07   | 0.30 |

**CONFIRMED (subtle ×1.5 break):** (i) Δ → 0 as φ → 0 (Δ = 0.000 at
φ = 0.1: on a white observable, whitening is a no-op and raw ≈ whitened).
(ii) Δ rises with the amplification (Spearman(amp, Δ) = 0.83). (iii)
Secondary consistency check PASSES cleanly: the φ-swept Δ equals the M2
SNR-swept Δ (grid_v5, fixed φ = 0.95) at the *same induced SNR* — 0.11 vs
0.11 at SNR 0.5, 0.07 vs 0.07 at SNR ≈ 2 — so the effect genuinely
operates through the amplified SNR, i.e. the φ-sweep and the SNR-sweep are
one experiment. The state-variance amplification IS the mechanism for the
subtle break.

**FALSIFIED (strong form):** (a) Δ is NOT monotone to φ = 1: for ×1.5 it
peaks at φ = 0.95 (0.11) and DIPS at φ = 0.99 (0.07), because at the
near-unit-root the raw detector's own baseline degrades (calibrated
threshold 1829 at φ = 0.99 vs 275 at φ = 0.95 — the same nonstationarity
penalty as the M4 local-level arena), an opposing effect the simple
1/(1−φ²) law omits. (b) For the coarse ×3 break the prediction fails
outright: Δ is large at every φ including φ = 0.1 (0.34) and
Spearman(amp, Δ) = −0.60 (negative). Raw z² detects a gross q-break
regardless of amplification, while ARIMA whitening attenuates it at all φ.

**Mechanism correction (order check, 80 prefixes/φ).** The ×3 low-φ
ARIMA underperformance is NOT over-differencing: at φ = 0.1/0.5 AIC picks
the *stationary* AR(1) (1,0,0) on 53/51 of 80 prefixes (differencing
(0,1,1) only dominates at φ = 0.99, 58/80). So the ARIMA rung's deficit
is the whitening ITSELF removing the state-shock variance signal that raw
z² retains — the core B2 mechanism — present at all φ, not an
order-selection artifact. This strengthens B2: prewhitening strips the
state-carried signal for q-breaks of any size; the 1/(1−φ²) amplification
only modulates *how visible* the residual raw advantage is on the subtle,
detectability-marginal break, and recedes at the unit root where raw's
baseline breaks down.

**Net.** The user's amplification prediction holds for the regime it
describes (subtle breaks, φ away from 1) and is the confirmed driver
there; it is not a universal monotone law (coarse breaks, and the
near-unit-root boundary, break it). Reported as such in §5. Abstract
carries the robust claim (raw ≥ whitened on q-breaks at every SNR)
without the over-strong tracking clause.

## 2026-07-13 — M4: local-level (RW-state) arena — DEMONSTRATED, not dismissed

`configs/grid_v7_llevel.yaml` (LocalLevelDGP, SNR=q/r ∈ {0.1,0.5,2.0},
level 1σ/3σ + variance ×1.5/×3, T=500, 500 reps); analysis
`experiments/llevel_analyze.py` → `grid_v7_llevel_summary.csv`,
`grid_v7_llevel_degeneracy.png`. The old §3 one-clause dismissal of the
arena is replaced by two demonstrated facts.

**(1) LEVEL detection is degenerate for EVERY method.** At 3σ all five
methods sit at the 5% FAR (raw_cusum 0.07–0.10, Kalman-innov 0.04–0.15,
raw_var 0.06–0.10, ARIMA-var 0.04–0.07, composite 0.07). A level break in
a random-walk state is absorbed by a well-specified filter as one large
ordinary innovation (no sustained signal), and the raw-Y CUSUM has no
fixed baseline — its calibrated threshold is 1533 / 2091 / 2372 (vs
O(10–100) in the AR(1) arena) and it still calibrates hot (FAR 7.8% at
SNR 0.1). No common null exists to rank against; the arena cannot rank
level detectors. The 2026-07-10 dismissal was CORRECT, now with evidence.

**(2) VARIANCE detection is NOT degenerate — but whitening becomes
MANDATORY.** On the ×1.5 variance break the raw z² CUSUM is at chance
(0.06 at every SNR; threshold 10374 / 17463 / 22297 — astronomically
large because z² on a nonstationary RW has no meaningful scale), while
the ARIMA-differencing rung detects (0.71 / 0.84 / 0.58) and the Kalman
composite detects (0.97 / 0.89 / 0.68). At ×3: raw_var stays at chance
(0.06–0.08) while ARIMA-var 0.97–1.00 and composite 0.98–0.99. This is
the exact COMPLEMENT of the AR(1) r-break result (M2): there a raw
variance CUSUM could win when observation noise dominates; here, where Y
is nonstationary, prewhitening is not merely helpful but required — the
raw statistic has no stationary baseline to calibrate against. It
reinforces, rather than contradicts, the paper's central prewhitening
message. FARs 4.8–7.8% (raw_cusum hot at low SNR, as noted).

**Paper action (M6).** §3 dismissal replaced by a short subsection (or
§8.4 expansion) stating (1) and (2) with the degeneracy figure; the
arena now carries a positive finding (whitening-mandatory-under-
nonstationarity), not a hand-wave.
